import os
import time
import math
import random
import psutil
import numpy as np
import torch

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)

# =========================
# Configuration
# =========================
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
DATA_PATH = "dataset/BGL/BGL.log"
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Few-shot prompt examples pulled only from train
MAX_FEW_SHOT_PER_CLASS = 8

# Speed control
MAX_TEST_SAMPLES = None   # set e.g. 200 for faster debugging

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# =========================
# Utilities
# =========================
def get_ram_usage_mb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_gpu_peak_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return None


def safe_roc_auc(y_true, y_score):
    if len(set(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_score)
    except Exception:
        return None


# =========================
# BGL parsing
# =========================
def parse_bgl_line(line):
    """
    Parse BGL line.

    Label:
      '-' -> 0 (normal)
      anything else -> 1 (anomaly)

    Based on your examples:
      0: label
      1: unix timestamp
      2: date
      3: node
      4: datetime
      5: node
      6: RAS
      7: KERNEL
      8: severity (INFO/FATAL/...)
      9+: message

    We remove the ground-truth label token and keep the useful tail.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split()
    if len(parts) < 10:
        return None

    label = 0 if parts[0] == "-" else 1
    message = " ".join(parts[8:]).strip()

    if not message:
        return None

    return {
        "label": label,
        "message": message,
        "raw": line,
    }


def load_bgl_dataset(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parsed = parse_bgl_line(line)
            if parsed is not None:
                records.append(parsed)

    if not records:
        raise ValueError("No valid records parsed from dataset.")

    return records


# =========================
# Few-shot prompt building
# =========================
def build_balanced_few_shot_examples(train_records, max_per_class=8):
    normal = [r for r in train_records if r["label"] == 0]
    anomaly = [r for r in train_records if r["label"] == 1]

    random.shuffle(normal)
    random.shuffle(anomaly)

    selected = normal[:max_per_class] + anomaly[:max_per_class]
    random.shuffle(selected)
    return selected


def build_prompt(few_shot_examples, log_message):
    instruction = (
        "You are a system log anomaly detector.\n"
        "Classify each log as either NORMAL or ANOMALY.\n"
        "Use the examples to infer the pattern.\n\n"
    )

    examples = []
    for ex in few_shot_examples:
        label_text = "ANOMALY" if ex["label"] == 1 else "NORMAL"
        examples.append(f"Log: {ex['message']}\nLabel: {label_text}")

    example_block = "\n\n".join(examples)

    prompt = (
        instruction
        + example_block
        + f"\n\nLog: {log_message}\nLabel:"
    )
    return prompt


# =========================
# Token / scoring helpers
# =========================
def move_to_model_device(batch, model):
    """
    Handles both regular model.device and hf device_map=auto cases.
    """
    if hasattr(model, "device"):
        target = model.device
    else:
        target = torch.device(DEVICE)

    return {k: v.to(target) for k, v in batch.items()}


def get_label_token_ids(tokenizer, label_text):
    """
    Encode label string without special tokens.
    We use a leading space because causal LMs are space-sensitive.
    """
    ids = tokenizer.encode(" " + label_text, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Empty tokenization for label: {label_text}")
    return ids


def continuation_logprob(model, tokenizer, prompt, continuation_token_ids):
    """
    Compute log P(continuation | prompt) by teacher forcing.

    We feed:
      [prompt_tokens] + [continuation_tokens]

    and sum the log-probabilities assigned to the continuation tokens.
    """
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    full_ids = prompt_ids + continuation_token_ids

    input_ids = torch.tensor([full_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    batch = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    batch = move_to_model_device(batch, model)

    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits  # [1, seq_len, vocab]

    log_probs = torch.log_softmax(logits, dim=-1)

    total_logprob = 0.0
    prompt_len = len(prompt_ids)

    for i, token_id in enumerate(continuation_token_ids):
        pos = prompt_len + i
        # token at pos is predicted by logits at pos-1
        token_logprob = log_probs[0, pos - 1, token_id].item()
        total_logprob += token_logprob

    return total_logprob


def score_labels(model, tokenizer, prompt):
    """
    Compare P(NORMAL | prompt) vs P(ANOMALY | prompt).
    Returns:
      pred_label: 0/1
      anomaly_prob: float in [0,1]
      normal_logprob
      anomaly_logprob
    """
    normal_ids = get_label_token_ids(tokenizer, "NORMAL")
    anomaly_ids = get_label_token_ids(tokenizer, "ANOMALY")

    normal_lp = continuation_logprob(model, tokenizer, prompt, normal_ids)
    anomaly_lp = continuation_logprob(model, tokenizer, prompt, anomaly_ids)

    # stable normalization for 2-way comparison
    max_lp = max(normal_lp, anomaly_lp)
    normal_p = math.exp(normal_lp - max_lp)
    anomaly_p = math.exp(anomaly_lp - max_lp)
    anomaly_prob = anomaly_p / (normal_p + anomaly_p)

    pred_label = 1 if anomaly_lp > normal_lp else 0

    return pred_label, anomaly_prob, normal_lp, anomaly_lp


# =========================
# Evaluation
# =========================
def evaluate_predictions(y_true, y_pred, y_score):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred, zero_division=0),
        "roc_auc": safe_roc_auc(y_true, y_score),
    }
    return results


# =========================
# Main
# =========================
def main():
    print("=" * 60)
    print("BGL Anomaly Detection with DeepSeek (Log-Prob Scoring)")
    print("=" * 60)
    print(f"Model:  {MODEL_ID}")
    print(f"Device: {DEVICE}")
    print(f"Data:   {DATA_PATH}")

    # -------------------------
    # Load dataset
    # -------------------------
    records = load_bgl_dataset(DATA_PATH)
    labels = [r["label"] for r in records]

    print(f"\nParsed records: {len(records)}")
    print(f"Normal logs:    {sum(1 for x in labels if x == 0)}")
    print(f"Anomaly logs:   {sum(1 for x in labels if x == 1)}")

    # -------------------------
    # Train/test split
    # -------------------------
    train_records, test_records = train_test_split(
        records,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    if MAX_TEST_SAMPLES is not None:
        test_records = test_records[:MAX_TEST_SAMPLES]

    print(f"\nTrain size: {len(train_records)}")
    print(f"Test size:  {len(test_records)}")

    few_shot_examples = build_balanced_few_shot_examples(
        train_records,
        max_per_class=MAX_FEW_SHOT_PER_CLASS,
    )

    print(f"Few-shot examples used: {len(few_shot_examples)}")
    print(f"  NORMAL:  {sum(1 for x in few_shot_examples if x['label'] == 0)}")
    print(f"  ANOMALY: {sum(1 for x in few_shot_examples if x['label'] == 1)}")

    # -------------------------
    # Load tokenizer/model
    # -------------------------
    print("\nLoading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model.eval()

    # -------------------------
    # Inference
    # -------------------------
    y_true = []
    y_pred = []
    y_score = []
    latencies = []

    debug_examples = []

    print("\nRunning inference on test set...")
    overall_start = time.time()

    for idx, record in enumerate(tqdm(test_records)):
        prompt = build_prompt(few_shot_examples, record["message"])

        start = time.time()
        pred_label, anomaly_prob, normal_lp, anomaly_lp = score_labels(
            model,
            tokenizer,
            prompt,
        )
        latency = time.time() - start

        y_true.append(record["label"])
        y_pred.append(pred_label)
        y_score.append(anomaly_prob)
        latencies.append(latency)

        if len(debug_examples) < 12:
            debug_examples.append({
                "message": record["message"],
                "true": record["label"],
                "pred": pred_label,
                "anomaly_prob": anomaly_prob,
                "normal_lp": normal_lp,
                "anomaly_lp": anomaly_lp,
            })

    overall_time = time.time() - overall_start

    # -------------------------
    # Metrics
    # -------------------------
    results = evaluate_predictions(y_true, y_pred, y_score)

    # -------------------------
    # Resource usage
    # -------------------------
    ram_mb = get_ram_usage_mb()
    gpu_mb = get_gpu_peak_memory_mb()

    # -------------------------
    # Output
    # -------------------------
    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"Accuracy:   {results['accuracy']:.4f}")
    print(f"Precision:  {results['precision']:.4f}")
    print(f"Recall:     {results['recall']:.4f}")
    print(f"F1-score:   {results['f1']:.4f}")
    if results["roc_auc"] is None:
        print("ROC-AUC:    undefined")
    else:
        print(f"ROC-AUC:    {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(results["confusion_matrix"])

    print("\nClassification Report:")
    print(results["classification_report"])

    print("\n" + "=" * 60)
    print("RUNTIME / MEMORY")
    print("=" * 60)
    print(f"Average latency per log: {np.mean(latencies):.4f} s")
    print(f"Total inference time:    {overall_time:.2f} s")
    print(f"Process RAM usage:       {ram_mb:.2f} MB")
    if gpu_mb is not None:
        print(f"Peak GPU memory:         {gpu_mb:.2f} MB")

    print("\n" + "=" * 60)
    print("DEBUG EXAMPLES")
    print("=" * 60)
    for i, ex in enumerate(debug_examples, 1):
        print(f"\n[{i}]")
        print(f"Message:       {ex['message']}")
        print(f"True label:    {ex['true']}")
        print(f"Pred label:    {ex['pred']}")
        print(f"Anomaly prob:  {ex['anomaly_prob']:.4f}")
        print(f"logP(NORMAL):  {ex['normal_lp']:.4f}")
        print(f"logP(ANOMALY): {ex['anomaly_lp']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
