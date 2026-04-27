import random
import re
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================
# CONFIG
# ==============================================================
MODEL_ID              = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
NORMAL_FILE           = "dataset/normal.log"
ABNORMAL_FILE         = "dataset/abnormal.log"
BATCH_SIZE            = 8      # Safe for V100 16 GB; raise to 16 if stable
MAX_LENGTH            = 1024
MAX_NEW_TOKENS        = 3      # Only "0" or "1" needed
OUTPUT_MISCLASSIFIED  = "misclassified.txt"
RANDOM_SEED           = 42


# ==============================================================
# PROMPT
# Few-shot examples chosen carefully:
#   - NORMAL examples are INFO/WARNING only (no FATAL severity)
#   - ABNORMAL examples cover the most commonly confused patterns
#   - Kept to 3 per class to stay token-efficient
# ==============================================================
SYSTEM_PROMPT = """\
You are a BGL (Blue Gene/L) log anomaly classifier.

Classify each log line as:
0 = NORMAL
1 = ABNORMAL

Rules:
0 NORMAL:
- INFO or WARNING severity messages
- Corrected hardware errors (cache parity corrected, DDR errors corrected, CE syms)
- Alignment exceptions
- Core file generation
- Routine retry or retransmission messages
- Register dumps that are part of a normal diagnostic sequence

1 ABNORMAL:
- rts panic
- kernel terminated
- Lustre mount FAILED
- Link has been severed
- Connection reset by peer / Connection timed out
- data TLB error interrupt
- data storage interrupt
- Fatal errors that stop execution

NORMAL examples (label 0):
Log: 1117838978 2005.06.03 R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected
Answer: 0

Log: 1118271740 2005.06.08 R03-M1-N9-C:J09-U11 RAS KERNEL INFO 1 ddr errors(s) detected and corrected on rank 0, symbol 25, bit 1
Answer: 0

Log: 1132236972 2005.11.17 R72-M1-N6-C:J04-U01 RAS KERNEL INFO 26741629 torus sender z- retransmission error(s) detected and corrected over 268 seconds
Answer: 0

ABNORMAL examples (label 1):
Log: 1121115817 2005.07.11 R01-M0-N1-C:J09-U11 RAS KERNEL FATAL rts panic! - stopping execution
Answer: 1

Log: 1124071359 2005.08.14 R21-M0-N8-I:J18-U11 RAS APP FATAL ciod: Error reading message prefix after LOAD_MESSAGE on CioStream socket to 172.16.96.116:42213: Link has been severed
Answer: 1

Log: 1126202752 2005.09.08 R01-M1-N4-I:J18-U11 RAS KERNEL FATAL Lustre mount FAILED : bglio23 : point /p/gb1
Answer: 1

Return exactly one character: 0 or 1.
"""


def build_prompt(log_line: str) -> str:
    return f"{SYSTEM_PROMPT}\nLog: {log_line}\nAnswer:"


# ==============================================================
# MODEL LOADING
# ==============================================================
def load_model() -> tuple:
    print("Loading tokenizer and model …")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"          # Required for batch decoding
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Model loaded on : {device}")
    if torch.cuda.is_available():
        print(f"VRAM used       : {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    return tokenizer, model


# ==============================================================
# PARSING
# ==============================================================
def parse_prediction(text: str) -> int:
    """
    Extract the classification label from generated text.
    Uses the LAST digit found — the model may output reasoning
    tokens before the final answer even with MAX_NEW_TOKENS=3.
    Defaults to 0 (normal) if no digit is found.
    """
    text = text.strip()
    matches = re.findall(r"[01]", text)
    if matches:
        return int(matches[-1])    # last match = final answer
    return 0                       # default to normal if uncertain


# ==============================================================
# INFERENCE — batched
# ==============================================================
def classify_batch(
    log_lines: list[str],
    tokenizer,
    model,
) -> list[int]:
    prompts = [build_prompt(log) for log in log_lines]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Track actual prompt length per sample (accounts for left-padding)
    input_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,                 # Greedy — deterministic, best for classification
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    predictions = []
    for output, input_len in zip(outputs, input_lengths):
        generated_tokens = output[input_len:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predictions.append(parse_prediction(generated_text))

    return predictions


# ==============================================================
# DATA LOADING
# ==============================================================
def load_logs(filepath: str, label: int) -> tuple[list[str], list[int]]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {filepath}")

    logs, labels = [], []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(line)
                labels.append(label)

    return logs, labels


# ==============================================================
# EVALUATION
# ==============================================================
def evaluate(
    log_lines: list[str],
    true_labels: list[int],
    tokenizer,
    model,
) -> tuple[dict, list[int]]:

    all_predictions = []
    batch_times     = []
    total           = len(log_lines)

    print(f"Evaluating {total:,} logs with batch_size={BATCH_SIZE} …\n")

    for start in range(0, total, BATCH_SIZE):
        batch_logs = log_lines[start : start + BATCH_SIZE]

        t0 = time.time()
        batch_preds = classify_batch(batch_logs, tokenizer, model)
        elapsed = time.time() - t0

        all_predictions.extend(batch_preds)
        batch_times.append(elapsed)

        done    = min(start + BATCH_SIZE, total)
        avg_ms  = elapsed / len(batch_logs) * 1000
        eta_sec = (avg_ms / 1000) * (total - done)
        print(
            f"  [{done:>6,}/{total:,}]  "
            f"avg {avg_ms:>7.1f} ms/log  "
            f"ETA {eta_sec/60:>5.1f} min"
        )

    # Metrics
    accuracy  = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, zero_division=0)
    recall    = recall_score(true_labels, all_predictions, zero_division=0)
    f1        = f1_score(true_labels, all_predictions, zero_division=0)
    cm        = confusion_matrix(true_labels, all_predictions, labels=[0, 1])

    total_time = sum(batch_times)

    results = {
        "accuracy":         accuracy,
        "precision":        precision,
        "recall":           recall,
        "f1_score":         f1,
        "confusion_matrix": cm,
        "total_time_sec":   total_time,
        "avg_ms_per_log":   total_time / total * 1000 if total else 0,
        "gpu_memory_mb":    (
            torch.cuda.memory_allocated() / 1024**2
            if torch.cuda.is_available() else 0
        ),
    }

    return results, all_predictions


# ==============================================================
# REPORTING
# ==============================================================
def print_results(results: dict):
    cm = results["confusion_matrix"]
    total_sec = results["total_time_sec"]

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Accuracy  : {results['accuracy']:.4f}")
    print(f"Precision : {results['precision']:.4f}")
    print(f"Recall    : {results['recall']:.4f}")
    print(f"F1-Score  : {results['f1_score']:.4f}")
    print("\nConfusion Matrix:")
    print("                   Pred Normal  Pred Abnormal")
    print(f"  Actual Normal      {cm[0][0]:>7,}        {cm[0][1]:>7,}")
    print(f"  Actual Abnormal    {cm[1][0]:>7,}        {cm[1][1]:>7,}")
    print("\nPerformance:")
    print(f"  Avg time/log : {results['avg_ms_per_log']:.1f} ms")
    print(f"  Total time   : {total_sec:.1f} s  ({total_sec / 60:.1f} min)")
    print(f"  GPU memory   : {results['gpu_memory_mb']:.0f} MB")


def save_misclassified(
    log_lines:   list[str],
    true_labels: list[int],
    predictions: list[int],
    output_path: str,
):
    misclassified = [
        (log, true, pred)
        for log, true, pred in zip(log_lines, true_labels, predictions)
        if true != pred
    ]

    total = len(log_lines)
    print(f"\nMisclassified: {len(misclassified):,} / {total:,} "
          f"({len(misclassified)/total*100:.1f}%)")

    if misclassified:
        with open(output_path, "w", encoding="utf-8") as f:
            for log, true, pred in misclassified:
                f.write(f"TRUE:{true} PRED:{pred} | {log}\n")
        print(f"Saved to {output_path}")


# ==============================================================
# MAIN
# ==============================================================
def main():
    random.seed(RANDOM_SEED)

    # Load model
    tokenizer, model = load_model()

    # Load logs
    print("\nLoading logs …")
    normal_logs,   normal_labels   = load_logs(NORMAL_FILE,   label=0)
    abnormal_logs, abnormal_labels = load_logs(ABNORMAL_FILE, label=1)

    print(f"  Normal logs   : {len(normal_logs):,}")
    print(f"  Abnormal logs : {len(abnormal_logs):,}")

    # Combine and shuffle (avoids misleading progress bar — all-normal first half)
    combined = list(zip(normal_logs + abnormal_logs,
                        normal_labels + abnormal_labels))
    random.shuffle(combined)
    all_logs, all_labels = zip(*combined)
    all_logs   = list(all_logs)
    all_labels = list(all_labels)

    print(f"  Total logs    : {len(all_logs):,}  (shuffled, no train/test split)\n")

    # Run evaluation
    print("=" * 60)
    print("STARTING EVALUATION")
    print("=" * 60)

    results, predictions = evaluate(all_logs, all_labels, tokenizer, model)

    # Report
    print_results(results)
    save_misclassified(all_logs, all_labels, predictions, OUTPUT_MISCLASSIFIED)


if __name__ == "__main__":
    main()
