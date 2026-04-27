#!/usr/bin/env python3
"""
BGL Log Anomaly Classifier

Hybrid approach:
1. Rule-based classifier for known BGL patterns
2. DeepSeek-R1-Distill-Qwen-1.5B fallback for unknown patterns

Outputs:
- metrics.json
- confusion_matrix.npy
- misclassified.txt
"""

import os
import re
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import psutil
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==============================================================
# CONFIGURATION
# ==============================================================

class Config:
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    NORMAL_FILE = "dataset/normal.log"
    ABNORMAL_FILE = "dataset/abnormal.log"

    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    BATCH_SIZE = 16
    MAX_LENGTH = 1024
    MAX_NEW_TOKENS = 3

    OUTPUT_MISCLASSIFIED = "misclassified.txt"
    OUTPUT_CONFUSION_MATRIX = "confusion_matrix.npy"
    OUTPUT_METRICS = "metrics.json"


# ==============================================================
# MEMORY HELPERS
# ==============================================================

def get_cpu_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def get_system_memory_percent() -> float:
    return psutil.virtual_memory().percent


def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2
    return 0.0


# ==============================================================
# RULE-BASED CLASSIFIER
# ==============================================================

ABNORMAL_PATTERNS = [
    "kernel terminated",
    "rts panic",
    "stopping execution",
    "data tlb error interrupt",
    "data storage interrupt",
    "link has been severed",
    "connection timed out",
    "connection reset by peer",
    "lustre mount failed",
    "error receiving packet on tree network",
    "input/output error",
    "i/o error",
]

NORMAL_PATTERNS = [
    "rts internal error",
    "instruction address",
    "data address",
    "exception syndrome register",
    "special purpose registers",
    "machine state register",
    "detected and corrected",
    "alignment exceptions",
    "generating core",
    "nfs mount failed",
    "retrying",
    "cache parity error corrected",
    "ddr error",
    "ddr errors",
    "ce sym",
    "suppressing further interrupts",
]


def rule_based_label(log: str) -> Optional[int]:
    x = log.lower()

    for pattern in ABNORMAL_PATTERNS:
        if pattern in x:
            return 1

    for pattern in NORMAL_PATTERNS:
        if pattern in x:
            return 0

    return None


# ==============================================================
# PROMPT
# ==============================================================

SYSTEM_PROMPT = """\
You are a BGL Blue Gene/L log anomaly classifier.

Classify the log line as:
0 = NORMAL
1 = ABNORMAL

Important:
Do not classify only from severity.
The word FATAL does not automatically mean abnormal.
Some RAS KERNEL FATAL lines are normal diagnostic dump lines.

NORMAL examples:
- RAS KERNEL FATAL rts internal error
- RAS KERNEL FATAL instruction address
- RAS KERNEL FATAL machine state register
- RAS KERNEL INFO ddr error(s) detected and corrected
- RAS KERNEL INFO NFS Mount failed, slept 15 seconds, retrying

ABNORMAL examples:
- RAS KERNEL FATAL rts: kernel terminated
- RAS KERNEL FATAL rts panic! - stopping execution
- RAS KERNEL FATAL data TLB error interrupt
- RAS KERNEL FATAL data storage interrupt
- RAS APP FATAL Link has been severed
- RAS APP FATAL Connection timed out
- RAS APP FATAL Connection reset by peer
- RAS KERNEL FATAL Lustre mount FAILED

Return exactly one digit: 0 or 1.
"""


def build_prompt(log_line: str) -> str:
    return f"{SYSTEM_PROMPT}\nLOG:\n{log_line}\nOUTPUT:"


# ==============================================================
# MODEL LOADING
# ==============================================================

def load_model(config: Config):
    print("Loading tokenizer and model...")
    cpu_before = get_cpu_memory_mb()

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    cpu_after = get_cpu_memory_mb()

    print(f"Model device              : {next(model.parameters()).device}")
    print(f"CPU memory after loading  : {cpu_after:.1f} MB")
    print(f"CPU memory model increase : {cpu_after - cpu_before:.1f} MB")
    print(f"System RAM usage          : {get_system_memory_percent():.1f}%")
    print(f"GPU memory                : {get_gpu_memory_mb():.1f} MB")

    return tokenizer, model


# ==============================================================
# DATA LOADING
# ==============================================================

def load_logs(filepath: str, label: int) -> Tuple[List[str], List[int]]:
    path = Path(filepath)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    logs = []
    labels = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                logs.append(line)
                labels.append(label)

    return logs, labels


def create_train_test_split(
    normal_logs: List[str],
    abnormal_logs: List[str],
    config: Config,
):
    normal_labels = [0] * len(normal_logs)
    abnormal_labels = [1] * len(abnormal_logs)

    train_normal, test_normal, train_normal_labels, test_normal_labels = train_test_split(
        normal_logs,
        normal_labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    train_abnormal, test_abnormal, train_abnormal_labels, test_abnormal_labels = train_test_split(
        abnormal_logs,
        abnormal_labels,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
    )

    train_logs = train_normal + train_abnormal
    train_labels = train_normal_labels + train_abnormal_labels

    test_logs = test_normal + test_abnormal
    test_labels = test_normal_labels + test_abnormal_labels

    combined = list(zip(test_logs, test_labels))
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(combined)

    test_logs, test_labels = zip(*combined)

    test_logs = list(test_logs)
    test_labels = list(test_labels)

    print("\nDataset split:")
    print(f"Training logs : {len(train_logs)}")
    print(f"Testing logs  : {len(test_logs)}")
    print(f"Test normal   : {test_labels.count(0)}")
    print(f"Test abnormal : {test_labels.count(1)}")

    return train_logs, test_logs, train_labels, test_labels


# ==============================================================
# PREDICTION PARSING
# ==============================================================

def parse_prediction(text: str) -> int:
    text = text.strip()
    text = text.replace("<think>", "").replace("</think>", "")

    match = re.search(r"[01]", text)

    if match:
        return int(match.group())

    return 0


# ==============================================================
# DEEPSEEK FALLBACK CLASSIFIER
# ==============================================================

def deepseek_classify_batch(
    log_lines: List[str],
    tokenizer,
    model,
    config: Config,
) -> List[int]:
    prompts = [build_prompt(log) for log in log_lines]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH,
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_lengths = inputs["attention_mask"].sum(dim=1)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    predictions = []

    for output, input_len in zip(outputs, input_lengths):
        generated_tokens = output[input_len:]
        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True,
        )
        predictions.append(parse_prediction(generated_text))

    return predictions


# ==============================================================
# HYBRID CLASSIFIER
# ==============================================================

def classify_batch(
    log_lines: List[str],
    tokenizer,
    model,
    config: Config,
):
    predictions = [None] * len(log_lines)
    sources = [""] * len(log_lines)

    unknown_logs = []
    unknown_indices = []

    for i, log in enumerate(log_lines):
        rule_label = rule_based_label(log)

        if rule_label is not None:
            predictions[i] = rule_label
            sources[i] = "rule"
        else:
            unknown_logs.append(log)
            unknown_indices.append(i)

    if unknown_logs:
        llm_predictions = deepseek_classify_batch(
            unknown_logs,
            tokenizer,
            model,
            config,
        )

        for idx, pred in zip(unknown_indices, llm_predictions):
            predictions[idx] = pred
            sources[idx] = "deepseek"

    return predictions, sources


# ==============================================================
# EVALUATION
# ==============================================================

def evaluate(
    log_lines: List[str],
    true_labels: List[int],
    tokenizer,
    model,
    config: Config,
):
    all_predictions = []
    all_sources = []

    total = len(log_lines)

    cpu_start = get_cpu_memory_mb()
    cpu_peak = cpu_start
    gpu_start = get_gpu_memory_mb()

    start_time = time.time()

    print(f"\nEvaluating {total} logs...")

    for start in range(0, total, config.BATCH_SIZE):
        batch_logs = log_lines[start:start + config.BATCH_SIZE]

        batch_t0 = time.time()

        batch_predictions, batch_sources = classify_batch(
            batch_logs,
            tokenizer,
            model,
            config,
        )

        batch_elapsed = time.time() - batch_t0

        all_predictions.extend(batch_predictions)
        all_sources.extend(batch_sources)

        cpu_now = get_cpu_memory_mb()
        cpu_peak = max(cpu_peak, cpu_now)

        done = min(start + config.BATCH_SIZE, total)
        avg_ms = batch_elapsed / len(batch_logs) * 1000

        rule_count = batch_sources.count("rule")
        deepseek_count = batch_sources.count("deepseek")

        print(
            f"[{done:>6}/{total}] "
            f"batch={len(batch_logs):>3} "
            f"rule={rule_count:>3} "
            f"deepseek={deepseek_count:>3} "
            f"avg={avg_ms:>8.1f} ms/log "
            f"cpu={cpu_now:.1f} MB "
            f"gpu={get_gpu_memory_mb():.1f} MB"
        )

    total_time = time.time() - start_time

    cpu_end = get_cpu_memory_mb()
    gpu_end = get_gpu_memory_mb()

    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, zero_division=0)
    recall = recall_score(true_labels, all_predictions, zero_division=0)
    f1 = f1_score(true_labels, all_predictions, zero_division=0)
    cm = confusion_matrix(true_labels, all_predictions, labels=[0, 1])

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": cm,
        "total_time_sec": total_time,
        "avg_ms_per_log": total_time / total * 1000 if total else 0,
        "num_samples": total,
        "rule_count": all_sources.count("rule"),
        "deepseek_count": all_sources.count("deepseek"),
        "cpu_memory_start_mb": cpu_start,
        "cpu_memory_end_mb": cpu_end,
        "cpu_memory_peak_mb": cpu_peak,
        "cpu_memory_delta_mb": cpu_end - cpu_start,
        "system_memory_percent": get_system_memory_percent(),
        "gpu_memory_start_mb": gpu_start,
        "gpu_memory_end_mb": gpu_end,
        "gpu_memory_delta_mb": gpu_end - gpu_start,
    }

    return results, all_predictions, all_sources


# ==============================================================
# REPORTING
# ==============================================================

def print_results(results: Dict):
    cm = results["confusion_matrix"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nClassification Metrics:")
    print(f"Accuracy  : {results['accuracy']:.4f} ({results['accuracy'] * 100:.2f}%)")
    print(f"Precision : {results['precision']:.4f} ({results['precision'] * 100:.2f}%)")
    print(f"Recall    : {results['recall']:.4f} ({results['recall'] * 100:.2f}%)")
    print(f"F1-Score  : {results['f1_score']:.4f} ({results['f1_score'] * 100:.2f}%)")

    print("\nConfusion Matrix:")
    print("                  Pred Normal  Pred Abnormal")
    print(f"Actual Normal      {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"Actual Abnormal    {cm[1][0]:>6}       {cm[1][1]:>6}")

    print("\nRouting:")
    print(f"Rule-based labels : {results['rule_count']}")
    print(f"DeepSeek labels   : {results['deepseek_count']}")

    print("\nPerformance:")
    print(f"Total time        : {results['total_time_sec']:.2f} sec")
    print(f"Avg time/log      : {results['avg_ms_per_log']:.2f} ms")
    print(f"Total samples     : {results['num_samples']}")

    print("\nCPU Memory:")
    print(f"Start RSS         : {results['cpu_memory_start_mb']:.1f} MB")
    print(f"End RSS           : {results['cpu_memory_end_mb']:.1f} MB")
    print(f"Peak RSS          : {results['cpu_memory_peak_mb']:.1f} MB")
    print(f"Delta RSS         : {results['cpu_memory_delta_mb']:.1f} MB")
    print(f"System RAM used   : {results['system_memory_percent']:.1f}%")

    print("\nGPU Memory:")
    print(f"Start allocated   : {results['gpu_memory_start_mb']:.1f} MB")
    print(f"End allocated     : {results['gpu_memory_end_mb']:.1f} MB")
    print(f"Delta allocated   : {results['gpu_memory_delta_mb']:.1f} MB")


def save_misclassified(
    log_lines: List[str],
    true_labels: List[int],
    predictions: List[int],
    sources: List[str],
    output_path: str,
):
    misclassified = []

    for log, true, pred, source in zip(log_lines, true_labels, predictions, sources):
        if true != pred:
            misclassified.append((log, true, pred, source))

    print(f"\nMisclassified: {len(misclassified)} / {len(log_lines)}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Misclassified BGL Logs\n")
        f.write(f"# Total: {len(misclassified)} / {len(log_lines)}\n")
        f.write("# Format: TRUE_LABEL PRED_LABEL SOURCE | LOG_LINE\n")
        f.write("#" + "=" * 80 + "\n\n")

        for log, true, pred, source in misclassified:
            f.write(f"TRUE:{true} PRED:{pred} SOURCE:{source} | {log}\n")

    print(f"Saved misclassified logs to {output_path}")


def save_metrics_json(results: Dict, output_path: str):
    serializable = {}

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable[key] = int(value)
        elif isinstance(value, np.floating):
            serializable[key] = float(value)
        else:
            serializable[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved metrics to {output_path}")


# ==============================================================
# MAIN
# ==============================================================

def main():
    config = Config()

    print("=" * 70)
    print("BGL LOG ANOMALY CLASSIFIER")
    print("=" * 70)

    print(f"Initial CPU memory : {get_cpu_memory_mb():.1f} MB")
    print(f"Initial RAM usage  : {get_system_memory_percent():.1f}%")

    tokenizer, model = load_model(config)

    print("\nLoading logs...")

    normal_logs, _ = load_logs(config.NORMAL_FILE, 0)
    abnormal_logs, _ = load_logs(config.ABNORMAL_FILE, 1)

    print(f"Normal logs   : {len(normal_logs)}")
    print(f"Abnormal logs : {len(abnormal_logs)}")
    print(f"Total logs    : {len(normal_logs) + len(abnormal_logs)}")

    _, test_logs, _, test_labels = create_train_test_split(
        normal_logs,
        abnormal_logs,
        config,
    )

    results, predictions, sources = evaluate(
        test_logs,
        test_labels,
        tokenizer,
        model,
        config,
    )

    print_results(results)

    save_misclassified(
        test_logs,
        test_labels,
        predictions,
        sources,
        config.OUTPUT_MISCLASSIFIED,
    )

    np.save(config.OUTPUT_CONFUSION_MATRIX, results["confusion_matrix"])
    print(f"Saved confusion matrix to {config.OUTPUT_CONFUSION_MATRIX}")

    save_metrics_json(results, config.OUTPUT_METRICS)

    print("\nClassification Report:")
    print(
        classification_report(
            test_labels,
            predictions,
            target_names=["Normal (0)", "Abnormal (1)"],
            zero_division=0,
        )
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
