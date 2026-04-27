#!/usr/bin/env python3

import re
import time
import json
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional

import psutil
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, classification_report
)
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForCausalLM


class Config:
    MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    NORMAL_FILE = "dataset/normal.log"
    ABNORMAL_FILE = "dataset/abnormal.log"

    BATCH_SIZE = 16
    MAX_LENGTH = 1024
    MAX_NEW_TOKENS = 3

    TEST_SIZE = 0.2
    RANDOM_SEED = 42

    OUTPUT_MISCLASSIFIED = "misclassified.txt"
    OUTPUT_CONFUSION_MATRIX = "confusion_matrix.npy"
    OUTPUT_METRICS = "metrics.json"

    USE_FP16 = True


SYSTEM_PROMPT = """\
You are a BGL log anomaly classifier. Classify the following log line as 0 normal or 1 abnormal.

Rules:
- Output ONLY a single digit: 0 or 1.
- Do not explain.
- Do not add whitespace.

Examples:

Log Entry: {1134530536 2005.12.13 R74-M1-N0-C:J12-U01 2005-12-13-19.22.16.368819 R74-M1-N0-C:J12-U01 RAS KERNEL INFO 2 ddr error(s) detected and corrected on rank 0, symbol 28 over 3365 seconds}
Expected Output: 0

Log Entry: {1134630981 2005.12.14 R37-M1-N8-I:J18-U11 2005-12-14-23.16.21.740107 R37-M1-N8-I:J18-U11 RAS APP FATAL ciod: Error reading message prefix on CioStream socket to 172.16.96.116:49934, Link has been severed}
Expected Output: 1

Log Entry: {1134357549 2005.12.11 R63-M1-N0-I:J18-U11 2005-12-11-19.19.09.737509 R63-M1-N0-I:J18-U11 RAS APP FATAL ciod: LOGIN chdir(/p/gb1/stella/RAPTOR/2183) failed: Input/output error}
Expected Output: 1

Log Entry: {1134748483 2005.12.16 R60-M1-N7-C:J17-U01 2005-12-16-07.54.43.245870 R60-M1-N7-C:J17-U01 RAS KERNEL INFO total of 20 ddr error(s) detected and corrected over 22070 seconds}
Expected Output: 0
"""


def build_prompt(log_line: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nLog Entry: {{{log_line}}}\nExpected Output:"


def load_model(config: Config):
    print("Loading tokenizer and model...")

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_ID,
        trust_remote_code=True
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float32

    if torch.cuda.is_available() and config.USE_FP16:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )

    model.eval()

    print(f"Model loaded on: {next(model.parameters()).device}")

    return tokenizer, model


def parse_prediction(text: str) -> int:
    text = text.strip()
    match = re.search(r"[01]", text)

    if match:
        return int(match.group())

    return 0


def classify_batch(
    log_lines: List[str],
    tokenizer,
    model,
    config: Config
) -> Tuple[List[int], Optional[np.ndarray]]:

    prompts = [build_prompt(log) for log in log_lines]

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=config.MAX_LENGTH
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    prompt_len = inputs["input_ids"].shape[1]

    token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            output_scores=True,
            return_dict_in_generate=True
        )

    predictions = []
    confidences = []

    for idx, output in enumerate(outputs.sequences):
        generated_tokens = output[prompt_len:]

        generated_text = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        pred = parse_prediction(generated_text)
        predictions.append(pred)

        if outputs.scores:
            logits = outputs.scores[0][idx]
            probs = torch.softmax(logits, dim=-1)

            p0 = probs[token_0].item()
            p1 = probs[token_1].item()

            confidence = p1 if pred == 1 else p0
            confidences.append(confidence)
        else:
            confidences.append(0.5)

    return predictions, np.array(confidences)


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
    test_size: float,
    random_seed: int
):
    normal_labels = [0] * len(normal_logs)
    abnormal_labels = [1] * len(abnormal_logs)

    train_normal, test_normal, train_normal_labels, test_normal_labels = train_test_split(
        normal_logs,
        normal_labels,
        test_size=test_size,
        random_state=random_seed
    )

    train_abnormal, test_abnormal, train_abnormal_labels, test_abnormal_labels = train_test_split(
        abnormal_logs,
        abnormal_labels,
        test_size=test_size,
        random_state=random_seed
    )

    train_logs = train_normal + train_abnormal
    train_labels = train_normal_labels + train_abnormal_labels

    test_logs = test_normal + test_abnormal
    test_labels = test_normal_labels + test_abnormal_labels

    train_combined = list(zip(train_logs, train_labels))

    np.random.seed(random_seed)
    np.random.shuffle(train_combined)

    train_logs, train_labels = zip(*train_combined)

    train_logs = list(train_logs)
    train_labels = list(train_labels)

    print("\nDataset split:")
    print(f"  Training: {len(train_logs)} logs")
    print(f"  Testing : {len(test_logs)} logs")

    return train_logs, test_logs, train_labels, test_labels


def get_cpu_memory_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**2


def get_system_memory_usage():
    mem = psutil.virtual_memory()

    return {
        "system_ram_used_mb": mem.used / 1024**2,
        "system_ram_total_mb": mem.total / 1024**2,
        "system_ram_percent": mem.percent
    }


def evaluate(
    log_lines: List[str],
    true_labels: List[int],
    tokenizer,
    model,
    config: Config,
    verbose: bool = True
):
    all_predictions = []
    all_confidences = []
    batch_times = []

    total = len(log_lines)

    if verbose:
        print(f"\nEvaluating {total} logs...")

    peak_cpu_memory_mb = get_cpu_memory_mb()

    for start in range(0, total, config.BATCH_SIZE):
        batch_logs = log_lines[start:start + config.BATCH_SIZE]

        t0 = time.time()

        batch_predictions, batch_confidences = classify_batch(
            batch_logs,
            tokenizer,
            model,
            config
        )

        elapsed = time.time() - t0

        all_predictions.extend(batch_predictions)
        all_confidences.extend(batch_confidences)
        batch_times.append(elapsed)

        current_cpu_memory_mb = get_cpu_memory_mb()
        peak_cpu_memory_mb = max(peak_cpu_memory_mb, current_cpu_memory_mb)

        if verbose:
            done = min(start + config.BATCH_SIZE, total)
            avg_ms = elapsed / len(batch_logs) * 1000

            print(
                f"[{done:>6}/{total}] "
                f"batch={len(batch_logs):>3} "
                f"avg={avg_ms:>8.1f} ms/log "
                f"cpu_ram={current_cpu_memory_mb:>8.1f} MB"
            )

    accuracy = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, zero_division=0)
    recall = recall_score(true_labels, all_predictions, zero_division=0)
    f1 = f1_score(true_labels, all_predictions, zero_division=0)

    cm = confusion_matrix(true_labels, all_predictions, labels=[0, 1])

    roc_auc = 0.5

    if len(set(true_labels)) > 1 and len(all_confidences) > 0:
        try:
            roc_auc = roc_auc_score(true_labels, all_confidences)
        except Exception as e:
            print(f"Warning: Could not compute ROC-AUC: {e}")

    total_time = sum(batch_times)
    avg_ms = total_time / total * 1000 if total else 0

    gpu_memory_mb = 0
    gpu_peak_memory_mb = 0

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.memory_allocated() / 1024**2
        gpu_peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

    cpu_memory_mb = get_cpu_memory_mb()
    system_memory = get_system_memory_usage()

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "total_time_sec": total_time,
        "avg_ms_per_log": avg_ms,
        "gpu_memory_mb": gpu_memory_mb,
        "gpu_peak_memory_mb": gpu_peak_memory_mb,
        "cpu_memory_mb": cpu_memory_mb,
        "cpu_peak_memory_mb": peak_cpu_memory_mb,
        "system_ram_used_mb": system_memory["system_ram_used_mb"],
        "system_ram_total_mb": system_memory["system_ram_total_mb"],
        "system_ram_percent": system_memory["system_ram_percent"],
        "num_samples": total
    }

    return results, all_predictions, np.array(all_confidences)


def print_results(results: Dict):
    cm = results["confusion_matrix"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nClassification Metrics:")
    print(f"   Accuracy  : {results['accuracy']:.4f}")
    print(f"   Precision : {results['precision']:.4f}")
    print(f"   Recall    : {results['recall']:.4f}")
    print(f"   F1-Score  : {results['f1_score']:.4f}")
    print(f"   ROC-AUC   : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print(f"   {'':>16} {'Pred Normal':>14} {'Pred Abnormal':>16}")
    print(f"   {'Actual Normal':>16} {cm[0][0]:>14} {cm[0][1]:>16}")
    print(f"   {'Actual Abnormal':>16} {cm[1][0]:>14} {cm[1][1]:>16}")

    print("\nPerformance:")
    print(f"   Avg time per log     : {results['avg_ms_per_log']:.1f} ms")
    print(f"   Total time           : {results['total_time_sec']:.1f} seconds")
    print(f"   Total samples        : {results['num_samples']}")

    print("\nMemory Usage:")
    print(f"   CPU memory current   : {results['cpu_memory_mb']:.1f} MB")
    print(f"   CPU memory peak      : {results['cpu_peak_memory_mb']:.1f} MB")
    print(f"   System RAM used      : {results['system_ram_used_mb']:.1f} MB")
    print(f"   System RAM total     : {results['system_ram_total_mb']:.1f} MB")
    print(f"   System RAM percent   : {results['system_ram_percent']:.1f}%")
    print(f"   GPU memory current   : {results['gpu_memory_mb']:.1f} MB")
    print(f"   GPU memory peak      : {results['gpu_peak_memory_mb']:.1f} MB")


def save_misclassified(
    log_lines: List[str],
    true_labels: List[int],
    predictions: List[int],
    confidences: np.ndarray,
    output_path: str
):
    misclassified = []

    for i, (log, true, pred) in enumerate(zip(log_lines, true_labels, predictions)):
        if true != pred:
            confidence = confidences[i] if i < len(confidences) else 0.5
            misclassified.append((log, true, pred, confidence))

    print(
        f"\nMisclassified: {len(misclassified)} / {len(log_lines)} "
        f"({len(misclassified) / len(log_lines) * 100:.1f}%)"
    )

    if misclassified:
        misclassified.sort(key=lambda x: x[3])

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Misclassified BGL Logs\n")
            f.write(f"# Total: {len(misclassified)} / {len(log_lines)}\n")
            f.write("# Format: TRUE_LABEL PRED_LABEL CONFIDENCE | LOG_LINE\n")
            f.write("#" + "=" * 80 + "\n\n")

            for log, true, pred, conf in misclassified:
                f.write(f"TRUE:{true} PRED:{pred} CONF:{conf:.3f} | {log}\n")

        print(f"Saved misclassified logs to {output_path}")


def save_confusion_matrix(cm: np.ndarray, output_path: str):
    np.save(output_path, cm)
    print(f"Confusion matrix saved to {output_path}")


def save_metrics_json(results: Dict, output_path: str):
    serializable_results = {}

    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.float32):
            serializable_results[key] = float(value)
        elif isinstance(value, np.float64):
            serializable_results[key] = float(value)
        elif isinstance(value, np.int32):
            serializable_results[key] = int(value)
        elif isinstance(value, np.int64):
            serializable_results[key] = int(value)
        else:
            serializable_results[key] = value

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable_results, f, indent=2)

    print(f"Metrics saved to {output_path}")


def main():
    config = Config()

    print("=" * 70)
    print("BGL LOG ANOMALY CLASSIFIER")
    print(f"Model: {config.MODEL_ID}")
    print("=" * 70)

    tokenizer, model = load_model(config)

    print("\nLoading logs...")

    try:
        normal_logs, _ = load_logs(config.NORMAL_FILE, 0)
        abnormal_logs, _ = load_logs(config.ABNORMAL_FILE, 1)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    print(f"Normal logs   : {len(normal_logs)}")
    print(f"Abnormal logs : {len(abnormal_logs)}")
    print(f"Total logs    : {len(normal_logs) + len(abnormal_logs)}")

    train_logs, test_logs, train_labels, test_labels = create_train_test_split(
        normal_logs,
        abnormal_logs,
        test_size=config.TEST_SIZE,
        random_seed=config.RANDOM_SEED
    )

    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)

    results, predictions, confidences = evaluate(
        test_logs,
        test_labels,
        tokenizer,
        model,
        config,
        verbose=True
    )

    print_results(results)

    save_misclassified(
        test_logs,
        test_labels,
        predictions,
        confidences,
        config.OUTPUT_MISCLASSIFIED
    )

    save_confusion_matrix(
        results["confusion_matrix"],
        config.OUTPUT_CONFUSION_MATRIX
    )

    save_metrics_json(
        results,
        config.OUTPUT_METRICS
    )

    print("\n" + "=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            test_labels,
            predictions,
            target_names=["Normal (0)", "Abnormal (1)"]
        )
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
