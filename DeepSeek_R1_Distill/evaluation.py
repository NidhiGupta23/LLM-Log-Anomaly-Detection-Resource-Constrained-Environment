import time
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from classifier import classify_batch, unzip_results
from config import Config
from memory import get_cpu_memory_mb, get_system_memory_percent


def evaluate(
    log_lines: List[str],
    true_labels: List[int],
    tokenizer,
    model,
    config: Config,
) -> Tuple[Dict, List[int], List[str], List[float]]:
    """
    Run the hybrid classifier over all log lines in batches.
    Returns (metrics_dict, predictions, sources, anomaly_scores).
    """
    all_predictions: List[int] = []
    all_scores: List[float] = []
    all_sources: List[str] = []

    batch_times: List[float] = []

    total = len(log_lines)
    cpu_start = cpu_peak = get_cpu_memory_mb()
    wall_start = time.time()

    print(f"\nEvaluating {total} logs...")

    for start in range(0, total, config.batch_size):
        batch_logs = log_lines[start : start + config.batch_size]

        t0 = time.time()
        batch_results = classify_batch(batch_logs, tokenizer, model, config)
        elapsed = time.time() - t0
        batch_times.append(elapsed)

        preds, scores, sources = unzip_results(batch_results)
        all_predictions.extend(preds)
        all_scores.extend(scores)
        all_sources.extend(sources)

        cpu_now = get_cpu_memory_mb()
        cpu_peak = max(cpu_peak, cpu_now)

        done = min(start + config.batch_size, total)
        avg_ms = elapsed / len(batch_logs) * 1000
        print(
            f"  [{done:>6}/{total}] "
            f"batch={len(batch_logs):>3} "
            f"rule={sources.count('rule'):>3} "
            f"deepseek={sources.count('deepseek'):>3} "
            f"avg={avg_ms:>7.1f} ms/log "
            f"cpu={cpu_now:.1f} MB"
        )

    wall_total = time.time() - wall_start
    cpu_end = get_cpu_memory_mb()

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    accuracy  = accuracy_score(true_labels, all_predictions)
    precision = precision_score(true_labels, all_predictions, zero_division=0)
    recall    = recall_score(true_labels, all_predictions, zero_division=0)
    f1        = f1_score(true_labels, all_predictions, zero_division=0)
    cm        = confusion_matrix(true_labels, all_predictions, labels=[0, 1])

    try:
        roc_auc = roc_auc_score(true_labels, all_scores)
    except ValueError:
        roc_auc = 0.5

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "overall_time_to_execute_dataset_sec": wall_total,
        "avg_time_per_log_ms": wall_total / total * 1000 if total else 0.0,
        "num_samples": total,
        "rule_count": all_sources.count("rule"),
        "deepseek_count": all_sources.count("deepseek"),
        "cpu_memory_start_mb": cpu_start,
        "cpu_memory_end_mb": cpu_end,
        "cpu_memory_peak_mb": cpu_peak,
        "cpu_memory_delta_mb": cpu_end - cpu_start,
        "system_memory_percent": get_system_memory_percent(),
    }

    return results, all_predictions, all_sources, all_scores


def print_results(results: Dict) -> None:
    cm = results["confusion_matrix"]

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print("\nEvaluation Metrics:")
    print(f"  Accuracy  : {results['accuracy']:.4f}")
    print(f"  Precision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")
    print(f"  F1-score  : {results['f1_score']:.4f}")
    print(f"  ROC-AUC   : {results['roc_auc']:.4f}")

    print("\nConfusion Matrix:")
    print("                  Pred Normal  Pred Abnormal")
    print(f"  Actual Normal      {cm[0][0]:>6}       {cm[0][1]:>6}")
    print(f"  Actual Abnormal    {cm[1][0]:>6}       {cm[1][1]:>6}")

    print("\nTiming:")
    print(f"  Overall time   : {results['overall_time_to_execute_dataset_sec']:.2f} sec")
    print(f"  Avg per log    : {results['avg_time_per_log_ms']:.2f} ms")

    print("\nMemory:")
    print(f"  Start RSS      : {results['cpu_memory_start_mb']:.1f} MB")
    print(f"  End RSS        : {results['cpu_memory_end_mb']:.1f} MB")
    print(f"  Peak RSS       : {results['cpu_memory_peak_mb']:.1f} MB")
    print(f"  Delta RSS      : {results['cpu_memory_delta_mb']:.1f} MB")
    print(f"  System RAM     : {results['system_memory_percent']:.1f}%")

    print("\nRouting:")
    print(f"  Rule-based     : {results['rule_count']}")
    print(f"  DeepSeek       : {results['deepseek_count']}")
    print(f"  Total          : {results['num_samples']}")


def print_classification_report(true_labels: List[int], predictions: List[int]) -> None:
    print("\nClassification Report:")
    print(
        classification_report(
            true_labels,
            predictions,
            target_names=["Normal (0)", "Abnormal (1)"],
            zero_division=0,
        )
    )

