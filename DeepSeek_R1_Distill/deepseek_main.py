#!/usr/bin/env python3
"""
BGL Blue Gene/L log anomaly classifier — entry point.
"""

from config import Config
from data import create_test_split, load_logs
from evaluation import evaluate, print_classification_report, print_results
from memory import get_cpu_memory_mb, get_system_memory_percent
from model import load_model
from outputs import save_confusion_matrix, save_metrics_json, save_misclassified


def main() -> None:
    config = Config()

    print("=" * 70)
    print("BGL LOG ANOMALY CLASSIFIER")
    print("=" * 70)
    print(f"Initial CPU memory : {get_cpu_memory_mb():.1f} MB")
    print(f"Initial RAM usage  : {get_system_memory_percent():.1f}%")

    tokenizer, model = load_model(config)

    print("\nLoading logs...")
    normal_logs, _  = load_logs(config.normal_file,   label=0)
    abnormal_logs, _ = load_logs(config.abnormal_file, label=1)
    print(f"  Normal   : {len(normal_logs)}")
    print(f"  Abnormal : {len(abnormal_logs)}")
    print(f"  Total    : {len(normal_logs) + len(abnormal_logs)}")

    test_logs, test_labels = create_test_split(normal_logs, abnormal_logs, config)

    results, predictions, sources, scores = evaluate(
        test_logs, test_labels, tokenizer, model, config
    )

    print_results(results)
    print_classification_report(test_labels, predictions)

    save_misclassified(
        test_logs, test_labels, predictions, sources,
        config.output_misclassified,
    )
    save_confusion_matrix(results["confusion_matrix"], config.output_confusion_matrix)
    save_metrics_json(results, config.output_metrics)

    print("\nDone.")


if __name__ == "__main__":
    main()

