import json
from typing import Dict, List

import numpy as np


def save_misclassified(
    log_lines: List[str],
    true_labels: List[int],
    predictions: List[int],
    sources: List[str],
    output_path: str,
) -> None:
    misclassified = [
        (log, true, pred, src)
        for log, true, pred, src in zip(log_lines, true_labels, predictions, sources)
        if true != pred
    ]

    print(f"\nMisclassified: {len(misclassified)} / {len(log_lines)}")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# Misclassified BGL Logs\n")
        f.write(f"# Total: {len(misclassified)} / {len(log_lines)}\n")
        f.write("# Format: TRUE_LABEL PRED_LABEL SOURCE | LOG_LINE\n")
        f.write("#" + "=" * 80 + "\n\n")
        for log, true, pred, src in misclassified:
            f.write(f"TRUE:{true} PRED:{pred} SOURCE:{src} | {log}\n")

    print(f"Saved misclassified logs → {output_path}")


def save_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    np.save(output_path, cm)
    print(f"Saved confusion matrix   → {output_path}")


def save_metrics_json(results: Dict, output_path: str) -> None:
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

    print(f"Saved metrics            → {output_path}")

