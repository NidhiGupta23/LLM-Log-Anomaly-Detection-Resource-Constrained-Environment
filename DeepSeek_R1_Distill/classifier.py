from dataclasses import dataclass
from typing import List, Tuple

from config import Config
from rules import rule_based_label
from model import llm_classify_batch


@dataclass
class ClassificationResult:
    prediction: int      # 0 = normal, 1 = abnormal
    anomaly_score: float # same as prediction for now (binary)
    source: str          # "rule" | "deepseek"


def classify_batch(
    log_lines: List[str],
    tokenizer,
    model,
    config: Config,
) -> List[ClassificationResult]:
    """
    Hybrid classifier: apply rules first, fall back to LLM for unknowns.
    Returns one ClassificationResult per log line, in the same order.
    """
    results: List[ClassificationResult | None] = [None] * len(log_lines)

    llm_logs: List[str] = []
    llm_indices: List[int] = []

    for i, log in enumerate(log_lines):
        rule_label = rule_based_label(log)
        if rule_label is not None:
            results[i] = ClassificationResult(
                prediction=rule_label,
                anomaly_score=float(rule_label),
                source="rule",
            )
        else:
            llm_logs.append(log)
            llm_indices.append(i)

    if llm_logs:
        llm_preds = llm_classify_batch(llm_logs, tokenizer, model, config)
        for idx, pred in zip(llm_indices, llm_preds):
            results[idx] = ClassificationResult(
                prediction=pred,
                anomaly_score=float(pred),
                source="deepseek",
            )

    return results  # type: ignore[return-value]  # all slots filled


def unzip_results(
    batch_results: List[ClassificationResult],
) -> Tuple[List[int], List[float], List[str]]:
    """Convenience helper to split a result list into three parallel lists."""
    predictions = [r.prediction for r in batch_results]
    scores = [r.anomaly_score for r in batch_results]
    sources = [r.source for r in batch_results]
    return predictions, scores, sources

