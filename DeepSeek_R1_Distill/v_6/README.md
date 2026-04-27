## Prompt
# ---------------------------------------------------------------------------
# Normal: diagnostic / self-healing lines that look alarming but are benign
# ---------------------------------------------------------------------------

NORMAL_CONTENT_PATTERNS: list[str] = [
    # RTS internal diagnostics (dump, not crash)
    "rts internal error",
    "instruction address",
    "data address",
    "exception syndrome register",
    "special purpose registers",
    "machine state register",
    "alignment exceptions",
    "generating core",
    "ciod: error loading",
    "ciod: generated",
    "detected and corrected",
    "cache parity error corrected",
    "ddr error",          # "ddr error(s) detected and corrected"
    "ddr errors",
    "ce sym",
    "suppressing further interrupts",
    # Transient mount / network retries that succeed
    "nfs mount failed",   # always followed by "retrying" — benign
    "retrying",
]

## Results

EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.8404
  Precision : 0.3118
  Recall    : 1.0000
  F1-score  : 0.4754
  ROC-AUC   : 0.9140

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         308           64
  Actual Abnormal         0           29

Timing:
  Overall time   : 196.85 sec
  Avg per log    : 490.90 ms

Memory:
  Start RSS      : 7703.7 MB
  End RSS        : 7929.7 MB
  Peak RSS       : 7929.7 MB
  Delta RSS      : 226.0 MB
  System RAM     : 9.7%

Routing:
  Rule-based     : 268
  DeepSeek       : 133
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.83      0.91       372
Abnormal (1)       0.31      1.00      0.48        29

    accuracy                           0.84       401
   macro avg       0.66      0.91      0.69       401
weighted avg       0.95      0.84      0.87       401


Misclassified: 64 / 401
