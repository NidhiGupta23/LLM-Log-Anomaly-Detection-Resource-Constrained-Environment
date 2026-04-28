## Prompt
# Abnormal: process/kernel termination
_ABNORMAL_TERMINATION = [
    "kernel terminated",
    "rts panic",
    "stopping execution",
    "job terminated",
    "process terminated",
    "ciod: exiting",
    "ciod: terminating",
    "killed by signal",
    "error receiving packet on tree network",
    "link has been severed",
]

# Abnormal: storage / mount failures
_ABNORMAL_STORAGE = [
    "lustre mount failed",
    "lustre mount error",
    "data tlb error interrupt",
    "data storage interrupt",
]

# Normal pattern
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
    "ASSERT condition: ASSERT expression=0",
    "machine check enable",
    "INFO: program interrupt",
    "iar dear ",
    "data cache search parity error detected. attempting to correct",
    "program interrupt:",
    "ERROR idoproxydb hit ASSERT condition",
    "0 critical input interrupts",
    "Source line=1043",
    "wait state enable",
    "disable store gathering",
    "icache prefetch threshold",
    # Self-corrected hardware errors
    "detected and corrected",
    "cache parity error corrected",
    "ddr error",          # "ddr error(s) detected and corrected"
    "ddr errors",
    "ce sym",
    "suppressing further interrupts",
    # Transient mount / network retries that succeed
    "nfs mount failed",   # always followed by "retrying" — benign
    "retrying",
    "errno=0",
    "data store interrupt caused by icbi",
]

## Results

EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.9252
  Precision : 0.4915
  Recall    : 1.0000
  F1-score  : 0.6591
  ROC-AUC   : 0.9597

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         342           30
  Actual Abnormal         0           29

Timing:
  Overall time   : 126.47 sec
  Avg per log    : 315.38 ms

Memory:
  Start RSS      : 7702.9 MB
  End RSS        : 7875.8 MB
  Peak RSS       : 7916.2 MB
  Delta RSS      : 172.9 MB
  System RAM     : 9.8%

Routing:
  Rule-based     : 316
  DeepSeek       : 85
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.92      0.96       372
Abnormal (1)       0.49      1.00      0.66        29

    accuracy                           0.93       401
   macro avg       0.75      0.96      0.81       401
weighted avg       0.96      0.93      0.94       401


Misclassified: 30 / 401
