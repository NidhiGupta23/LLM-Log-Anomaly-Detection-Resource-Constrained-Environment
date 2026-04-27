## Rules
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

## Prompt
_SYSTEM_PROMPT = """\
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


## Results
======================================================================
EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.7830
  Precision : 0.2500
  Recall    : 1.0000
  F1-score  : 0.4000
  ROC-AUC   : 0.8831

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         285           87
  Actual Abnormal         0           29

Timing:
  Overall time   : 108.44 sec
  Avg per log    : 270.43 ms

Memory:
  Start RSS      : 7702.1 MB
  End RSS        : 7784.9 MB
  Peak RSS       : 7812.0 MB
  Delta RSS      : 82.8 MB
  System RAM     : 10.2%

Routing:
  Rule-based     : 263
  DeepSeek       : 138
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.77      0.87       372
Abnormal (1)       0.25      1.00      0.40        29

    accuracy                           0.78       401
   macro avg       0.62      0.88      0.63       401
weighted avg       0.95      0.78      0.83       401


Misclassified: 87 / 401
