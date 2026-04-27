## Prompt
You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.
 
You will receive a structured log entry with the following fields:
  TYPE      – high-level event type (e.g. RAS)
  COMPONENT – subsystem that emitted the event (e.g. KERNEL, APP)
  LEVEL     – severity (e.g. FATAL, INFO, WARNING)
  CONTENT   – free-text description of the event
 
Classify the entry as:
  0 = NORMAL
  1 = ABNORMAL
 
CRITICAL rules:
- Do NOT classify based on LEVEL alone.
- FATAL does NOT automatically mean abnormal — many FATAL lines are
  diagnostic register dumps that are part of normal error recovery.
- Focus on CONTENT: what actually happened?
 
ABNORMAL indicators in CONTENT:
- Kernel or process termination  ("kernel terminated", "rts panic",
  "stopping execution", "job terminated", "killed by signal")
- ciod socket / read failures    ("ciod: error reading message",
  "ciod: read error", "ciod: socket error", "ciod: failed to connect")
- Network packet errors          ("error receiving packet on tree network",
  "link has been severed", "connection timed out", "connection reset by peer")
- Storage / mount failures       ("Lustre mount FAILED", "data TLB error interrupt",
  "data storage interrupt", "I/O error")
- Illegal instruction / machine check interrupt
 
NORMAL indicators in CONTENT:
- Register dump lines            ("rts internal error", "instruction address",
  "machine state register", "exception syndrome register")
- Self-corrected hardware errors ("detected and corrected", "cache parity error
  corrected", "ddr error(s) detected and corrected", "CE sym")
- Transient recoverable failures ("NFS mount failed … retrying",
  "suppressing further interrupts")
 
EXAMPLES:
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts internal error          → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=instruction address: …      → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=INFO  CONTENT=ddr error(s) detected …    → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=INFO  CONTENT=NFS Mount failed, retrying  → 0
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts: kernel terminated      → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts panic! stopping exec    → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=Link has been severed        → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=Connection timed out         → 1
  TYPE=RAS COMPONENT=APP    LEVEL=FATAL CONTENT=ciod: error reading message  → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=Lustre mount FAILED          → 1
  TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=Error receiving packet on
    tree network (expecting type …)                                          → 1
 
Return exactly one digit: 0 or 1. No explanation.

## Rules
_ABNORMAL_TERMINATION = [
    "kernel terminated",
    "rts panic",
    "stopping execution",
    "job terminated",
    "process terminated",
    "ciod: exiting",
    "ciod: terminating",
    "killed by signal",
]
 
# Abnormal: network / socket / IPC failures
_ABNORMAL_NETWORK = [
    "error receiving packet on tree network",   # expecting type …
    "link has been severed",
    "connection timed out",
    "connection reset by peer",
    "ciod: error reading message",
    "ciod: read error",
    "ciod: socket error",
    "ciod: failed to read",
    "ciod: failed to connect",
    "socket closed unexpectedly",
    "unexpected eof",
    "i/o error on socket",
]
 
# Abnormal: storage / mount failures
_ABNORMAL_STORAGE = [
    "lustre mount failed",
    "lustre mount error",
    "data tlb error interrupt",
    "data storage interrupt",
    "input/output error",
    "i/o error",
]
 
# Abnormal: illegal / hardware faults that require intervention
_ABNORMAL_ILLEGAL = [
    "illegal instruction",
    "illegal operation",
    "program interrupt",
    "machine check interrupt",
    "external interrupt",
    "ciod: error",                   # generic ciod error catch-all (after specifics)
]
 
ABNORMAL_CONTENT_PATTERNS: list[str] = (
    _ABNORMAL_TERMINATION
    + _ABNORMAL_NETWORK
    + _ABNORMAL_STORAGE
    + _ABNORMAL_ILLEGAL
)
 
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
]

## Results
EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 0.8304
  Precision : 0.2990
  Recall    : 1.0000
  F1-score  : 0.4603
  ROC-AUC   : 0.9086

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         304           68
  Actual Abnormal         0           29

Timing:
  Overall time   : 184.17 sec
  Avg per log    : 459.28 ms

Memory:
  Start RSS      : 7703.6 MB
  End RSS        : 7860.7 MB
  Peak RSS       : 7860.7 MB
  Delta RSS      : 157.1 MB
  System RAM     : 9.4%

Routing:
  Rule-based     : 279
  DeepSeek       : 122
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      0.82      0.90       372
Abnormal (1)       0.30      1.00      0.46        29

    accuracy                           0.83       401
   macro avg       0.65      0.91      0.68       401
weighted avg       0.95      0.83      0.87       401


Misclassified: 68 / 401
