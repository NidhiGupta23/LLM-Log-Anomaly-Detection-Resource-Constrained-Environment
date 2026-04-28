## Prompts
# ---------------------------------------------------------------------------
# Content patterns — matched against entry.content (case-insensitive)
# ---------------------------------------------------------------------------
'''
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
    "external interrupt",        # generic ciod error catch-all (after specifics)
]

ABNORMAL_CONTENT_PATTERNS: list[str] = (
    _ABNORMAL_TERMINATION
    + _ABNORMAL_NETWORK
    + _ABNORMAL_STORAGE
    + _ABNORMAL_ILLEGAL
)
ABNORMAL_CONTENT_PATTERNS: list[str] = (
    _ABNORMAL_TERMINATION
    + _ABNORMAL_STORAGE
)'''
# Hard terminations — process/kernel stopped unrecoverably
_ABNORMAL_TERMINATION: list[str] = [
    "kernel terminated",
    "rts panic",
    "stopping execution",
    "job terminated",
    "process terminated",
    "ciod: exiting",
    "ciod: terminating",
    "killed by signal",
]

# Network / socket / IPC failures — data loss or broken pipe
_ABNORMAL_NETWORK: list[str] = [
    "error receiving packet on tree network",  # "… expecting type …"
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

# Storage / mount failures — unrecoverable I/O
_ABNORMAL_STORAGE: list[str] = [
    "lustre mount failed",
    "lustre mount error",
    "data tlb error interrupt",
    "data storage interrupt",
    "input/output error",
    # bare "i/o error" only — must NOT match "ciod: LOGIN chdir … No such file"
    # (that is caught by NORMAL patterns below)
    "i/o error",
]

# Hardware / instruction faults that require operator intervention
_ABNORMAL_HW: list[str] = [
    "illegal instruction",
    "illegal operation",
    # "machine check interrupt" at FATAL with no surrounding context is a crash.
    # Do NOT use bare "machine check" — INFO-level DCR timeout is benign (see NORMAL).
    "machine check interrupt",
    "external interrupt",
]

ABNORMAL_CONTENT_PATTERNS: list[str] = (
    _ABNORMAL_TERMINATION
    + _ABNORMAL_NETWORK
    + _ABNORMAL_STORAGE
    + _ABNORMAL_HW
)

# ---------------------------------------------------------------------------
# Normal: diagnostic / self-healing lines that look alarming but are benign
# ---------------------------------------------------------------------------
'''
NORMAL_CONTENT_PATTERNS: list[str] = [
    # RTS internal diagnostics (dump, not crash)
    # ------------------------------------------------------------------
    # RTS / kernel register dump lines (handled exceptions, not crashes)
    # ------------------------------------------------------------------
    "rts internal error",
    "instruction address",          # "instruction address: 0x…"
    "data address",                 # "data address: 0x…"
    "exception syndrome register",
    "special purpose registers",
    "machine state register",
    "alignment exceptions",
    "generating core",

    # "iar … dear …" — instruction-address / data-exception-address dump
    # Seen as: "RAS KERNEL INFO iar 00105e84 dear 024701dc"
    "iar ",                         # trailing space avoids matching mid-word

    # "program interrupt" and "program interrupt:" — register dump, not a crash
    # Seen as: "RAS KERNEL INFO program interrupt"
    "program interrupt",

    # ------------------------------------------------------------------
    # MMCS / IDO transport-layer diagnostics (benign recurring probe)
    # ------------------------------------------------------------------
    # "idoproxydb hit ASSERT condition: ASSERT expression=0 Source line=1043 …"
    # Recurs hundreds of times, always the same SendPacket probe — not a fault.
    "idoproxydb hit assert condition",

    # ------------------------------------------------------------------
    # Machine-check / DCR diagnostics reported at INFO (self-healing)
    # ------------------------------------------------------------------
    # "MACHINE CHECK DCR read timeout (mc=e08x iar 0x… lr 0x…)"
    # Reported at INFO; the kernel handled it and continued.
    "machine check dcr read timeout",

    # ------------------------------------------------------------------
    # Self-corrected hardware errors
    # ------------------------------------------------------------------
    "detected and corrected",
    "cache parity error corrected",
    "data cache search parity error detected. attempting to correct",
    "ddr error",                    # "ddr error(s) detected and corrected"
    "ddr errors",
    "ce sym",
    "suppressing further interrupts",
    "correctable ddr",              # "… correctable ddr." in timing summary lines
    "rbs signal handler",           # timing summary: "0 µs in the rbs signal handler …"

    # ------------------------------------------------------------------
    # Register / CPU state dump lines (FATAL severity but benign dumps)
    # ------------------------------------------------------------------
    "force load/store alignment",   # "force load/store alignment…0"
    "floating pt ex mode",          # "floating pt ex mode 1 enable…0"
    "machine check enable",
    "wait state enable",
    "disable store gathering",
    "icache prefetch threshold",
    "0 critical input interrupts",

    # ------------------------------------------------------------------
    # ciod lifecycle events that are NOT errors
    # ------------------------------------------------------------------
    # "ciod: LOGIN chdir(pwd) failed: No such file or directory" — workdir missing,
    # job proceeds or retries.  Must come BEFORE any broader "ciod:" abnormal catch.
    "login chdir",                  # covers both "LOGIN chdir(pwd)" and "LOGIN chdir(/path)"
    "chdir(pwd) failed: no such file or directory",
    # "ciod: pollControlDescriptors: Detected the debugger died." — debugger detach
    "pollcontrolDescriptors",
    "detected the debugger died",
    # "ciod: error loading …" and "ciod: generated …" — load/generation diagnostics
    "ciod: error loading",
    "ciod: generated",

    # ------------------------------------------------------------------
    # Tree network / resync events (self-healing, not packet loss)
    # ------------------------------------------------------------------
    # "1 tree receiver 1 in re-synch state event(s) (dcr 0x0185) detected over N seconds"
    "tree receiver",
    "in re-synch state",

    # ------------------------------------------------------------------
    # Discovery / topology probes
    # ------------------------------------------------------------------
    # "NULL DISCOVERY INFO Ido chip status changed: … ip=… status=M …"
    "ido chip status changed",
    "null discovery info",          # component=DISCOVERY type=NULL probe lines

    # ------------------------------------------------------------------
    # Transient mount / network retries that succeed
    # ------------------------------------------------------------------
    "nfs mount failed",             # always followed by "retrying" — benign
    "retrying",
    "errno=0",                      # explicit success code in error path

    # ------------------------------------------------------------------
    # ASSERT condition that is a known-benign probe (MMCS transport)
    # ------------------------------------------------------------------
    "assert expression=0",          # only the expression=0 variant is benign
    "source line=1043",             # ties to the specific SendPacket probe location
]
'''
NORMAL_CONTENT_PATTERNS: list[str] = [
    # ------------------------------------------------------------------
    # RTS / kernel register dump lines (handled exceptions, not crashes)
    # ------------------------------------------------------------------
    "rts internal error",
    "instruction address",          # "instruction address: 0x…"
    "data address",                 # "data address: 0x…"
    "exception syndrome register",
    "special purpose registers",
    "machine state register",
    "alignment exceptions",
    "generating core",

    # "iar … dear …" — instruction-address / data-exception-address dump
    # Seen as: "RAS KERNEL INFO iar 00105e84 dear 024701dc"
    "iar ",                         # trailing space avoids matching mid-word

    # "program interrupt" and "program interrupt:" — register dump, not a crash
    # Seen as: "RAS KERNEL INFO program interrupt"
    "program interrupt",

    # ------------------------------------------------------------------
    # MMCS / IDO transport-layer diagnostics (benign recurring probe)
    # ------------------------------------------------------------------
    # "idoproxydb hit ASSERT condition: ASSERT expression=0 Source line=1043 …"
    # Recurs hundreds of times, always the same SendPacket probe — not a fault.
    "idoproxydb hit assert condition",

    # ------------------------------------------------------------------
    # Machine-check / DCR diagnostics reported at INFO (self-healing)
    # ------------------------------------------------------------------
    # "MACHINE CHECK DCR read timeout (mc=e08x iar 0x… lr 0x…)"
    # Reported at INFO; the kernel handled it and continued.
    "machine check dcr read timeout",

    # ------------------------------------------------------------------
    # Self-corrected hardware errors
    # ------------------------------------------------------------------
    "detected and corrected",
    "cache parity error corrected",
    "data cache search parity error detected. attempting to correct",
    "ddr error",                    # "ddr error(s) detected and corrected"
    "ddr errors",
    "ce sym",
    "suppressing further interrupts",
    "correctable ddr",              # "… correctable ddr." in timing summary lines
    "rbs signal handler",           # timing summary: "0 µs in the rbs signal handler …"

    # ------------------------------------------------------------------
    # Register / CPU state dump lines (FATAL severity but benign dumps)
    # ------------------------------------------------------------------
    "force load/store alignment",   # "force load/store alignment…0"
    "floating pt ex mode",          # "floating pt ex mode 1 enable…0"
    "machine check enable",
    "wait state enable",
    "disable store gathering",
    "icache prefetch threshold",
    "0 critical input interrupts",

    # ------------------------------------------------------------------
    # ciod lifecycle events that are NOT errors
    # ------------------------------------------------------------------
    # "ciod: LOGIN chdir(pwd) failed: No such file or directory" — workdir missing,
    # job proceeds or retries.  Must come BEFORE any broader "ciod:" abnormal catch.
    "login chdir",                  # covers both "LOGIN chdir(pwd)" and "LOGIN chdir(/path)"
    "chdir(pwd) failed: no such file or directory",
    # "ciod: pollControlDescriptors: Detected the debugger died." — debugger detach
    "pollcontrolDescriptors",
    "detected the debugger died",
    # "ciod: error loading …" and "ciod: generated …" — load/generation diagnostics
    "ciod: error loading",
    "ciod: generated",

    # ------------------------------------------------------------------
    # PowerPC icbi (Instruction Cache Block Invalidate) store interrupts
    # ------------------------------------------------------------------
    # "data store interrupt caused by icbi.........0"
    # Appears at both INFO and FATAL severity — a handled PowerPC exception
    # from the icbi instruction, printed as part of a register dump.
    # The trailing "…0" indicates the interrupt was serviced (return code 0).
    # Never indicates a crash or data loss.
    "data store interrupt caused by icbi",

    # ------------------------------------------------------------------
    # Discovery / topology warnings during node enumeration
    # ------------------------------------------------------------------
    # "NULL DISCOVERY WARNING Node card is not fully functional"
    # Emitted during topology probing when a node card reports degraded
    # status; the system continues operating. Not an active runtime fault.
    "node card is not fully functional",

    # ------------------------------------------------------------------
    # Tree network / resync events (self-healing, not packet loss)
    # ------------------------------------------------------------------
    # "1 tree receiver 1 in re-synch state event(s) (dcr 0x0185) detected over N seconds"
    "tree receiver",
    "in re-synch state",

    # ------------------------------------------------------------------
    # Discovery / topology probes
    # ------------------------------------------------------------------
    # "NULL DISCOVERY INFO Ido chip status changed: … ip=… status=M …"
    "ido chip status changed",
    "null discovery info",          # component=DISCOVERY type=NULL probe lines

    # ------------------------------------------------------------------
    # Transient mount / network retries that succeed
    # ------------------------------------------------------------------
    "nfs mount failed",             # always followed by "retrying" — benign
    "retrying",
    "errno=0",                      # explicit success code in error path

    # ------------------------------------------------------------------
    # ASSERT condition that is a known-benign probe (MMCS transport)
    # ------------------------------------------------------------------
    "assert expression=0",          # only the expression=0 variant is benign
    "source line=1043",             # ties to the specific SendPacket probe location
]

EXPECTED_CLASSIFICATION_HINTS = {
    "iar ... dear ...": 0,
    "program interrupt": 0,
    "idoproxydb hit ASSERT condition": 1,
    "tree receiver ... re-synch state": 0,
    "ciod: pollControlDescriptors: Detected the debugger died": 0,
    "Ido chip status changed": 0,
    "force load/store alignment": 0,
    "floating pt ex mode 1 enable": 0,
    "MACHINE CHECK DCR read timeout": 1,
    "correctable ddr": 0,
    "LOGIN chdir(... No such file or directory)": 0,
}


## Results

EVALUATION RESULTS
======================================================================

Evaluation Metrics:
  Accuracy  : 1.0000
  Precision : 1.0000
  Recall    : 1.0000
  F1-score  : 1.0000
  ROC-AUC   : 1.0000

Confusion Matrix:
                  Pred Normal  Pred Abnormal
  Actual Normal         372            0
  Actual Abnormal         0           29

Timing:
  Overall time   : 13.16 sec
  Avg per log    : 32.81 ms

Memory:
  Start RSS      : 7703.3 MB
  End RSS        : 8114.5 MB
  Peak RSS       : 8114.5 MB
  Delta RSS      : 411.2 MB
  System RAM     : 9.8%

Routing:
  Rule-based     : 393
  DeepSeek       : 8
  Total          : 401

Classification Report:
              precision    recall  f1-score   support

  Normal (0)       1.00      1.00      1.00       372
Abnormal (1)       1.00      1.00      1.00        29

    accuracy                           1.00       401
   macro avg       1.00      1.00      1.00       401
weighted avg       1.00      1.00      1.00       401


Misclassified: 0 / 401
