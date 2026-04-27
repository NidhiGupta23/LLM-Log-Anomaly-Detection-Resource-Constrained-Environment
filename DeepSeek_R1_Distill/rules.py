"""
Rule-based labelling for structured BGL log entries.

Rules operate on the parsed BGLEntry fields (level + content) rather than
raw text, which gives finer-grained control and avoids false matches on
metadata fields like node names.

Label priority:
  1. Content-based ABNORMAL rules  (highest confidence)
  2. Content-based NORMAL  rules
  3. None → fall through to the LLM
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from data import BGLEntry


# ---------------------------------------------------------------------------
# Content patterns — matched against entry.content (case-insensitive)
# ---------------------------------------------------------------------------

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
]

'''# Abnormal: network / socket / IPC failures
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
)'''
ABNORMAL_CONTENT_PATTERNS: list[str] = (
    _ABNORMAL_TERMINATION
    + _ABNORMAL_STORAGE
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
    "ciod: error loading",
    "ciod: generated",
    "info"
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rule_based_label(entry: "BGLEntry") -> Optional[int]:
    """
    Classify a parsed BGLEntry using deterministic rules.

    Returns:
        1  – ABNORMAL (high confidence)
        0  – NORMAL   (high confidence)
        None – uncertain; let the LLM decide
    """
    content_lower = entry.content.lower()

    # --- Abnormal content rules ---
    for pattern in ABNORMAL_CONTENT_PATTERNS:
        if pattern in content_lower:
            return 1

    # --- Normal content rules ---
    for pattern in NORMAL_CONTENT_PATTERNS:
        if pattern in content_lower:
            return 0

    return None
