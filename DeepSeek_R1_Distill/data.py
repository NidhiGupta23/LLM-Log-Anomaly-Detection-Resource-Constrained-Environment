from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from config import Config


# ---------------------------------------------------------------------------
# Structured BGL log entry
# ---------------------------------------------------------------------------

@dataclass
class BGLEntry:
    """
    Represents one parsed line from a BGL (Blue Gene/L) log file.

    Fields (space-delimited columns, in order):
      timestamp   – Unix epoch integer
      date        – Calendar date as YYYY.MM.DD
      node        – Hardware node identifier
      time        – Full timestamp with microseconds (YYYY-MM-DD-HH.MM.SS.xxxxxx)
      node_repeat – Repeated node identifier from the structured dataset
      type        – High-level event type (e.g. RAS)
      component   – Subsystem that emitted the event (e.g. KERNEL, APP)
      level       – Severity level (e.g. FATAL, INFO, WARNING)
      content     – Free-text description of the event (remainder of the line)
      raw         – Original unparsed line, kept for fallback / debugging
    """
    timestamp:   str
    date:        str
    node:        str
    time:        str
    node_repeat: str
    type:        str
    component:   str
    level:       str
    content:     str
    raw:         str

    def to_classifier_text(self) -> str:
        """
        Canonical string fed to the rule engine and the LLM.
        Keeps all semantically useful fields; drops redundant node_repeat.
        """
        return (
            f"{self.type} {self.component} {self.level} {self.content}"
        )


def parse_bgl_line(line: str) -> Optional[BGLEntry]:
    """
    Parse a single BGL log line into a BGLEntry.

    Expected format (9 space-delimited columns, last column may contain spaces):
        <timestamp> <date> <node> <time> <node_repeat> <type> <component> <level> <content…>

    Returns None for blank lines or lines with fewer than 9 columns.
    """
    line = line.strip()
    if not line:
        return None

    parts = line.split(None, 8)   # split on whitespace, at most 9 tokens
    if len(parts) < 9:
        return None

    return BGLEntry(
        timestamp=parts[0],
        date=parts[1],
        node=parts[2],
        time=parts[3],
        node_repeat=parts[4],
        type=parts[5],
        component=parts[6],
        level=parts[7],
        content=parts[8],
        raw=line,
    )


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def load_logs(filepath: str, label: int) -> Tuple[List[BGLEntry], List[int]]:
    """
    Parse every non-blank line in *filepath* into a BGLEntry and assign
    *label* (0 = normal, 1 = abnormal) to each entry.

    Lines that cannot be parsed (fewer than 9 columns) are silently skipped
    and a warning counter is printed at the end.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    entries: List[BGLEntry] = []
    labels:  List[int]     = []
    skipped = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            entry = parse_bgl_line(line)
            if entry is None:
                skipped += 1
                continue
            entries.append(entry)
            labels.append(label)

    if skipped:
        print(f"  [warn] {filepath}: skipped {skipped} unparseable line(s)")

    return entries, labels


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def create_test_split(
    normal_entries:   List[BGLEntry],
    abnormal_entries: List[BGLEntry],
    config: Config,
) -> Tuple[List[BGLEntry], List[int]]:
    """
    Stratified test split over BGLEntry lists.
    Returns only the test partition — the model is not fine-tuned,
    so the train partition is intentionally discarded.
    """
    _, test_normal = train_test_split(
        normal_entries,
        test_size=config.test_size,
        random_state=config.random_seed,
    )
    _, test_abnormal = train_test_split(
        abnormal_entries,
        test_size=config.test_size,
        random_state=config.random_seed,
    )

    test_entries: List[BGLEntry] = test_normal + test_abnormal
    test_labels:  List[int]     = [0] * len(test_normal) + [1] * len(test_abnormal)

    # Shuffle to avoid ordering bias during evaluation
    rng = np.random.default_rng(config.random_seed)
    indices = rng.permutation(len(test_entries))
    test_entries = [test_entries[i] for i in indices]
    test_labels  = [test_labels[i]  for i in indices]

    print("\nDataset split:")
    print(f"  Test total    : {len(test_entries)}")
    print(f"  Test normal   : {test_labels.count(0)}")
    print(f"  Test abnormal : {test_labels.count(1)}")

    return test_entries, test_labels
