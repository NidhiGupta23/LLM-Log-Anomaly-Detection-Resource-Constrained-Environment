"""
eval_bgl.py — Evaluate / predict with a quantised BGL GGUF model.

Tracks: time_taken_seconds, avg_time_per_log_ms, throughput_logs_per_second,
        cpu_ram_used_mb, peak_cpu_memory_mb, model_used.

Supports two input formats automatically:
  A) Labelled   : label + 9 fields  (standard BGL / your train-val-test splits)
  B) Unlabelled : 9 fields, no leading label column

──────────────────────────────────────────────────────────────
THREE INFERENCE MODES  (select with --mode)
──────────────────────────────────────────────────────────────
  full      Send the complete raw log line to the LLM.
            e.g. "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 ... message"

  extended  Send node + type + component + level + content.
            e.g. "[R02-M1-N0] type=RAS  comp=KERNEL  level=FATAL  <message>"

  minimal   Send node + component + level + content only.
            e.g. "[R02-M1-N0] comp=KERNEL  level=FATAL  <message>"
──────────────────────────────────────────────────────────────

Usage:
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test ../bgl_splits/test.log --mode full
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test test.log             --mode extended
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test test.log             --mode minimal
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test test.log             --mode minimal --debug 5
"""

import argparse
import json
import os
import re
import sys
import time
import tracemalloc

# ---------------------------------------------------------------------------
# Optional dependencies — graceful fallback if missing
# ---------------------------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ===========================================================================
# INFERENCE MODES
# ===========================================================================

MODES = ("full", "extended", "minimal", "4column")

"""
BGL column layout
─────────────────────────────────────────────────────────────────
Labelled  (10 fields, split on first 9 whitespace boundaries):
  idx  0  label        ('-' = normal, anything else = abnormal)
  idx  1  timestamp    (Unix epoch)
  idx  2  date
  idx  3  time
  idx  4  node         e.g. R02-M1-N0-C:J12-U11
  idx  5  type         e.g. RAS / KERNEL / APP …
  idx  6  location     e.g. R02-M1-N0
  idx  7  component    e.g. MMCS / KERNEL / …
  idx  8  level        e.g. INFO / WARN / FATAL / ERROR …
  idx  9  content      (remainder of line)

Unlabelled (9 fields, no leading label):
  idx  0  timestamp
  idx  1  date
  idx  2  time
  idx  3  node
  idx  4  type
  idx  5  location
  idx  6  component
  idx  7  level
  idx  8  content
"""

_N_LABELLED   = 10   # number of split fields for labelled lines
_N_UNLABELLED = 9    # number of split fields for unlabelled lines

def percentile(values: list, q: float) -> float:
    """
    Compute percentile q from a list of numeric values.
    q should be between 0 and 100.
    """
    if not values:
        return 0.0

    values = sorted(values)
    k = (len(values) - 1) * (q / 100)
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)

# ===========================================================================
# METRICS COLLECTION
# ===========================================================================

class PerfMetrics:
    """
    Collects performance and memory metrics over an inference run.

    Tracked values
    --------------
    time_taken_seconds      : wall-clock seconds for the full inference loop
    avg_time_per_log_ms     : mean milliseconds spent per log line
    throughput_logs_per_sec : log lines processed per second
    cpu_ram_used_mb         : RSS memory delta (end - start) in MB
    peak_cpu_memory_mb      : peak RSS (resident set size) in MB during the run
    model_used              : basename of the GGUF file
    """

    def __init__(self, gguf_path: str):
        self.model_used          = os.path.basename(gguf_path)
        self._t_start            = None
        self._t_end              = None
        self._n_samples          = 0
        self.cpu_ram_used_mb     = 0.0
        self.peak_cpu_memory_mb  = 0.0
        self._rss_at_start       = 0.0
        self.cpu_cores_available = os.cpu_count() or 0
        self.cpu_time_used_seconds = 0.0
        self.avg_cpu_cores_used = 0.0
        self._cpu_time_start = 0.0
        self._rss_samples_mb = []
        self._rss_delta_samples_mb = []

        self.p95_cpu_memory_mb = 0.0
        self.p99_cpu_memory_mb = 0.0
        self.p95_cpu_ram_delta_mb = 0.0
        self.p99_cpu_ram_delta_mb = 0.0
        self.total_prompt_tokens     = 0
        self.total_completion_tokens = 0


    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self):
        tracemalloc.start()
        if _HAS_PSUTIL:
            #self._rss_at_start = psutil.Process(os.getpid()).memory_info().rss
            proc = psutil.Process(os.getpid())
            self._rss_at_start = proc.memory_info().rss
            cpu_times = proc.cpu_times()
            self._cpu_time_start = cpu_times.user + cpu_times.system
        self._t_start = time.perf_counter()

    def sample_memory(self):
        """
        Sample current process RSS memory.

        rss_mb:
            Absolute process memory, including Python, llama.cpp, and loaded model.

        delta_mb:
            Extra memory compared with the start of inference.
            Since perf.start() is called after model loading in your code,
            this mostly represents inference-time memory growth.
        """
        if not _HAS_PSUTIL:
            return

        proc = psutil.Process(os.getpid())
        rss_mb = proc.memory_info().rss / 1024 / 1024
        start_mb = self._rss_at_start / 1024 / 1024
        delta_mb = max(0.0, rss_mb - start_mb)

        self._rss_samples_mb.append(rss_mb)
        self._rss_delta_samples_mb.append(delta_mb)



    def stop(self, n_samples: int):
        self._t_end     = time.perf_counter()
        self._n_samples = n_samples

        _, peak_traced = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_traced_mb = peak_traced / 1024 / 1024

        if _HAS_PSUTIL:
            proc         = psutil.Process(os.getpid())
            cpu_times = proc.cpu_times()
            cpu_time_end = cpu_times.user + cpu_times.system
            self.cpu_time_used_seconds = max(0.0, cpu_time_end - self._cpu_time_start)

            wall_time = self.time_taken_seconds
            self.avg_cpu_cores_used = (
                self.cpu_time_used_seconds / wall_time if wall_time > 0 else 0.0
            )
            rss_end_mb   = proc.memory_info().rss / 1024 / 1024
            rss_start_mb = self._rss_at_start / 1024 / 1024
            self.cpu_ram_used_mb    = max(0.0, rss_end_mb - rss_start_mb)
            self.peak_cpu_memory_mb = max(peak_traced_mb, rss_end_mb)
        else:
            self.cpu_ram_used_mb    = peak_traced_mb
            self.peak_cpu_memory_mb = peak_traced_mb

        if self._rss_samples_mb:
            self.p95_cpu_memory_mb = percentile(self._rss_samples_mb, 95)
            self.p99_cpu_memory_mb = percentile(self._rss_samples_mb, 99)

        if self._rss_delta_samples_mb:
            self.p95_cpu_ram_delta_mb = percentile(self._rss_delta_samples_mb, 95)
            self.p99_cpu_ram_delta_mb = percentile(self._rss_delta_samples_mb, 99)

    # Call this once per inference result
    def record_tokens(self, usage: dict):
        self.total_prompt_tokens     += usage.get("prompt_tokens", 0)
        self.total_completion_tokens += usage.get("completion_tokens", 0)

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------
    @property
    def time_taken_seconds(self) -> float:
        if self._t_start is None or self._t_end is None:
            return 0.0
        return self._t_end - self._t_start

    @property
    def avg_time_per_log_ms(self) -> float:
        if self._n_samples == 0:
            return 0.0
        return (self.time_taken_seconds / self._n_samples) * 1000

    @property
    def throughput_logs_per_second(self) -> float:
        if self.time_taken_seconds == 0:
            return 0.0
        return self._n_samples / self.time_taken_seconds

    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def avg_prompt_tokens(self) -> float:
        return self.total_prompt_tokens / self._n_samples if self._n_samples else 0.0

    @property
    def avg_completion_tokens(self) -> float:
        return self.total_completion_tokens / self._n_samples if self._n_samples else 0.0



    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "model_used"                : self.model_used,
            "time_taken_seconds"        : round(self.time_taken_seconds, 3),
            "avg_time_per_log_ms"       : round(self.avg_time_per_log_ms, 3),
            "throughput_logs_per_second": round(self.throughput_logs_per_second, 2),
            "cpu_ram_used_mb"           : round(self.cpu_ram_used_mb, 2),
            "peak_cpu_memory_mb"        : round(self.peak_cpu_memory_mb, 2),
            "cpu_cores_available"      : self.cpu_cores_available,
            "cpu_time_used_seconds"    : round(self.cpu_time_used_seconds, 3),
            "avg_cpu_cores_used"       : round(self.avg_cpu_cores_used, 2),
            "total_prompt_tokens"       : self.total_prompt_tokens,
            "total_completion_tokens"   : self.total_completion_tokens,
            "total_tokens"              : self.total_tokens,
            "avg_prompt_tokens"         : round(self.avg_prompt_tokens, 1),
            "avg_completion_tokens"     : round(self.avg_completion_tokens, 1),
            "p95_cpu_memory_mb"       : round(self.p95_cpu_memory_mb, 2),
            "p99_cpu_memory_mb"       : round(self.p99_cpu_memory_mb, 2),
            "p95_cpu_ram_delta_mb"    : round(self.p95_cpu_ram_delta_mb, 2),
            "p99_cpu_ram_delta_mb"    : round(self.p99_cpu_ram_delta_mb, 2),
        }

    def print_summary(self):
        d = self.as_dict()
        print()
        print("=" * 55)
        print("  PERFORMANCE METRICS")
        print("=" * 55)
        print(f"  Model                  : {d['model_used']}")
        print(f"  Time taken             : {d['time_taken_seconds']:.3f} s")
        print(f"  Avg time per log       : {d['avg_time_per_log_ms']:.1f} ms")
        print(f"  Throughput             : {d['throughput_logs_per_second']:.2f} logs/s")
        print(f"  CPU RAM consumed       : {d['cpu_ram_used_mb']:.1f} MB"
              + ("" if _HAS_PSUTIL else "  (install psutil: pip install psutil)"))
        print(f"  Peak CPU RAM           : {d['peak_cpu_memory_mb']:.1f} MB")
        print(f"  CPU cores available    : {d['cpu_cores_available']}")
        print(f"  CPU time used          : {d['cpu_time_used_seconds']:.3f} s")
        print(f"  Avg CPU cores used     : {d['avg_cpu_cores_used']:.2f}")
        print(f"  P95 CPU memory         : {d['p95_cpu_memory_mb']:.1f} MB")
        print(f"  P99 CPU memory         : {d['p99_cpu_memory_mb']:.1f} MB")
        print(f"  P95 RAM delta          : {d['p95_cpu_ram_delta_mb']:.1f} MB")
        print(f"  P99 RAM delta          : {d['p99_cpu_ram_delta_mb']:.1f} MB")
        # Token usage details
        print("=" * 55)
        print("  TOKEN USAGE")
        print("=" * 55)
        print(f"  {'Prompt tokens':<25} {self.total_prompt_tokens:>10,}")
        print(f"  {'Completion tokens':<25} {self.total_completion_tokens:>10,}")
        print(f"  {'Total tokens':<25} {self.total_tokens:>10,}")
        print("-" * 55)
        print(f"  {'Avg prompt / log':<25} {self.avg_prompt_tokens:>9.1f}")
        print(f"  {'Avg completion / log':<25} {self.avg_completion_tokens:>9.1f}")
        print(f"  {'Avg total / log':<25} {self.avg_prompt_tokens + self.avg_completion_tokens:>9.1f}")
        print("-" * 55)
        pct_prompt     = 100 * self.total_prompt_tokens     / max(1, self.total_tokens)
        pct_completion = 100 * self.total_completion_tokens / max(1, self.total_tokens)
        print(f"  {'Prompt share':<25} {pct_prompt:>9.1f} %")
        print(f"  {'Completion share':<25} {pct_completion:>9.1f} %")
        print("=" * 55)


# ===========================================================================
# PARSING
# ===========================================================================

def _is_unix_timestamp(token: str) -> bool:
    try:
        return int(token) > 100_000_000
    except ValueError:
        return False


def detect_format(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            return "unlabelled" if _is_unix_timestamp(line.split()[0]) else "labelled"
    return "labelled"


def parse_labelled_line(raw: str) -> dict:
    """
    Split a labelled BGL line into its 10 logical columns.

    Returns a dict with keys:
      label_tag, node, type, component, level, content
    Falls back gracefully for short lines.
    """
    parts = raw.strip().split(None, _N_LABELLED - 1)
    if len(parts) < _N_LABELLED:
        return {
            "label_tag": "-",
            "node": "UNKNOWN", "type": "UNKNOWN",
            "component": "UNKNOWN", "level": "UNKNOWN",
            "content": raw.strip(),
        }
    return {
        "label_tag": parts[0],
        "node"     : parts[4],   # e.g. R02-M1-N0-C:J12-U11
        "type"     : parts[5],   # e.g. RAS
        "component": parts[7],   # e.g. KERNEL
        "level"    : parts[8],   # e.g. FATAL
        "content"  : parts[9],   # remainder of line
    }


def parse_unlabelled_line(raw: str) -> dict:
    """
    Split an unlabelled BGL line (no leading label column).

    Returns a dict with keys:
      node, type, component, level, content
    Falls back gracefully for short lines.
    """
    parts = raw.strip().split(None, _N_UNLABELLED - 1)
    if len(parts) < _N_UNLABELLED:
        return {
            "node": "UNKNOWN", "type": "UNKNOWN",
            "component": "UNKNOWN", "level": "UNKNOWN",
            "content": raw.strip(),
        }
    return {
        "node"     : parts[4],   # e.g. R02-M1-N0-C:J12-U11
        "type"     : parts[5],   # e.g. RAS
        "component": parts[6],   # e.g. KERNEL
        "level"    : parts[7],   # e.g. FATAL
        "content"  : parts[8],   # remainder of line
    }


# ===========================================================================
# PROMPT BUILDING  (one function per mode + dispatcher)
# ===========================================================================

_RULES_TEMPLATE = """\
Classify this BGL supercomputer log line as 0 (normal) or 1 (abnormal).

0 NORMAL  : informational messages, corrected hardware errors, DDR/CE errors
            corrected, cache parity corrected, retries, recovery messages,
            core file generation, alignment exceptions, routine warnings,
            register dumps, diagnostic messages.
1 ABNORMAL: kernel terminated, RTS panic, Lustre/NFS mount FAILED, link
            severed, connection reset or timeout, fatal machine-check
            interrupt, fatal hardware errors, errors that halt execution.

Rules:
- Judge on content, not severity label alone.
- Some FATAL log lines are normal recovery/diagnostic events.
- Reply with ONLY the single digit 0 or 1. Nothing else.

Log: {log_repr}
Label:"""


# ── Mode helpers ─────────────────────────────────────────────────────────────

def _repr_full(raw: str, _parsed: dict) -> str:
    """Mode 'full': send the entire raw log line verbatim."""
    return raw.strip()


def _repr_extended(raw: str, parsed: dict) -> str:
    """Mode 'extended': node  type  component  level  content."""
    return (
        f"[{parsed['node']}]  "
        f"type={parsed['type']}  "
        f"comp={parsed['component']}  "
        f"level={parsed['level']}  "
        f"{parsed['content']}"
    )


def _repr_minimal(raw: str, parsed: dict) -> str:
    """Mode 'minimal': node  component  level  content."""
    return (
        f"[{parsed['node']}]  "
        f"comp={parsed['component']}  "
        f"level={parsed['level']}  "
        f"{parsed['content']}"
    )

def _repr_4column(raw: str, parsed: dict) -> str:
    """Mode '5column': type  component  level  content."""
    return (
        f"comp={parsed['component']}  "
        f"{parsed['content']}"
    )

_MODE_REPR = {
    "full"    : _repr_full,
    "extended": _repr_extended,
    "minimal" : _repr_minimal,
    "4column" : _repr_4column,
}


def build_prompt(raw_line: str, unlabelled: bool, mode: str) -> str:
    """
    Build the full prompt string for one log line.

    Parameters
    ----------
    raw_line   : original text of the log line
    unlabelled : True when the file has no leading label column
    mode       : one of 'full', 'extended', 'minimal', '4column'
    """
    if mode not in _MODE_REPR:
        raise ValueError(f"Unknown mode '{mode}'. Choose from: {MODES}")

    parsed = (parse_unlabelled_line(raw_line) if unlabelled
              else parse_labelled_line(raw_line))

    log_repr = _MODE_REPR[mode](raw_line, parsed)
    return _RULES_TEMPLATE.format(log_repr=log_repr)


# ===========================================================================
# ANSWER EXTRACTION
# ===========================================================================

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_prediction(raw_output: str) -> str:
    """
    Strip <think>...</think> traces, then find the last '0' or '1'.

    Steps:
      1. Remove <think> block.
      2. Direct equality check ('0' or '1' after stripping).
      3. Regex: digit NOT surrounded by other digits.
      4. Character scan fallback.
      5. Return '?' if nothing found.
    """
    cleaned = _THINK_RE.sub("", raw_output).strip()

    if cleaned in ("0", "1"):
        return cleaned

    hits = re.findall(r"(?<!\d)([01])(?!\d)", cleaned)
    if hits:
        return hits[-1]

    for ch in reversed(cleaned):
        if ch in ("0", "1"):
            return ch

    return "?"


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_jsonl(path: str, limit: int) -> list:
    records = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            records.append({"raw_line": r["raw_line"], "label": r["label"]})
            if limit and len(records) >= limit:
                break
    return records


def load_log(path: str, limit: int, unlabelled: bool) -> list:
    records = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            if unlabelled:
                records.append({"raw_line": raw, "label": None})
            else:
                parsed = parse_labelled_line(raw)
                label  = "0" if parsed["label_tag"] == "-" else "1"
                records.append({"raw_line": raw, "label": label})
            if limit and len(records) >= limit:
                break
    return records


def load_data(path: str, limit: int) -> tuple:
    if path.endswith(".jsonl"):
        return load_jsonl(path, limit), False
    unlabelled = detect_format(path) == "unlabelled"
    return load_log(path, limit, unlabelled), unlabelled


# ===========================================================================
# CLASSIFICATION METRICS
# ===========================================================================

def compute_metrics(y_true: list, y_pred: list) -> dict:
    tp = sum(t == "1" and p == "1" for t, p in zip(y_true, y_pred))
    tn = sum(t == "0" and p == "0" for t, p in zip(y_true, y_pred))
    fp = sum(t == "0" and p == "1" for t, p in zip(y_true, y_pred))
    fn = sum(t == "1" and p == "0" for t, p in zip(y_true, y_pred))
    accuracy  = (tp + tn) / len(y_true) if y_true else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return {
        "accuracy": accuracy, "precision": precision,
        "recall": recall,     "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "total": len(y_true),
        "invalid_predictions": sum(1 for p in y_pred if p not in ("0", "1")),
    }


def print_eval_metrics(m: dict):
    print()
    print("=" * 55)
    print("  EVALUATION RESULTS")
    print("=" * 55)
    print(f"  Samples evaluated : {m['total']}")
    print(f"  Invalid outputs   : {m['invalid_predictions']}")
    print()
    print(f"  Accuracy          : {m['accuracy']:.4f}  ({m['tp']+m['tn']}/{m['total']})")
    print(f"  Precision         : {m['precision']:.4f}")
    print(f"  Recall            : {m['recall']:.4f}")
    print(f"  F1 Score          : {m['f1']:.4f}")
    print()
    print("  Confusion matrix (rows=actual, cols=predicted):")
    print(f"               Pred 0   Pred 1")
    print(f"  Actual 0  :  {m['tn']:>6}   {m['fp']:>6}")
    print(f"  Actual 1  :  {m['fn']:>6}   {m['tp']:>6}")
    print("=" * 55)


def print_predict_summary(y_pred: list):
    n0 = y_pred.count("0")
    n1 = y_pred.count("1")
    nb = sum(1 for p in y_pred if p not in ("0", "1"))
    print()
    print("=" * 55)
    print("  PREDICTION SUMMARY  (unlabelled mode)")
    print("=" * 55)
    print(f"  Total lines  : {len(y_pred)}")
    print(f"  Normal  (0)  : {n0}  ({100*n0/max(1,len(y_pred)):.1f}%)")
    print(f"  Abnormal (1) : {n1}  ({100*n1/max(1,len(y_pred)):.1f}%)")
    print(f"  Invalid      : {nb}")
    print("=" * 55)


# ===========================================================================
# INFERENCE MODE DESCRIPTION
# ===========================================================================

_MODE_DESC = {
    "full"    : "complete raw log line",
    "extended": "node + type + component + level + content",
    "minimal" : "node + component + level + content",
    "4column" : "type + component + level + content",
}


def _show_mode_example(raw_line: str, unlabelled: bool, mode: str):
    """Print a sample prompt snippet so the user can verify the chosen mode."""
    prompt = build_prompt(raw_line, unlabelled, mode)
    # Show only the 'Log:' line for brevity
    for line in prompt.splitlines():
        if line.startswith("Log:"):
            print(f"  Example prompt input : {line}")
            return


# ===========================================================================
# MAIN
# ===========================================================================

def run_eval(args):
    try:
        from llama_cpp import Llama
    except ImportError:
        sys.exit("[ERROR] pip install llama-cpp-python")

    for p in (args.gguf, args.test):
        if not os.path.exists(p):
            sys.exit(f"[ERROR] File not found: {p}")

    if args.mode not in MODES:
        sys.exit(f"[ERROR] --mode must be one of: {', '.join(MODES)}")

    records, unlabelled = load_data(args.test, args.limit)
    eval_mode  = "PREDICT (unlabelled)" if unlabelled else "EVALUATE (labelled)"
    n0_gt = sum(1 for r in records if r["label"] == "0") if not unlabelled else 0
    n1_gt = sum(1 for r in records if r["label"] == "1") if not unlabelled else 0

    print("=" * 55)
    print(f"  BGL ANOMALY CLASSIFIER — {eval_mode}")
    print("=" * 55)
    print(f"  GGUF        : {args.gguf}")
    print(f"  Input       : {args.test}")
    print(f"  Format      : {'unlabelled' if unlabelled else f'labelled  (normal:{n0_gt}  abnormal:{n1_gt})'}")
    print(f"  Samples     : {len(records)}")
    print(f"  Infer mode  : {args.mode}  ({_MODE_DESC[args.mode]})")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  GPU layers  : {args.gpu_layers}")
    print()

    # Show a sample of how the first line will be formatted for the LLM
    if records:
        _show_mode_example(records[0]["raw_line"], unlabelled, args.mode)
        print()

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    print("Loading GGUF model ...", flush=True)
    llm = Llama(
        model_path   = args.gguf,
        n_gpu_layers = args.gpu_layers,
        n_ctx        = args.ctx_size,
        n_threads    = args.threads,
        n_threads_batch = args.threads,
        n_batch         = args.n_batch,
        use_mmap        = True,
        verbose      = False,
    )

    # -----------------------------------------------------------------------
    # Inference loop
    # -----------------------------------------------------------------------
    perf        = PerfMetrics(gguf_path=args.gguf)
    y_pred      = []
    raw_outputs = []
    errors      = []
    report_n    = max(1, len(records) // 20)
    debug_left  = args.debug

    print("Running inference ...\n", flush=True)
    perf.start()

    for i, rec in enumerate(records):
        # Progress report
        if i > 0 and i % report_n == 0:
            partial_elapsed = time.perf_counter() - perf._t_start
            pct = 100 * i / len(records)
            eta = (partial_elapsed / i) * (len(records) - i)
            if not unlabelled:
                acc = sum(r["label"] == p
                          for r, p in zip(records[:i], y_pred)) / i
                print(f"  [{i:>5}/{len(records)}] {pct:.1f}%  "
                      f"acc={acc:.3f}  ETA {eta:.0f}s", flush=True)
            else:
                print(f"  [{i:>5}/{len(records)}] {pct:.1f}%  "
                      f"abnormal: {y_pred.count('1')}  ETA {eta:.0f}s", flush=True)

        prompt = build_prompt(rec["raw_line"], unlabelled, args.mode)

        try:
            out  = llm(
                prompt,
                max_tokens  = args.max_tokens,
                temperature = 0.0,
                echo        = False,
                stop        = ["\n", "<|endoftext|>", "</s>"],
            )
            raw  = out["choices"][0]["text"]
            pred = extract_prediction(raw)
            perf.record_tokens(out.get("usage", {}))
        except Exception as exc:
            raw, pred = "", "?"
            errors.append((i, str(exc)))

        y_pred.append(pred)
        raw_outputs.append(raw)
        perf.sample_memory()
        # Debug output
        if debug_left > 0:
            true_lbl = rec["label"] if not unlabelled else "N/A"
            print(f"\n  [DEBUG {i}] true={true_lbl}  pred={pred}")
            print(f"  prompt  : {repr(prompt[-200:])}")
            print(f"  raw out : {repr(raw[:300])}\n")
            debug_left -= 1

    perf.stop(n_samples=len(records))

    # -----------------------------------------------------------------------
    # Print results
    # -----------------------------------------------------------------------
    if unlabelled:
        print_predict_summary(y_pred)
    else:
        m = compute_metrics([r["label"] for r in records], y_pred)
        print_eval_metrics(m)

        if args.show_errors:
            mistakes = [(r, p, o) for r, p, o in zip(records, y_pred, raw_outputs)
                        if r["label"] != p][:args.show_errors]
            if mistakes:
                print(f"\n  -- First {len(mistakes)} mispredictions --")
                for rec, pred, raw in mistakes:
                    print(f"  true={rec['label']} pred={pred}  "
                          f"{rec['raw_line'][:100]}")
                    if args.debug:
                        print(f"    raw: {repr(raw[:200])}")

    perf.print_summary()

    # -----------------------------------------------------------------------
    # Save output JSONL
    # -----------------------------------------------------------------------
    out_path = args.output or ("predictions.jsonl" if unlabelled else "")
    if out_path:
        perf_dict = perf.as_dict()
        eval_dict = (compute_metrics([r["label"] for r in records], y_pred)
                     if not unlabelled else {})

        with open(out_path, "w", encoding="utf-8") as fh:
            meta = {"_meta": True, "infer_mode": args.mode, **perf_dict}
            if eval_dict:
                meta.update(eval_dict)
            fh.write(json.dumps(meta) + "\n")

            for rec, pred, raw in zip(records, y_pred, raw_outputs):
                row = {"label": pred, "line": rec["raw_line"]}
                if not unlabelled:
                    row["true_label"] = rec["label"]
                    row["correct"]    = rec["label"] == pred
                fh.write(json.dumps(row) + "\n")

        print(f"\n  Results saved -> {out_path}")
        print(f"  (first line in the file contains run metadata + performance stats)")

    if errors:
        print(f"\n[WARN] {len(errors)} inference errors (first 3):")
        for idx, msg in errors[:3]:
            print(f"  sample {idx}: {msg}")


# ===========================================================================
# CLI
# ===========================================================================

def _build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate or predict with a BGL GGUF anomaly classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gguf",       required=True,
                   help="Path to the GGUF model file.")
    p.add_argument("--test",       required=True,
                   help="Path to the log file (.log) or labelled JSONL (.jsonl).")
    p.add_argument("--mode",       choices=MODES, default="minimal",
                   help=(
                       "Inference mode — what is sent to the LLM.\n"
                       "  full      : complete raw log line\n"
                       "  extended  : node + type + component + level + content\n"
                       "  minimal   : node + component + level + content\n"
                       "  4column   : type + component + level + content\n"
                   ))
    p.add_argument("--gpu_layers", type=int, default=0)
    p.add_argument("--threads",    type=int, default=os.cpu_count() or 4)
    p.add_argument("--ctx_size",   type=int, default=1024)
    p.add_argument("--max_tokens", type=int, default=50)
    p.add_argument("--limit",      type=int, default=0,
                   help="Max number of lines to process (0 = all).")
    p.add_argument("--output",     default="",
                   help="JSONL output file. First line = metadata/metrics, "
                        "remaining lines = per-sample predictions.")
    p.add_argument("--show_errors", type=int, default=10)
    p.add_argument("--n_batch", type=int, default=512)
    p.add_argument("--debug",       type=int, default=0,
                   help="Print raw model output for the first N samples.")
    return p


if __name__ == "__main__":
    run_eval(_build_parser().parse_args())
