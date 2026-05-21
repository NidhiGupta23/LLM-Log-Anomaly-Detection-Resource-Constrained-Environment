"""
eval_bgl.py — Evaluate / predict with a quantised BGL GGUF model.

Works with DeepSeek-R1-Distill reasoning models that output <think>...</think>
blocks before the final answer.  The script strips the reasoning trace and
extracts the last "0" or "1" in the response.

Supports two input formats automatically:
  A) Labelled   : label + 9 fields  (standard BGL / your train-val-test splits)
  B) Unlabelled : 9 fields, no leading label column
                  e.g.  "1135670489 2005.12.27 R37-M1-NC-C ..."

Usage:
  # Labelled test split
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test ../bgl_splits/test.log

  # Unlabelled file
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test Test_500_no_label_sorted.log

  # Debug: see raw model output for first 5 samples
  python eval_bgl.py --gguf bgl_1.5b_Q8_0.gguf --test ../bgl_splits/test.log --debug 5
"""

import argparse
import json
import os
import re
import sys
import time

# ---------------------------------------------------------------------------
# Prompt template — must match training exactly
# ---------------------------------------------------------------------------
_RULES = """\
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
- Reply with ONLY the single digit 0 or 1. Nothing else."""

# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
_N_LABELLED   = 10
_N_UNLABELLED = 9


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
    parts = raw.strip().split(None, _N_LABELLED - 1)
    if len(parts) < _N_LABELLED:
        return {"label_tag": "-", "component": "UNKNOWN", "content": raw.strip()}
    return {"label_tag": parts[0], "component": parts[7], "content": parts[9]}


def parse_unlabelled_line(raw: str) -> dict:
    parts = raw.strip().split(None, _N_UNLABELLED - 1)
    if len(parts) < _N_UNLABELLED:
        return {"component": "UNKNOWN", "content": raw.strip()}
    return {"component": parts[6], "content": parts[8]}


def format_for_prompt(parsed: dict) -> str:
    return f"[{parsed['component']}] {parsed['content']}"


def build_prompt(raw_line: str, unlabelled: bool) -> str:
    parsed = parse_unlabelled_line(raw_line) if unlabelled \
             else parse_labelled_line(raw_line)
    return f"{_RULES}\n\nLog: {format_for_prompt(parsed)}\nLabel:"


# ---------------------------------------------------------------------------
# Answer extraction  — handles DeepSeek-R1 reasoning traces
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def extract_prediction(raw_output: str) -> str:
    """
    Strip <think>...</think> reasoning traces, then find the last standalone
    '0' or '1' in the remaining text.

    Extraction order:
      1. Remove <think> block entirely.
      2. Look for a bare "0" or "1" (word-boundary match).
      3. Fall back to any digit character 0/1 anywhere in the output.
      4. Return "?" if nothing found.
    """
    cleaned = _THINK_RE.sub("", raw_output).strip()

    # Word-boundary search (most reliable — avoids matching "10" as "1")
    hits = re.findall(r"\b([01])\b", cleaned)
    if hits:
        return hits[-1]   # take the last occurrence

    # Loose fallback — any '0' or '1' character
    for ch in reversed(cleaned):
        if ch in ("0", "1"):
            return ch

    return "?"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
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


def print_metrics(m: dict, elapsed: float):
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
    print()
    print(f"  Wall time         : {elapsed:.1f}s")
    if elapsed > 0 and m['total'] > 0:
        print(f"  Throughput        : {m['total']/elapsed:.1f} samples/s")
    print("=" * 55)


def print_predict_summary(y_pred: list, elapsed: float):
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
    print(f"  Wall time    : {elapsed:.1f}s")
    if elapsed > 0 and len(y_pred) > 0:
        print(f"  Throughput   : {len(y_pred)/elapsed:.1f} samples/s")
    print("=" * 55)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_eval(args):
    try:
        from llama_cpp import Llama
    except ImportError:
        print("[ERROR] llama-cpp-python not installed.", file=sys.stderr)
        print("  CPU: pip install llama-cpp-python", file=sys.stderr)
        print("  GPU: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python "
              "--force-reinstall --no-cache-dir", file=sys.stderr)
        sys.exit(1)

    for p in (args.gguf, args.test):
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            sys.exit(1)

    records, unlabelled = load_data(args.test, args.limit)
    mode = "PREDICT (unlabelled)" if unlabelled else "EVALUATE (labelled)"

    print("=" * 55)
    print(f"  BGL ANOMALY CLASSIFIER — {mode}")
    print("=" * 55)
    print(f"  GGUF        : {args.gguf}")
    print(f"  Input       : {args.test}")
    print(f"  Format      : {'unlabelled' if unlabelled else 'labelled'}")
    print(f"  Samples     : {len(records)}")
    print(f"  Max tokens  : {args.max_tokens}  (reasoning model needs room to think)")
    print(f"  GPU layers  : {args.gpu_layers}")
    print()

    print("Loading GGUF model ...", flush=True)
    llm = Llama(
        model_path   = args.gguf,
        n_gpu_layers = args.gpu_layers,
        n_ctx        = args.ctx_size,
        n_threads    = args.threads,
        verbose      = False,
    )

    y_pred      = []
    raw_outputs = []
    errors      = []
    t0          = time.time()
    report_n    = max(1, len(records) // 20)
    debug_left  = args.debug

    print("Running inference ...\n", flush=True)

    for i, rec in enumerate(records):
        if i > 0 and i % report_n == 0:
            elapsed = time.time() - t0
            pct = 100 * i / len(records)
            eta = (elapsed / i) * (len(records) - i)
            if not unlabelled:
                acc = sum(r["label"] == p
                          for r, p in zip(records[:i], y_pred)) / i
                print(f"  [{i:>5}/{len(records)}] {pct:.1f}%  acc={acc:.3f}  ETA {eta:.0f}s",
                      flush=True)
            else:
                print(f"  [{i:>5}/{len(records)}] {pct:.1f}%  "
                      f"abnormal: {y_pred.count('1')}  ETA {eta:.0f}s", flush=True)

        prompt = build_prompt(rec["raw_line"], unlabelled)

        try:
            out     = llm(
                prompt,
                max_tokens  = args.max_tokens,   # enough room for <think> trace
                temperature = 0.0,               # greedy
                echo        = False,
                # Stop at the second newline after the digit — don't cut too early
                stop        = ["<|end|>", "<|endoftext|>", "</s>"],
            )
            raw = out["choices"][0]["text"]
            pred = extract_prediction(raw)
        except Exception as exc:
            raw  = ""
            pred = "?"
            errors.append((i, str(exc)))

        y_pred.append(pred)
        raw_outputs.append(raw)

        # Debug mode: print raw output for first N samples
        if debug_left > 0:
            true_lbl = rec['label'] if not unlabelled else "N/A"
            print(f"\n  --- DEBUG sample {i} | true={true_lbl} pred={pred} ---")
            print(f"  RAW OUTPUT: {repr(raw[:300])}")
            debug_left -= 1

    elapsed = time.time() - t0

    if unlabelled:
        print_predict_summary(y_pred, elapsed)
    else:
        m = compute_metrics([r["label"] for r in records], y_pred)
        print_metrics(m, elapsed)

        if args.show_errors:
            mistakes = [(r, p, o) for r, p, o in zip(records, y_pred, raw_outputs)
                        if r["label"] != p][:args.show_errors]
            if mistakes:
                print(f"\n  -- First {len(mistakes)} mispredictions --")
                for rec, pred, raw in mistakes:
                    print(f"  true={rec['label']} pred={pred}  "
                          f"{rec['raw_line'][:100]}")
                    if args.debug:
                        print(f"    raw_out: {repr(raw[:200])}")

    out_path = args.output or ("predictions.jsonl" if unlabelled else "")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            for rec, pred, raw in zip(records, y_pred, raw_outputs):
                row = {"raw_line": rec["raw_line"], "prediction": pred,
                       "raw_output": raw}
                if not unlabelled:
                    row["label"]   = rec["label"]
                    row["correct"] = rec["label"] == pred
                fh.write(json.dumps(row) + "\n")
        print(f"\n  Results saved -> {out_path}")

    if errors:
        print(f"\n[WARN] {len(errors)} inference errors (first 3):")
        for idx, msg in errors[:3]:
            print(f"  sample {idx}: {msg}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _build_parser():
    p = argparse.ArgumentParser(
        description="Evaluate or predict with a BGL GGUF anomaly classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--gguf",        required=True)
    p.add_argument("--test",        required=True)
    p.add_argument("--gpu_layers",  type=int, default=999)
    p.add_argument("--threads",     type=int, default=4)
    p.add_argument("--ctx_size",    type=int, default=1024,
                   help="Context size. Increase to 2048 if truncation warnings appear.")
    p.add_argument("--max_tokens",  type=int, default=512,
                   help="Max new tokens per sample. DeepSeek-R1 needs ~200-400 "
                        "for its <think> trace before the final digit.")
    p.add_argument("--limit",       type=int, default=0,
                   help="Process only the first N lines (0 = all).")
    p.add_argument("--output",      default="",
                   help="JSONL file for per-line predictions+raw output.")
    p.add_argument("--show_errors", type=int, default=10)
    p.add_argument("--debug",       type=int, default=0,
                   help="Print raw model output for the first N samples. "
                        "Use --debug 3 to verify extraction is working.")
    return p


if __name__ == "__main__":
    run_eval(_build_parser().parse_args())
