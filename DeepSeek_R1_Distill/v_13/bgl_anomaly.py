"""
BGL Log Anomaly Detector using DeepSeek-R1-Distill-Qwen-1.5B
=============================================================
Pipeline:
  1. Load normal & abnormal BGL log files (uploaded by user)
  2. Use DeepSeek-R1 reasoning to extract key patterns → save to
     normal_logs.txt / abnormal_logs.txt
  3. Load test .log file and classify each line as NORMAL or ABNORMAL
     using the extracted pattern knowledge
  4. Report: response time per detection, total execution time, memory usage

Dependencies (install once):
  pip install torch transformers accelerate psutil

Usage:
  python bgl_anomaly_detector.py \
      --normal  path/to/normal_bgl.log \
      --abnormal path/to/abnormal_bgl.log \
      --test    path/to/test.log

Optional flags:
  --model   "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"  (default)
  --device  "cpu" | "cuda" | "mps"                        (auto-detected)
  --batch   10   number of test log lines per inference batch
  --max_new_tokens 512
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path

import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_memory_mb() -> float:
    """Return current process RSS memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def read_log_lines(path: str) -> list[str]:
    """Read a log file; strip blank lines."""
    with open(path, "r", errors="replace") as f:
        lines = [ln.rstrip() for ln in f if ln.strip()]
    return lines


def chunk_lines(lines: list[str], chunk_size: int) -> list[list[str]]:
    """Split a list into chunks of at most chunk_size."""
    return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]


def extract_between_tags(text: str, tag: str) -> str:
    """Pull content between <tag> … </tag> (case-insensitive, dotall)."""
    pattern = rf"<{tag}>(.*?)</{tag}>"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text.strip()


# ─────────────────────────────────────────────
# Model wrapper
# ─────────────────────────────────────────────

class DeepSeekReasoner:
    """Thin wrapper around DeepSeek-R1-Distill-Qwen-1.5B."""

    def __init__(self, model_name: str, device: str, max_new_tokens: int = 512):
        self.device = device
        self.max_new_tokens = max_new_tokens

        print(f"\n[Model] Loading tokenizer from '{model_name}' …")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        print(f"[Model] Loading model (device={device}) …")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        print("[Model] Ready.\n")

    def generate(self, prompt: str) -> str:
        """Run a single-turn generation with the DeepSeek chat template."""
        # DeepSeek-R1-Distill uses the Qwen2/DeepSeek chat template
        messages = [{"role": "user", "content": prompt}]
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            # Fallback if the tokenizer has no chat template
            text = f"<|User|>{prompt}<|Assistant|>"

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=0.6,
                top_p=0.95,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the newly generated tokens
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


# ─────────────────────────────────────────────
# Phase 1 – Pattern extraction
# ─────────────────────────────────────────────

EXTRACTION_PROMPT = """\
You are an expert log analysis assistant. I will give you a sample of BGL \
(Blue Gene/L supercomputer) log lines that are classified as {label}.

Your task:
1. Identify the key keywords, phrases, error codes, severity levels, or \
patterns that indicate why these logs are {label}.
2. Output a concise bullet-point list of the most discriminative features.
3. Wrap your final answer in <answer> … </answer> tags.

Log samples:
{logs}

Reason step-by-step inside <think> … </think> tags, then give your \
<answer> bullet list.
"""

def extract_patterns(
    reasoner: DeepSeekReasoner,
    log_lines: list[str],
    label: str,   # "NORMAL" or "ABNORMAL"
    sample_size: int = 30,
) -> str:
    """Ask the model to identify discriminative patterns in the logs."""
    sample = "\n".join(log_lines[:sample_size])
    prompt = EXTRACTION_PROMPT.format(label=label, logs=sample)
    raw = reasoner.generate(prompt)
    # Try to pull just the answer section
    answer = extract_between_tags(raw, "answer")
    return answer


# ─────────────────────────────────────────────
# Phase 2 – Test-log classification
# ─────────────────────────────────────────────

CLASSIFICATION_PROMPT = """\
You are a BGL log anomaly detection expert.

NORMAL log patterns (keywords & features):
{normal_patterns}

ABNORMAL log patterns (keywords & features):
{abnormal_patterns}

Classify each of the following log lines as NORMAL or ABNORMAL. \
For each line output exactly one JSON object per line in the format:
{{"line_no": <int>, "verdict": "<NORMAL|ABNORMAL>", "reason": "<brief reason>"}}

Do NOT output anything else.

Log lines to classify (line numbers start at {start_no}):
{logs}
"""

def classify_batch(
    reasoner: DeepSeekReasoner,
    lines: list[str],
    start_no: int,
    normal_patterns: str,
    abnormal_patterns: str,
) -> list[dict]:
    """Classify a batch of log lines. Returns list of result dicts."""
    numbered = "\n".join(f"{start_no + i}: {ln}" for i, ln in enumerate(lines))
    prompt = CLASSIFICATION_PROMPT.format(
        normal_patterns=normal_patterns,
        abnormal_patterns=abnormal_patterns,
        start_no=start_no,
        logs=numbered,
    )
    raw = reasoner.generate(prompt)

    results = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            obj = json.loads(ln)
            results.append(obj)
        except json.JSONDecodeError:
            pass

    # Fallback: if parsing failed entirely, mark all lines as UNKNOWN
    if not results:
        for i, log_line in enumerate(lines):
            results.append({
                "line_no": start_no + i,
                "verdict": "UNKNOWN",
                "reason": "Model output could not be parsed",
            })
    return results


# ─────────────────────────────────────────────
# Metrics reporting
# ─────────────────────────────────────────────

def print_metrics(
    response_times: list[float],
    total_time: float,
    mem_start_mb: float,
    mem_end_mb: float,
    n_lines: int,
):
    print("\n" + "=" * 60)
    print("  PERFORMANCE METRICS")
    print("=" * 60)
    if response_times:
        avg_rt = sum(response_times) / len(response_times)
        min_rt = min(response_times)
        max_rt = max(response_times)
        print(f"  Anomaly detection response time")
        print(f"    Average  : {avg_rt:.3f}s per batch")
        print(f"    Fastest  : {min_rt:.3f}s")
        print(f"    Slowest  : {max_rt:.3f}s")
        print(f"    Per line : {(sum(response_times)/n_lines):.4f}s  ({n_lines} lines total)")
    print(f"  Total execution time : {total_time:.2f}s")
    print(f"  Memory at start      : {mem_start_mb:.1f} MB")
    print(f"  Memory at end        : {mem_end_mb:.1f} MB")
    print(f"  Memory delta         : {mem_end_mb - mem_start_mb:+.1f} MB")
    print("=" * 60 + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="BGL Log Anomaly Detector (DeepSeek-R1)")
    p.add_argument("--normal",   required=True, help="Path to normal BGL log file")
    p.add_argument("--abnormal", required=True, help="Path to abnormal BGL log file")
    p.add_argument("--test",     required=True, help="Path to test .log file")
    p.add_argument("--model",    default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
                   help="HuggingFace model name/path")
    p.add_argument("--device",   default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto' (default)")
    p.add_argument("--batch",    type=int, default=10,
                   help="Number of log lines per classification batch (default 10)")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Max tokens to generate per call (default 512)")
    p.add_argument("--sample",   type=int, default=30,
                   help="How many lines to sample from normal/abnormal for pattern extraction")
    return p.parse_args()


def resolve_device(pref: str) -> str:
    if pref == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return pref


def main():
    args = parse_args()
    global_start = time.perf_counter()
    mem_start = get_memory_mb()

    # ── Resolve device ──────────────────────────────────────────
    device = resolve_device(args.device)
    print(f"[Config] Device   : {device}")
    print(f"[Config] Model    : {args.model}")
    print(f"[Config] Batch    : {args.batch} lines")

    # ── Load logs ───────────────────────────────────────────────
    print(f"\n[Data] Reading normal logs  : {args.normal}")
    normal_lines   = read_log_lines(args.normal)
    print(f"       {len(normal_lines)} lines")

    print(f"[Data] Reading abnormal logs: {args.abnormal}")
    abnormal_lines = read_log_lines(args.abnormal)
    print(f"       {len(abnormal_lines)} lines")

    print(f"[Data] Reading test logs    : {args.test}")
    test_lines     = read_log_lines(args.test)
    print(f"       {len(test_lines)} lines")

    # ── Load model ──────────────────────────────────────────────
    reasoner = DeepSeekReasoner(args.model, device, args.max_new_tokens)

    # ── Phase 1 – Extract patterns ──────────────────────────────
    print("─" * 60)
    print("[Phase 1] Extracting NORMAL log patterns …")
    normal_patterns = extract_patterns(
        reasoner, normal_lines, "NORMAL", args.sample
    )
    Path("normal_logs.txt").write_text(normal_patterns, encoding="utf-8")
    print(f"  Saved → normal_logs.txt  ({len(normal_patterns)} chars)")

    print("[Phase 1] Extracting ABNORMAL log patterns …")
    abnormal_patterns = extract_patterns(
        reasoner, abnormal_lines, "ABNORMAL", args.sample
    )
    Path("abnormal_logs.txt").write_text(abnormal_patterns, encoding="utf-8")
    print(f"  Saved → abnormal_logs.txt ({len(abnormal_patterns)} chars)")

    # ── Phase 2 – Classify test logs ────────────────────────────
    print("\n" + "─" * 60)
    print(f"[Phase 2] Classifying {len(test_lines)} test log lines …")

    batches        = chunk_lines(test_lines, args.batch)
    all_results    = []
    response_times = []
    line_cursor    = 1

    for batch_idx, batch in enumerate(batches, 1):
        print(f"  Batch {batch_idx}/{len(batches)}  (lines {line_cursor}–{line_cursor+len(batch)-1})")
        t0 = time.perf_counter()
        results = classify_batch(
            reasoner,
            batch,
            line_cursor,
            normal_patterns,
            abnormal_patterns,
        )
        elapsed = time.perf_counter() - t0
        response_times.append(elapsed)
        all_results.extend(results)
        line_cursor += len(batch)
        print(f"    → {len(results)} results in {elapsed:.2f}s")

    # ── Save classification results ──────────────────────────────
    out_path = Path("classification_results.jsonl")
    with out_path.open("w", encoding="utf-8") as f:
        for r in all_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[Output] Classification results → {out_path}")

    # ── Summary ──────────────────────────────────────────────────
    normal_count   = sum(1 for r in all_results if r.get("verdict") == "NORMAL")
    abnormal_count = sum(1 for r in all_results if r.get("verdict") == "ABNORMAL")
    unknown_count  = len(all_results) - normal_count - abnormal_count

    print("\n[Summary]")
    print(f"  NORMAL   : {normal_count}")
    print(f"  ABNORMAL : {abnormal_count}")
    print(f"  UNKNOWN  : {unknown_count}")

    # Print first 5 results for a quick preview
    print("\n[Preview – first 5 classifications]")
    for r in all_results[:5]:
        verdict = r.get("verdict", "?")
        reason  = r.get("reason", "")[:80]
        print(f"  Line {r.get('line_no','?'):>4}  [{verdict}]  {reason}")

    # ── Metrics ──────────────────────────────────────────────────
    total_time = time.perf_counter() - global_start
    mem_end    = get_memory_mb()
    print_metrics(response_times, total_time, mem_start, mem_end, len(test_lines))

    # Also save a metrics summary
    metrics = {
        "total_lines_tested"          : len(test_lines),
        "normal_count"                : normal_count,
        "abnormal_count"              : abnormal_count,
        "unknown_count"               : unknown_count,
        "total_execution_time_s"      : round(total_time, 3),
        "avg_batch_response_time_s"   : round(sum(response_times)/len(response_times), 3) if response_times else 0,
        "min_batch_response_time_s"   : round(min(response_times), 3) if response_times else 0,
        "max_batch_response_time_s"   : round(max(response_times), 3) if response_times else 0,
        "avg_per_line_response_time_s": round(sum(response_times)/len(test_lines), 4) if test_lines else 0,
        "memory_start_mb"             : round(mem_start, 2),
        "memory_end_mb"               : round(mem_end, 2),
        "memory_delta_mb"             : round(mem_end - mem_start, 2),
        "device"                      : device,
        "model"                       : args.model,
    }
    Path("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("[Output] Metrics summary     → metrics.json")
    print("[Done]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
