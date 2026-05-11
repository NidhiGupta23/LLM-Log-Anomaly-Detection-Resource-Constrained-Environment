"""
DeepSeek-R1-Distill-Qwen-1.5B — Zero-Shot and Few-Shot Inference on BGL
======================================================================
Usage:
  python deepseek_bgl_inference.py --mode zero_shot --log_file BGL.log --n 2000
  python deepseek_bgl_inference.py --mode few_shot  --log_file BGL.log --n 2000

Outputs:
  results_zero_shot.csv  — predictions + ground truth + timing per sample
  results_few_shot.csv
  metrics_zero_shot.json — F1, Precision, Recall, AUC-ROC, timing summary
  metrics_few_shot.json
"""

import argparse
import json
import re
import time
import random
import os
from dataclasses import dataclass, asdict
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
from transformers import AutoTokenizer, AutoModelForCausalLM

# ── Reproducibility ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# ── Few-shot examples (from training split only — never from test set) ───────
# These are representative BGL examples covering normal and anomalous cases.
# Derived from BGL documentation and training data — not from test set.
FEW_SHOT_EXAMPLES = [
    # Normal — register dump (FATAL level but normal diagnostic)
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts internal error 0000000000001c00",
        "label": 0,
        "explanation": "Register dump line — part of normal error recovery diagnostic."
    },
    # Normal — self-corrected hardware error
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=INFO CONTENT=ddr error(s) detected and corrected on rank 0",
        "label": 0,
        "explanation": "Self-corrected memory error — hardware recovered automatically."
    },
    # Normal — transient recoverable failure
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=INFO CONTENT=NFS Mount failed retrying",
        "label": 0,
        "explanation": "Transient failure with automatic retry — not an anomaly."
    },
    # Anomalous — kernel termination
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts: kernel terminated",
        "label": 1,
        "explanation": "Kernel terminated — definitive process failure."
    },
    # Anomalous — ciod socket failure
    {
        "log": "TYPE=RAS COMPONENT=APP LEVEL=FATAL CONTENT=ciod: error reading message header",
        "label": 1,
        "explanation": "ciod socket error — communication failure between compute nodes."
    },
]


# ── BGL Log Parser ───────────────────────────────────────────────────────────
def parse_bgl_log(filepath: str, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Parse raw BGL .log file into a DataFrame with columns:
      log_text : str  — TYPE, COMPONENT, LEVEL, CONTENT formatted
      label    : int  — 0 = normal, 1 = anomalous
      raw_line : str  — original line for reference

    BGL line format (space-separated):
      col 0  : label ('-' = normal, else = anomaly type)
      col 1  : integer timestamp
      col 2  : date
      col 3  : node
      col 4  : full timestamp
      col 5  : node (repeated)
      col 6  : TYPE  (e.g. RAS)
      col 7  : COMPONENT (e.g. KERNEL, APP)
      col 8  : LEVEL (e.g. FATAL, INFO, WARNING)
      col 9+ : CONTENT
    """
    records = []
    print(f"Parsing {filepath} ...")

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 10:
                continue  # skip malformed lines

            label_raw = parts[0]
            label = 0 if label_raw == "-" else 1
            log_type = parts[6] if len(parts) > 6 else "UNKNOWN"
            component = parts[7] if len(parts) > 7 else "UNKNOWN"
            level = parts[8] if len(parts) > 8 else "UNKNOWN"
            content = " ".join(parts[9:]) if len(parts) > 9 else ""

            log_text = (
                f"TYPE={log_type} "
                f"COMPONENT={component} "
                f"LEVEL={level} "
                f"CONTENT={content}"
            )
            records.append({
                "log_text": log_text,
                "label": label,
                "raw_line": line,
            })

    df = pd.DataFrame(records)
    print(f"  Total parsed lines : {len(df)}")
    print(f"  Anomaly count      : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")

    # Sample n lines with fixed seed — stratified to preserve anomaly ratio
    if n and n < len(df):
        normal = df[df["label"] == 0].sample(frac=1, random_state=seed)
        anomal = df[df["label"] == 1].sample(frac=1, random_state=seed)
        # Keep same anomaly ratio in sample
        anomaly_ratio = len(anomal) / len(df)
        n_anomal = max(1, int(n * anomaly_ratio))
        n_normal = n - n_anomal
        df = pd.concat([
            normal.head(n_normal),
            anomal.head(n_anomal)
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)
        print(f"  Sampled {n} lines  : {df['label'].sum()} anomalous ({df['label'].mean()*100:.1f}%)")

    return df


# ── Prompt Builders ──────────────────────────────────────────────────────────
_SYSTEM_HEADER = """\
You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.

You will receive a structured log entry with these fields:
  TYPE      – high-level event type (e.g. RAS)
  COMPONENT – subsystem (e.g. KERNEL, APP)
  LEVEL     – severity (e.g. FATAL, INFO, WARNING)
  CONTENT   – free-text description of the event

Classify the entry as EXACTLY one digit:
  0 = NORMAL
  1 = ABNORMAL

CRITICAL rules:
- Do NOT classify based on LEVEL alone.
- FATAL does NOT automatically mean abnormal.
- Focus on CONTENT.

ABNORMAL indicators:
- Kernel/process termination (rts: kernel terminated, rts panic, stopping execution)
- ciod failures (ciod: error reading, ciod: socket error, ciod: failed to connect)
- Network errors (error receiving packet, link has been severed, connection timed out)
- Storage failures (Lustre mount FAILED, I/O error, data storage interrupt)
- Illegal instruction / machine check interrupt

NORMAL indicators:
- Register dumps (rts internal error, instruction address, machine state register)
- Self-corrected errors (detected and corrected, cache parity error corrected, CE sym)
- Transient retries (NFS mount failed retrying, suppressing further interrupts)

Return EXACTLY one digit: 0 or 1. No explanation. No other text.
"""


def build_zero_shot_prompt(log_text: str) -> str:
    return (
        f"{_SYSTEM_HEADER}\n"
        f"Log entry: {log_text}\n"
        f"OUTPUT:"
    )


def build_few_shot_prompt(log_text: str) -> str:
    examples_block = ""
    for ex in FEW_SHOT_EXAMPLES:
        examples_block += f"Log entry: {ex['log']}\nOUTPUT: {ex['label']}\n\n"

    return (
        f"{_SYSTEM_HEADER}\n"
        f"EXAMPLES:\n\n"
        f"{examples_block}"
        f"Now classify this entry:\n"
        f"Log entry: {log_text}\n"
        f"OUTPUT:"
    )


# ── Model Loading ─────────────────────────────────────────────────────────────
def load_model(model_id: str):
    print(f"\nLoading model: {model_id}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"  Model loaded on: {next(model.parameters()).device}")
    return tokenizer, model, device


# ── Inference ─────────────────────────────────────────────────────────────────
def parse_prediction(text: str) -> int:
    """Extract first 0 or 1 from generated text. Default to 0 if unclear."""
    # Strip thinking tags (DeepSeek-R1 style)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()
    match = re.search(r"\b([01])\b", text)
    if match:
        return int(match.group(1))
    # Fallback: look for any 0 or 1 character
    match = re.search(r"[01]", text)
    return int(match.group()) if match else 0


def classify_batch(
    prompts: List[str],
    tokenizer,
    model,
    max_length: int = 512,
    max_new_tokens: int = 16,
    batch_size: int = 8,
) -> Tuple[List[int], List[float]]:
    """Run inference in mini-batches. Returns (predictions, per-sample times)."""
    all_preds = []
    all_times = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start: start + batch_size]

        t0 = time.perf_counter()
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()
        batch_time = (t1 - t0) / len(batch)  # per-sample time

        for output, input_len in zip(outputs, input_lengths):
            generated_tokens = output[input_len:]
            generated_text = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )
            all_preds.append(parse_prediction(generated_text))
            all_times.append(batch_time)

        done = min(start + batch_size, len(prompts))
        print(f"  Progress: {done}/{len(prompts)} logs", end="\r")

    print()
    return all_preds, all_times


# ── Evaluation ────────────────────────────────────────────────────────────────
def compute_metrics(labels: List[int], preds: List[int], times: List[float]) -> dict:
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        auc = roc_auc_score(labels, preds)
    except ValueError:
        auc = None  # only one class in labels

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc_roc": round(auc, 4) if auc else None,
        "total_samples": len(labels),
        "anomaly_count": sum(labels),
        "predicted_anomaly_count": sum(preds),
        "avg_inference_time_ms": round(sum(times) / len(times) * 1000, 2),
        "total_inference_time_s": round(sum(times), 2),
        "classification_report": classification_report(
            labels, preds, target_names=["Normal", "Anomalous"]
        )
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"],
                        default="both", help="Inference mode")
    parser.add_argument("--log_file", type=str, default="BGL.log",
                        help="Path to raw BGL .log file")
    parser.add_argument("--n", type=int, default=2000,
                        help="Number of log lines to evaluate (default 2000 for dev; use 10000 for thesis)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for inference")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max input token length")
    parser.add_argument("--output_dir", type=str, default=".",
                        help="Directory to save results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Parse dataset
    df = parse_bgl_log(args.log_file, args.n)

    # 2. Load model (once, shared across modes)
    tokenizer, model, device = load_model(MODEL_ID)

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\n{'='*60}")
        print(f"  Running: {mode.upper()}")
        print(f"{'='*60}")

        # Build prompts
        if mode == "zero_shot":
            prompts = [build_zero_shot_prompt(log) for log in df["log_text"]]
        else:
            prompts = [build_few_shot_prompt(log) for log in df["log_text"]]

        # Run inference
        preds, times = classify_batch(
            prompts, tokenizer, model,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )

        # Compute metrics
        labels = df["label"].tolist()
        metrics = compute_metrics(labels, preds, times)

        # Print results
        print(f"\n  Results ({mode}):")
        print(f"    Precision : {metrics['precision']:.4f}")
        print(f"    Recall    : {metrics['recall']:.4f}")
        print(f"    F1        : {metrics['f1']:.4f}")
        print(f"    AUC-ROC   : {metrics['auc_roc']}")
        print(f"    Avg time  : {metrics['avg_inference_time_ms']:.1f} ms/sample")
        print(f"\n  Classification Report:\n{metrics['classification_report']}")

        # Save predictions CSV
        result_df = df[["log_text", "label"]].copy()
        result_df["prediction"] = preds
        result_df["inference_time_ms"] = [t * 1000 for t in times]
        result_df["correct"] = (result_df["label"] == result_df["prediction"]).astype(int)
        csv_path = os.path.join(args.output_dir, f"results_{mode}.csv")
        result_df.to_csv(csv_path, index=False)
        print(f"  Saved predictions: {csv_path}")

        # Save metrics JSON
        report_str = metrics.pop("classification_report")
        json_path = os.path.join(args.output_dir, f"metrics_{mode}.json")
        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=2)
        # Put it back for display
        metrics["classification_report"] = report_str
        print(f"  Saved metrics    : {json_path}")

    print("\nAll done.")


if __name__ == "__main__":
    main()

