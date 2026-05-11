"""
DeepSeek-R1-Distill-Qwen-1.5B inference for BGL anomaly detection.

Use for zero-shot / few-shot evaluation on a fixed BGL test set.

Recommended final workflow:
  1. Create a frozen test CSV once.
  2. Run this script on the same CSV for every VM/model condition.
  3. Use the same scoring script for all methods.

CSV input format:
  log_text,label

Raw BGL input is also supported:
  label is column 0; '-' = normal, anything else = anomaly.
"""

import argparse
import json
import os
import platform
import random
import re
import time
from datetime import datetime, timezone
from typing import List, Tuple

import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


RANDOM_SEED = 42
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


SYSTEM_HEADER = """\
You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.

You will receive one structured log entry with fields such as TYPE, COMPONENT,
LEVEL, and CONTENT.

Classify the entry as exactly one digit:
0 = NORMAL
1 = ABNORMAL

Rules:
- Return only 0 or 1.
- Do not explain.
- Do not classify based on LEVEL alone.
- FATAL does not automatically mean abnormal.
- Focus mainly on CONTENT and event meaning.
"""


FEW_SHOT_EXAMPLES = [
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts internal error 0000000000001c00",
        "label": 0,
    },
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=INFO CONTENT=ddr error(s) detected and corrected on rank 0",
        "label": 0,
    },
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=INFO CONTENT=NFS Mount failed retrying",
        "label": 0,
    },
    {
        "log": "TYPE=RAS COMPONENT=KERNEL LEVEL=FATAL CONTENT=rts: kernel terminated",
        "label": 1,
    },
    {
        "log": "TYPE=RAS COMPONENT=APP LEVEL=FATAL CONTENT=ciod: error reading message header",
        "label": 1,
    },
]


def parse_raw_bgl(filepath: str) -> pd.DataFrame:
    records = []

    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 10:
                continue

            label_raw = parts[0]
            label = 0 if label_raw == "-" else 1

            log_type = parts[6]
            component = parts[7]
            level = parts[8]
            content = " ".join(parts[9:])

            log_text = (
                f"TYPE={log_type} "
                f"COMPONENT={component} "
                f"LEVEL={level} "
                f"CONTENT={content}"
            )

            records.append(
                {
                    "id": line_id,
                    "log_text": log_text,
                    "label": label,
                    "raw_line": line,
                }
            )

    return pd.DataFrame(records)


def load_input(args) -> pd.DataFrame:
    if args.input_csv:
        df = pd.read_csv(args.input_csv)
        required = {"log_text", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

        if "id" not in df.columns:
            df.insert(0, "id", range(len(df)))

        df = df[["id", "log_text", "label"]].copy()
    else:
        df = parse_raw_bgl(args.log_file)

    df["label"] = df["label"].astype(int)

    if args.n and args.n < len(df):
        df = stratified_sample(df, args.n, args.seed)

    return df.reset_index(drop=True)


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    normal = df[df["label"] == 0]
    anomalous = df[df["label"] == 1]

    anomaly_ratio = len(anomalous) / len(df)
    n_anomalous = max(1, int(n * anomaly_ratio))
    n_normal = n - n_anomalous

    sampled = pd.concat(
        [
            normal.sample(n=min(n_normal, len(normal)), random_state=seed),
            anomalous.sample(n=min(n_anomalous, len(anomalous)), random_state=seed),
        ]
    )

    return sampled.sample(frac=1, random_state=seed)


def build_zero_shot_prompt(log_text: str) -> str:
    return f"{SYSTEM_HEADER}\nLog entry:\n{log_text}\n\nAnswer:"


def build_few_shot_prompt(log_text: str) -> str:
    examples = []
    for ex in FEW_SHOT_EXAMPLES:
        examples.append(f"Log entry:\n{ex['log']}\n\nAnswer: {ex['label']}")

    examples_block = "\n\n".join(examples)

    return (
        f"{SYSTEM_HEADER}\n"
        f"Examples:\n\n"
        f"{examples_block}\n\n"
        f"Now classify this log entry.\n\n"
        f"Log entry:\n{log_text}\n\n"
        f"Answer:"
    )


def load_model(model_id: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    return tokenizer, model, device


def parse_prediction(text: str) -> int:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = cleaned.strip()

    match = re.search(r"\b([01])\b", cleaned)
    if match:
        return int(match.group(1))

    return -1


def classify_batch(
    prompts: List[str],
    tokenizer,
    model,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
) -> Tuple[List[int], List[str], List[float]]:
    predictions = []
    raw_outputs = []
    times = []

    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]

        t0 = time.perf_counter()

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        elapsed = time.perf_counter() - t0
        per_sample_time = elapsed / len(batch)

        for output in outputs:
            generated_tokens = output[prompt_len:]
            generated_text = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            raw_outputs.append(generated_text)
            predictions.append(parse_prediction(generated_text))
            times.append(per_sample_time)

        done = min(start + batch_size, len(prompts))
        print(f"Progress: {done}/{len(prompts)}", end="\r")

    print()
    return predictions, raw_outputs, times


def compute_metrics(labels: List[int], preds: List[int], times: List[float]) -> dict:
    valid_pairs = [(y, p) for y, p in zip(labels, preds) if p in (0, 1)]
    invalid_count = len(labels) - len(valid_pairs)

    if valid_pairs:
        valid_labels = [y for y, _ in valid_pairs]
        valid_preds = [p for _, p in valid_pairs]

        tn, fp, fn, tp = confusion_matrix(
            valid_labels,
            valid_preds,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(valid_labels, valid_preds, zero_division=0)
        recall = recall_score(valid_labels, valid_preds, zero_division=0)
        f1 = f1_score(valid_labels, valid_preds, zero_division=0)
        accuracy = accuracy_score(valid_labels, valid_preds)

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0

        report = classification_report(
            valid_labels,
            valid_preds,
            labels=[0, 1],
            target_names=["normal", "anomaly"],
            zero_division=0,
        )
    else:
        tn = fp = fn = tp = 0
        precision = recall = f1 = accuracy = fpr = fnr = 0.0
        report = "No valid predictions."

    return {
        "total_samples": len(labels),
        "valid_predictions": len(valid_pairs),
        "invalid_predictions": invalid_count,
        "invalid_rate": round(invalid_count / len(labels), 4) if labels else 0,
        "anomaly_count": int(sum(labels)),
        "predicted_anomaly_count": int(sum(1 for p in preds if p == 1)),
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "avg_inference_time_ms": round(sum(times) / len(times) * 1000, 2),
        "total_inference_time_s": round(sum(times), 2),
        "classification_report": report,
    }


def build_metadata(args, mode: str, device: str, sample_count: int) -> dict:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "model_id": args.model_id,
        "sample_count": sample_count,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "max_new_tokens": args.max_new_tokens,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "input_csv": args.input_csv,
        "log_file": args.log_file,
    }


def run_mode(args, mode: str, df: pd.DataFrame, tokenizer, model, device: str):
    if mode == "zero_shot":
        prompts = [build_zero_shot_prompt(x) for x in df["log_text"]]
    elif mode == "few_shot":
        prompts = [build_few_shot_prompt(x) for x in df["log_text"]]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    preds, raw_outputs, times = classify_batch(
        prompts=prompts,
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
    )

    labels = df["label"].tolist()
    metrics = compute_metrics(labels, preds, times)
    metadata = build_metadata(args, mode, device, len(df))

    result_df = df.copy()
    result_df["prediction"] = preds
    result_df["raw_model_output"] = raw_outputs
    result_df["inference_time_ms"] = [round(t * 1000, 2) for t in times]
    result_df["is_valid_prediction"] = result_df["prediction"].isin([0, 1])
    result_df["correct"] = result_df["label"] == result_df["prediction"]

    csv_path = os.path.join(args.output_dir, f"predictions_{mode}.csv")
    json_path = os.path.join(args.output_dir, f"metrics_{mode}.json")

    result_df.to_csv(csv_path, index=False)

    report = metrics.pop("classification_report")
    payload = {
        "metadata": metadata,
        "metrics": metrics,
        "classification_report": report,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    metrics["classification_report"] = report

    print(f"\nResults: {mode}")
    print(f"Accuracy : {metrics['accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall   : {metrics['recall']}")
    print(f"F1       : {metrics['f1']}")
    print(f"Invalid  : {metrics['invalid_predictions']} ({metrics['invalid_rate']})")
    print(f"Saved predictions: {csv_path}")
    print(f"Saved metrics    : {json_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"], default="both")
    parser.add_argument("--model_id", default=MODEL_ID)

    parser.add_argument("--input_csv", default=None, help="Frozen CSV with log_text,label columns")
    parser.add_argument("--log_file", default="BGL.log", help="Raw BGL log file if input_csv is not used")
    parser.add_argument("--n", type=int, default=None, help="Optional stratified sample size")

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)

    parser.add_argument("--output_dir", default="outputs_deepseek_bgl")

    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_input(args)

    print(f"Loaded samples : {len(df)}")
    print(f"Anomalies      : {int(df['label'].sum())}")
    print(f"Anomaly ratio  : {df['label'].mean():.4f}")

    tokenizer, model, device = load_model(args.model_id)

    modes = ["zero_shot", "few_shot"] if args.mode == "both" else [args.mode]

    for mode in modes:
        run_mode(args, mode, df, tokenizer, model, device)

    print("\nDone.")


if __name__ == "__main__":
    main()

