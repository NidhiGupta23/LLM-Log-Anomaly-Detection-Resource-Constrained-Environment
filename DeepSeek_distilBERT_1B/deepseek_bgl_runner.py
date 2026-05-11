import argparse
import json
import os
import random
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, classification_report
from transformers import AutoTokenizer, AutoModelForCausalLM

RANDOM_SEED = 42
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


@dataclass
class RunConfig:
    model_id: str
    mode: str
    batch_size: int
    max_input_tokens: int
    max_new_tokens: int
    temperature: float
    do_sample: bool
    n_eval: int
    seed: int
    prompt_version: str
    dataset: str = "BGL"


SYSTEM_INSTRUCTIONS = (
    "You are classifying one BGL log entry for anomaly detection.\n"
    "Return exactly one label: 0 or 1.\n"
    "0 means normal. 1 means anomalous.\n"
    "Do not explain your answer.\n"
    "Do not output any other text.\n"
    "Judge based mainly on the event content, not severity alone."
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_bgl_log(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(f):
            line = line.rstrip("\n")
            if not line.strip():
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
            log_text = f"TYPE={log_type} COMPONENT={component} LEVEL={level} CONTENT={content}"
            records.append(
                {
                    "line_id": line_id,
                    "log_text": log_text,
                    "label": label,
                    "raw_line": line,
                    "alert_tag": label_raw,
                }
            )
    if not records:
        raise ValueError("No valid BGL records were parsed. Check file path and format.")
    return pd.DataFrame(records)


def stratified_sample(df: pd.DataFrame, n: Optional[int], seed: int) -> pd.DataFrame:
    if n is None or n <= 0 or n >= len(df):
        return df.sample(frac=1, random_state=seed).reset_index(drop=True)
    parts = []
    for label, group in df.groupby("label"):
        frac = len(group) / len(df)
        k = max(1, round(n * frac))
        k = min(k, len(group))
        parts.append(group.sample(n=k, random_state=seed))
    out = pd.concat(parts, axis=0)
    if len(out) > n:
        out = out.sample(n=n, random_state=seed)
    elif len(out) < n:
        remaining = df.loc[~df.index.isin(out.index)]
        add_n = min(n - len(out), len(remaining))
        if add_n > 0:
            out = pd.concat([out, remaining.sample(n=add_n, random_state=seed)], axis=0)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_few_shot_examples(path: Optional[str], seed: int) -> List[Dict]:
    if path is None:
        return [
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
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("few-shot JSON must contain a list of examples")
        return data
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        required = {"log", "label"}
        if not required.issubset(df.columns):
            raise ValueError("few-shot CSV must contain columns: log,label")
        return df[["log", "label"]].to_dict(orient="records")
    raise ValueError("few-shot examples file must be .json or .csv")


def build_prompt(log_text: str, mode: str, few_shot_examples: List[Dict]) -> str:
    user_block = []
    user_block.append(SYSTEM_INSTRUCTIONS)
    if mode == "few_shot":
        user_block.append("\nExamples:")
        for ex in few_shot_examples:
            user_block.append(f"Log entry: {ex['log']}\nLabel: {int(ex['label'])}")
    user_block.append(f"\nLog entry: {log_text}\nLabel:")
    return "\n".join(user_block)


def load_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def parse_prediction(text: str) -> Tuple[int, int]:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\b([01])\b", cleaned)
    if m:
        return int(m.group(1)), 0
    m = re.search(r"[01]", cleaned)
    if m:
        return int(m.group(0)), 1
    return 0, 1


def score_from_prediction(pred: int) -> float:
    return float(pred)


def classify(prompts: List[str], tokenizer, model, batch_size: int, max_input_tokens: int, max_new_tokens: int, temperature: float, do_sample: bool):
    preds, raw_texts, parse_fallbacks, per_sample_ms = [], [], [], []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        t0 = time.perf_counter()
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        device = model.device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        elapsed = (time.perf_counter() - t0) * 1000 / len(batch)
        for output, input_len in zip(outputs, input_lengths):
            gen_tokens = output[int(input_len):]
            gen_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            pred, used_fallback = parse_prediction(gen_text)
            preds.append(pred)
            raw_texts.append(gen_text)
            parse_fallbacks.append(used_fallback)
            per_sample_ms.append(elapsed)
    return preds, raw_texts, parse_fallbacks, per_sample_ms


def compute_metrics(labels: List[int], preds: List[int], parse_fallbacks: List[int], times_ms: List[float]) -> Dict:
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    metrics = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "total_samples": len(labels),
        "anomaly_count": int(sum(labels)),
        "predicted_anomaly_count": int(sum(preds)),
        "parse_fallback_count": int(sum(parse_fallbacks)),
        "avg_inference_time_ms": round(float(sum(times_ms) / len(times_ms)), 2),
        "total_inference_time_s": round(float(sum(times_ms) / 1000), 2),
        "classification_report": classification_report(labels, preds, labels=[0, 1], target_names=["Normal", "Anomalous"], zero_division=0),
    }
    try:
        metrics["roc_auc_from_hard_labels"] = round(float(roc_auc_score(labels, [score_from_prediction(p) for p in preds])), 4)
    except Exception:
        metrics["roc_auc_from_hard_labels"] = None
    return metrics


def save_outputs(df: pd.DataFrame, config: RunConfig, metrics: Dict, out_dir: str):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"bgl_{config.mode}_{config.n_eval if config.n_eval else 'all'}"
    csv_path = out / f"results_{stem}.csv"
    json_path = out / f"metrics_{stem}.json"
    cfg_path = out / f"config_{stem}.json"
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        payload = dict(metrics)
        payload["note"] = "roc_auc_from_hard_labels is weak and kept only for compatibility; prefer probability or logit-based AUC when available."
        json.dump(payload, f, indent=2)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    return csv_path, json_path, cfg_path


def main():
    parser = argparse.ArgumentParser(description="DeepSeek BGL zero-shot/few-shot benchmark runner")
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--mode", choices=["zero_shot", "few_shot", "both"], default="both")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_input_tokens", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--few_shot_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    df = parse_bgl_log(args.log_file)
    df = stratified_sample(df, args.n, args.seed)
    tokenizer, model = load_model(MODEL_ID)
    few_shot_examples = load_few_shot_examples(args.few_shot_file, args.seed)
    modes = [args.mode] if args.mode != "both" else ["zero_shot", "few_shot"]

    for mode in modes:
        prompts = [build_prompt(x, mode, few_shot_examples) for x in df["log_text"].tolist()]
        preds, raw_generations, parse_fallbacks, times_ms = classify(
            prompts,
            tokenizer,
            model,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        result_df = df.copy()
        result_df["mode"] = mode
        result_df["prediction"] = preds
        result_df["correct"] = (result_df["label"] == result_df["prediction"]).astype(int)
        result_df["raw_generation"] = raw_generations
        result_df["parse_fallback"] = parse_fallbacks
        result_df["inference_time_ms"] = times_ms
        metrics = compute_metrics(result_df["label"].tolist(), preds, parse_fallbacks, times_ms)
        config = RunConfig(
            model_id=MODEL_ID,
            mode=mode,
            batch_size=args.batch_size,
            max_input_tokens=args.max_input_tokens,
            max_new_tokens=args.max_new_tokens,
            temperature=0.0,
            do_sample=False,
            n_eval=len(result_df),
            seed=args.seed,
            prompt_version="v2_single_label_user_only",
        )
        csv_path, json_path, cfg_path = save_outputs(result_df, config, metrics, args.output_dir)
        print(f"[{mode}] accuracy={metrics['accuracy']} precision={metrics['precision']} recall={metrics['recall']} f1={metrics['f1']}")
        print(f"saved: {csv_path}")
        print(f"saved: {json_path}")
        print(f"saved: {cfg_path}")


if __name__ == "__main__":
    main()

