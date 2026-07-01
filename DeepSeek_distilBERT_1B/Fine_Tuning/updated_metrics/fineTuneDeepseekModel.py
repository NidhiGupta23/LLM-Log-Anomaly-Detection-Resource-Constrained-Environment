"""
BGL Anomaly Classifier — QLoRA Fine-tuning Script
Target model : deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
GPU VM       : 90 GB RAM, 8 vCPU  (>=8 GB VRAM recommended)
Inference    : Optimised for 8 GB and 16 GB CPU-only VMs via GGUF export

Pipeline
--------
1. Load BGL dataset file (all 10 fields, space-separated)
2. Derive binary labels from the `label` column ("-" -> 0, anything else -> 1)
3. Stratified split into train / val / test (seed=42) and save split files
4. Fine-tune DeepSeek-R1-Distill-Qwen-1.5B with LoRA in fp16
   (V100 = compute capability 7.0: 4-bit NF4 requires cc 7.5+, so no bitsandbytes)
5. Save LoRA adapter -> merge into full model -> ready for GGUF export

BGL dataset column order (space-separated, 10 fields)
------------------------------------------------------
  label  timestamp  date  node  time  node_repeat  type  component  level  content...
  [0]    [1]        [2]   [3]   [4]   [5]          [6]   [7]        [8]    [9]

  "-" in the label column = non-alert (NORMAL -> 0)
  anything else            = alert     (ABNORMAL -> 1)

Prompt optimisation (faster CPU inference)
------------------------------------------
  Dropped columns with no/low anomaly signal:
    node      - hardware address, zero semantic content
    level     - rules explicitly say "don't judge on severity label alone"
    type      - always "RAS" in BGL, zero variance
    timestamp/date/time/node_repeat - already excluded previously

  Kept columns:
    component - KERNEL vs APP helps distinguish fatal vs app errors
    content   - the primary anomaly signal

  Prompt shrinks from:
    "[KERNEL][FATAL] Node:R25-M0-NA-I:J00-U01 | kernel terminated"
  to:
    "[KERNEL] kernel terminated"

Install dependencies (GPU VM)
------------------------------
  pip install torch transformers accelerate peft trl datasets
  # bitsandbytes NOT needed for V100 (fp16 LoRA, no 4-bit quantisation)

GGUF export (after this script finishes)
-----------------------------------------
  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
  cmake -B build && cmake --build build --config Release -j$(nproc)

  # Convert to F16 GGUF
  python convert_hf_to_gguf.py ../bgl_merged --outtype f16 --outfile bgl_1.5b.f16.gguf

  # 8 GB VM  -> Q4_K_M (~1.1 GB RAM)
  ./build/bin/llama-quantize bgl_1.5b.f16.gguf bgl_1.5b_Q4_K_M.gguf Q4_K_M

  # 16 GB VM -> Q8_0   (~1.9 GB RAM, higher accuracy)
  ./build/bin/llama-quantize bgl_1.5b.f16.gguf bgl_1.5b_Q8_0.gguf Q8_0

Usage examples
--------------
  # Minimal
  python finetune_deepseek.py --bgl_file BGL.log

  # Full control
  python finetune_deepseek.py \
      --bgl_file      BGL.log \
      --model_name    deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
      --output_dir    ./bgl_adapter \
      --merged_dir    ./bgl_merged \
      --splits_dir    ./bgl_splits \
      --epochs        3 \
      --lr            2e-4 \
      --batch_size    8 \
      --grad_accum    4 \
      --max_seq_len   384 \
      --lora_r        16 \
      --lora_alpha    32 \
      --val_split     0.1 \
      --test_split    0.1 \
      --seed          42
"""

import argparse
import json
import os
import sys
import random

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)

# ---------------------------------------------------------------------------
# trl import
# ---------------------------------------------------------------------------
try:
    from trl import SFTTrainer
except ImportError as e:
    print(f"[ERROR] Could not import SFTTrainer from trl: {e}", file=sys.stderr)
    print("        Run: pip install -U trl", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# DataCollatorForCompletionOnlyLM — self-contained implementation.
#
# This class has moved between trl, trl.extras, and transformers across
# recent releases and is now absent from all of them in trl >= 0.17 /
# transformers >= 5.9.  Rather than chasing the moving target we implement
# the essential behaviour directly:
#
#   • Tokenise each "text" sample.
#   • Find the response_template token sequence inside the input_ids.
#   • Set labels = -100 for every token UP TO AND INCLUDING the template,
#     so the loss is computed only on the completion tokens (the "0"/"1").
# ---------------------------------------------------------------------------
from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class DataCollatorForCompletionOnlyLM:
    """
    Mask prompt tokens from the language-modelling loss.
    Only the tokens that follow `response_template` are trained.

    Args:
        response_template: The string (e.g. "\\nLabel:") that marks the
                           boundary between prompt and completion.
        tokenizer:         The tokeniser used to encode the template.
        mlm:               Always False — we are doing causal LM, not MLM.
    """
    response_template: str
    tokenizer: Any
    mlm: bool = False

    def __post_init__(self):
        # Pre-tokenise the template once; strip BOS if the tokeniser adds one.
        self._template_ids = self.tokenizer.encode(
            self.response_template, add_special_tokens=False
        )
        if not self._template_ids:
            raise ValueError(
                f"response_template {self.response_template!r} tokenised to an "
                "empty sequence — choose a different template string."
            )

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        # Each feature must already have a "text" key (added by SFTTrainer
        # via dataset_text_field) OR pre-tokenised input_ids/labels.
        texts = [f["text"] if "text" in f else
                 self.tokenizer.decode(f["input_ids"]) for f in features]

        batch = self.tokenizer(
            texts,
            padding        = True,
            truncation     = True,
            return_tensors = "pt",
        )

        labels = batch["input_ids"].clone()
        tpl    = self._template_ids
        tpl_len = len(tpl)

        for i, seq in enumerate(labels):
            seq_list = seq.tolist()
            # Find the last occurrence of the template in the token sequence.
            found = -1
            for j in range(len(seq_list) - tpl_len, -1, -1):
                if seq_list[j : j + tpl_len] == tpl:
                    found = j
                    break
            if found == -1:
                # Template not found: mask everything (don't train on this sample).
                labels[i] = -100
            else:
                # Mask everything up to and including the template itself.
                labels[i, : found + tpl_len] = -100

        # Also mask padding tokens.
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# BGL log parsing
# ---------------------------------------------------------------------------
#
# Full BGL line format (space-separated, 10 fields):
#   label  timestamp  date  node  time  node_repeat  type  component  level  content...
#   [0]    [1]        [2]   [3]   [4]   [5]          [6]   [7]        [8]    [9]
#
_BGL_FIELD_COUNT = 10


def parse_bgl_line(raw_line: str) -> dict:
    """
    Parse a raw BGL log line that includes the leading label column.
    Only extracts the fields actually used in the prompt (component, content).
    Returns {"raw": ..., "label_tag": "-"} on parse failure.
    """
    parts = raw_line.strip().split(None, _BGL_FIELD_COUNT - 1)
    if len(parts) < _BGL_FIELD_COUNT:
        # Fallback: try without the leading label (9-field format)
        parts9 = raw_line.strip().split(None, _BGL_FIELD_COUNT - 2)
        if len(parts9) >= _BGL_FIELD_COUNT - 1:
            return {
                "label_tag" : "-",
                "component" : parts9[6],
                "content"   : parts9[8] if len(parts9) > 8 else "",
            }
        return {"raw": raw_line.strip(), "label_tag": "-"}

    return {
        "label_tag" : parts[0],   # "-" = normal, else = alert type
        "component" : parts[7],
        "content"   : parts[9],
    }


def derive_binary_label(parsed: dict) -> str:
    """"-" -> "0" (NORMAL),  anything else -> "1" (ABNORMAL)."""
    return "0" if parsed.get("label_tag", "-") == "-" else "1"


def format_log_for_prompt(parsed: dict) -> str:
    """
    Compact prompt representation - only component + content.

    Dropped (no/low anomaly signal):
      node      - hardware address
      level     - rules say "don't judge on severity label alone"
      type      - always "RAS", zero variance
      timestamp/date/time/node_repeat - temporal/duplicate metadata

    Before: "[KERNEL][FATAL] Node:R25-M0-NA-I:J00-U01 | kernel terminated"
    After:  "[KERNEL] kernel terminated"
    Saves ~10-15 tokens per sample -> faster CPU inference.
    """
    if "raw" in parsed:
        return parsed["raw"]
    return f"[{parsed['component']}] {parsed['content']}"


# ---------------------------------------------------------------------------
# Prompt template
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

# DataCollatorForCompletionOnlyLM uses this to find where the label begins
# so only the "0"/"1" token contributes to the training loss.
_RESPONSE_TEMPLATE = "\nLabel:"


def build_training_text(raw_line: str, label: str) -> str:
    """Build a full training string: prompt + label (the only trained token)."""
    log_input = format_log_for_prompt(parse_bgl_line(raw_line))
    prompt    = f"{_RULES}\n\nLog: {log_input}{_RESPONSE_TEMPLATE}"
    return f"{prompt} {label}"


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_bgl_file(path: str) -> list:
    """
    Load the full BGL dataset file.
    Returns a list of dicts: {raw_line, label, label_tag}.
    """
    if not os.path.exists(path):
        print(f"[ERROR] BGL file not found: {path}", file=sys.stderr)
        sys.exit(1)

    records = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw_line in fh:
            if not raw_line.strip():
                continue
            parsed = parse_bgl_line(raw_line)
            records.append({
                "raw_line"  : raw_line.strip(),
                "label"     : derive_binary_label(parsed),
                "label_tag" : parsed.get("label_tag", "-"),
            })

    n_normal   = sum(1 for r in records if r["label"] == "0")
    n_abnormal = sum(1 for r in records if r["label"] == "1")
    print(f"  Loaded {len(records):>7} lines  <- {path}")
    print(f"  Label distribution : {n_normal} normal (0)  /  {n_abnormal} abnormal (1)")
    if n_normal == 0 or n_abnormal == 0:
        print("[WARN] One class is entirely empty - verify your BGL file format.")
    return records


def split_dataset(records, val_split, test_split, seed):
    """
    Stratified three-way split: train / val / test.
    Splits each class independently to preserve class ratios.
    """
    random.seed(seed)

    normal   = [r for r in records if r["label"] == "0"]
    abnormal = [r for r in records if r["label"] == "1"]

    def _split_class(items):
        items = items[:]
        random.shuffle(items)
        n      = len(items)
        n_test = max(1, int(n * test_split))
        n_val  = max(1, int(n * val_split))
        return (
            items[n_test + n_val:],        # train
            items[n_test:n_test + n_val],  # val
            items[:n_test],                # test
        )

    tr0, va0, te0 = _split_class(normal)
    tr1, va1, te1 = _split_class(abnormal)

    train = tr0 + tr1;  random.shuffle(train)
    val   = va0 + va1;  random.shuffle(val)
    test  = te0 + te1;  random.shuffle(test)

    return train, val, test


def save_splits(train, val, test, out_dir):
    """Write each split as a JSONL file and a plain .log file."""
    os.makedirs(out_dir, exist_ok=True)
    for name, records in [("train", train), ("val", val), ("test", test)]:
        jsonl_path = os.path.join(out_dir, f"{name}.jsonl")
        log_path   = os.path.join(out_dir, f"{name}.log")
        with open(jsonl_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        with open(log_path, "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(r["raw_line"] + "\n")
        n0 = sum(1 for r in records if r["label"] == "0")
        n1 = sum(1 for r in records if r["label"] == "1")
        print(f"  {name:<5} -> {len(records):>6} samples  "
              f"(normal:{n0}  abnormal:{n1})  -> {jsonl_path}")


def build_hf_datasets(train, val):
    def to_text(r):
        return {"text": build_training_text(r["raw_line"], r["label"])}
    return (
        Dataset.from_list([to_text(r) for r in train]),
        Dataset.from_list([to_text(r) for r in val]),
    )


# ---------------------------------------------------------------------------
# LoRA model setup (fp16, no bitsandbytes — V100 compatible)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, lora_r, lora_alpha):
    print(f"\nLoading base model: {model_name}")
    print(  "  Backend: fp16 — V100 is compute capability 7.0 and cannot run")
    print(  "  4-bit NF4 (requires cc 7.5+). At 1.5B params fp16 uses ~3.5 GB")
    print(  "  VRAM, well within the 16 GB V100 budget. No bitsandbytes needed.")

    # Load straight to fp16 on GPU — no quantisation required at 1.5B scale.
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype       = torch.float16,
        device_map        = "auto",
        trust_remote_code = True,
    )

    # Gradient checkpointing: halves activation memory at ~20% extra compute.
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()   # Required for PEFT + grad checkpointing

    # DeepSeek-R1-Distill-Qwen-1.5B uses the Qwen2 architecture;
    # same LoRA target modules apply.
    lora_config = LoraConfig(
        r             = lora_r,
        lora_alpha    = lora_alpha,
        lora_dropout  = 0.05,
        bias          = "none",
        task_type     = "CAUSAL_LM",
        target_modules= [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code = True,
        padding_side      = "right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


# ---------------------------------------------------------------------------
# Merge & save full model (needed for GGUF export)
# ---------------------------------------------------------------------------

def merge_and_save(adapter_dir, merged_dir, model_name):
    print("\nMerging LoRA adapter into base model ...")
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype       = torch.float16,
        device_map        = "cpu",
        trust_remote_code = True,
    )
    merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=True)

    os.makedirs(merged_dir, exist_ok=True)
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"  Merged model saved -> {merged_dir}")
    print()
    print("  -- NEXT STEPS: GGUF Export ------------------------------------------")
    print("  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp")
    print("  cmake -B build && cmake --build build --config Release -j$(nproc)")
    print(f"  python convert_hf_to_gguf.py ../{merged_dir} --outtype f16 \\")
    print( "         --outfile bgl_deepseek1_5b.f16.gguf")
    print("  # 8 GB VM  -> Q4_K_M (~1.1 GB RAM):")
    print("  ./build/bin/llama-quantize bgl_deepseek1_5b.f16.gguf bgl_deepseek1_5b_Q4_K_M.gguf Q4_K_M")
    print("  # 16 GB VM -> Q8_0   (~1.9 GB RAM, higher accuracy):")
    print("  ./build/bin/llama-quantize bgl_deepseek1_5b.f16.gguf bgl_deepseek1_5b_Q8_0.gguf Q8_0")
    print("  ---------------------------------------------------------------------")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    print("=" * 60)
    print("BGL ANOMALY CLASSIFIER -- LoRA FINE-TUNING (fp16, V100-compatible)")
    print(f"  Base model  : {args.model_name}")
    print("=" * 60)
    print(f"  BGL file    : {args.bgl_file}")
    print(f"  Splits dir  : {args.splits_dir}")
    print(f"  Output dir  : {args.output_dir}")
    print(f"  Merged dir  : {args.merged_dir}")
    print(f"  Seed        : {args.seed}")
    print(f"  Val split   : {args.val_split}")
    print(f"  Test split  : {args.test_split}")
    print(f"  Epochs      : {args.epochs}")
    print(f"  Batch size  : {args.batch_size} x {args.grad_accum} (grad accum)")
    print(f"  LoRA r/a    : {args.lora_r} / {args.lora_alpha}")
    print(f"  Max seq len : {args.max_seq_len}")
    print()

    print("-- Step 1: Loading BGL dataset --------------------------------------")
    records = load_bgl_file(args.bgl_file)

    print("\n-- Step 2: Stratified split (seed=42) --------------------------------")
    train_r, val_r, test_r = split_dataset(
        records,
        val_split  = args.val_split,
        test_split = args.test_split,
        seed       = args.seed,
    )
    save_splits(train_r, val_r, test_r, args.splits_dir)

    train_ds, val_ds = build_hf_datasets(train_r, val_r)
    print(f"\n  HF train : {len(train_ds)} samples")
    print(f"  HF val   : {len(val_ds)} samples")
    print(f"  Test set : {len(test_r)} samples (saved to {args.splits_dir}/test.log)")

    print("\n-- Step 3: Loading model & building LoRA ----------------------------")
    model, tokenizer = load_model_and_tokenizer(
        args.model_name, args.lora_r, args.lora_alpha
    )

    collator = DataCollatorForCompletionOnlyLM(
        response_template = _RESPONSE_TEMPLATE,
        tokenizer         = tokenizer,
    )

    training_args = TrainingArguments(
        output_dir                  = args.output_dir,
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        per_device_eval_batch_size  = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        learning_rate               = args.lr,
        lr_scheduler_type           = "cosine",
        warmup_ratio                = 0.05,
        # V100 does not support bfloat16 — always use fp16.
        bf16                        = False,
        fp16                        = True,
        logging_steps               = 20,
        eval_strategy               = "epoch",
        save_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "eval_loss",
        save_total_limit            = 2,
        report_to                   = "none",
        dataloader_num_workers      = min(4, args.num_workers),
        # paged_adamw_8bit requires bitsandbytes — use adamw_torch instead.
        # Memory impact is negligible at 1.5B scale with fp16.
        optim                       = "adamw_torch",
        seed                        = args.seed,
    )

    trainer = SFTTrainer(
        model              = model,
        processing_class   = tokenizer,
        train_dataset      = train_ds,
        eval_dataset       = val_ds,
        args               = training_args,
        data_collator      = collator,
    )


    print("\n-- Step 4: Training -------------------------------------------------")
    trainer.train()

    print(f"\nSaving LoRA adapter -> {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    config_path = os.path.join(args.output_dir, "training_config.json")
    with open(config_path, "w") as fh:
        json.dump(vars(args), fh, indent=2)
    print(f"Training config   -> {config_path}")

    print("\n-- Step 5: Merging adapter into full model --------------------------")
    merge_and_save(args.output_dir, args.merged_dir, args.model_name)

    print("\nFine-tuning complete.")
    print(f"  LoRA adapter : {args.output_dir}")
    print(f"  Merged model : {args.merged_dir}  (ready for GGUF quantisation)")
    print(f"  Test split   : {args.splits_dir}/test.log")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser():
    p = argparse.ArgumentParser(
        description=(
            "LoRA fine-tuning for BGL anomaly classification -- "
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = p.add_argument_group("Data")
    g.add_argument(
        "--bgl_file", required=True,
        help=(
            "Full BGL dataset file. Each line must start with the label column "
            "('-' = normal, any other tag = abnormal) followed by 9 more "
            "space-separated fields."
        ),
    )
    g.add_argument("--splits_dir",  default="./bgl_splits")
    g.add_argument("--val_split",   type=float, default=0.1)
    g.add_argument("--test_split",  type=float, default=0.1)
    g.add_argument("--seed",        type=int,   default=42)

    g = p.add_argument_group("Model")
    g.add_argument("--model_name",  default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    g.add_argument("--output_dir",  default="./bgl_adapter")
    g.add_argument("--merged_dir",  default="./bgl_merged")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs",      type=int,   default=3)
    g.add_argument("--lr",          type=float, default=2e-4)
    g.add_argument("--batch_size",  type=int,   default=8,
                   help="1.5B model fits batch=8 in ~8 GB VRAM.")
    g.add_argument("--grad_accum",  type=int,   default=4,
                   help="Effective batch = batch_size x grad_accum.")
    g.add_argument("--max_seq_len", type=int,   default=384,
                   help="384 fits all BGL prompts; saves VRAM vs 512.")
    g.add_argument("--num_workers", type=int,   default=4)

    g = p.add_argument_group("LoRA")
    g.add_argument("--lora_r",      type=int,   default=16)
    g.add_argument("--lora_alpha",  type=int,   default=32,
                   help="Convention: 2 x lora_r.")

    return p


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[WARN] No GPU detected. Fine-tuning on CPU will be extremely slow.",
              file=sys.stderr)
    train(_build_parser().parse_args())
