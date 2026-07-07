import json
import re
import time
import psutil
import os
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Fixed few-shot examples — 5 abnormal + 5 normal
# These are real BGL log lines (label field excluded).
# Must be kept identical across all models, modes, and VMs for fair comparison.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
    # Abnormal (label = 1)
    {
        "log": (
            "1117869876 2005.06.04 R27-M1-N4-I:J18-U01 "
            "2005-06-04-00.24.36.222560 R27-M1-N4-I:J18-U01 RAS APP FATAL "
            "ciod: failed to read message prefix on control stream "
            "(CioStream socket to 172.16.96.116:33370"
        ),
        "label": "1",
    },
    {
        "log": (
            "1118557583 2005.06.11 R30-M0-N9-C:J16-U01 "
            "2005-06-11-23.26.23.330548 R30-M0-N9-C:J16-U01 RAS KERNEL FATAL "
            "data TLB error interrupt"
        ),
        "label": "1",
    },
    {
        "log": (
            "1118709403 2005.06.13 R10-M1-N5-C:J15-U11 "
            "2005-06-13-17.36.43.927885 R10-M1-N5-C:J15-U11 RAS KERNEL FATAL "
            "data storage interrupt"
        ),
        "label": "1",
    },
    {
        "log": (
            "1119319455 2005.06.20 R12-M1-NC-I:J18-U11 "
            "2005-06-20-19.04.15.002425 R12-M1-NC-I:J18-U11 RAS APP FATAL "
            "ciod: Error creating node map from file "
            "/p/gb2/cabot/miranda/newmaps/8k_128x64x1_8x4x4.map: No child processes"
        ),
        "label": "1",
    },
]

# Shared classification rules used in both prompts
_RULES = """Rules:
- 0 = NORMAL: informational messages, corrected hardware errors, cache parity
  corrected, DDR errors corrected, CE symbols, core file generation, alignment
  exceptions, register dumps, routine warnings, diagnostic messages,
  recovery/retry messages.
- 1 = ABNORMAL: kernel terminated, rts panic, Lustre mount FAILED, link
  severed, connection reset, connection timeout, fatal machine check interrupt,
  fatal hardware errors, fatal errors that stop execution.
- Focus on content, type, level, and component.
- Do NOT classify based on level alone.
- Some FATAL entries are normal recovery or diagnostic events.
- Respond with ONLY the single digit 0 or 1. Nothing else."""


class BGLAnomalyClassifier:
    def __init__(self, max_token_given,model_name="meta-llama/Llama-3.2-3B-Instruct"):
        """
        Initialize the meta-llama/Llama-3.2-3B-Instruct model for log classification.
        Supports few-shot modes.
        """
        print(f"Loading model: {model_name}")

        self.model_name = model_name

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # DeepSeek-R1 is a reasoning model — it generates a chain of thought
        # before its final answer. 512 tokens gives it room to reason and
        # then output 0 or 1. With only 10 tokens it gets cut off mid-thought
        # and produces garbage like "I need to figure out what this log line..."
        self.max_new_tokens = max_token_given

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def create_zero_shot_prompt(self, log_line: str) -> str:
        """
        Zero-shot prompt — no examples provided.
        Following DeepSeek-R1 recommendation: no system prompt,
        all instructions inside the user turn.
        Output instruction is unambiguous: exactly one digit, nothing else.
        """
        return (
            "You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.\n\n"
            f"{_RULES}\n\n"
            f"Log line: {log_line.strip()}\n\n"
            "Classification:"
        )

    def create_few_shot_prompt(self, log_line: str) -> str:
        """
        Few-shot prompt —  fixed labeled examples (4 normal, 4 abnormal)
        followed by the test log line.
        The original label field is never included in any example log text.
        Examples are identical across all models and VMs for fair comparison.
        """
        examples_block = ""
        for ex in FEW_SHOT_EXAMPLES:
            examples_block += f"Log line: {ex['log']}\nClassification: {ex['label']}\n\n"

        return (
            "You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.\n\n"
            f"{_RULES}\n\n"
            "Examples:\n\n"
            f"{examples_block}"
            "Now classify this log line:\n"
            f"Log line: {log_line.strip()}\n\n"
            "Classification:"
        )

    # ------------------------------------------------------------------
    # Prediction parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, log_line: str) -> dict:
        """
        Parse model output into a Label.

        meta-llama/Llama-3.2-3B-Instruct generates a reasoning chain first, then the answer.
        Strategy: strip <think> blocks, then read the LAST 0 or 1 in the
        output — the final digit is the answer, not the first one found
        inside the reasoning text.

        Parsing order:
          1. Strip <think>...</think> blocks and whitespace.
          2. Exact match on cleaned text: "0" or "1".
          3. Last standalone 0 or 1 token (answer comes after reasoning).
          4. Last 0 or 1 character anywhere.
          5. Default to "0" and log a warning.
        """
        # Strip reasoning tags — DeepSeek-R1 wraps thinking in <think>...</think>
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        cleaned = cleaned.strip()

        # Exact match — ideal case, model output just "0" or "1"
        if cleaned in ("0", "1"):
            return {"Label": cleaned, "line": log_line.strip()}

        # Last standalone 0 or 1 token — answer appears at end of reasoning
        matches = list(re.finditer(r"\b([01])\b", cleaned))
        if matches:
            return {"Label": matches[-1].group(1), "line": log_line.strip()}

        # Last 0 or 1 character anywhere in output
        matches = list(re.finditer(r"[01]", cleaned))
        if matches:
            return {"Label": matches[-1].group(), "line": log_line.strip()}

        # Could not parse — default to normal and flag it
        print(f"  [WARN] Could not parse response: '{response[:80]}' — defaulting to 0")
        return {"Label": "0", "line": log_line.strip()}

    # ------------------------------------------------------------------
    # Single-line classification
    # ------------------------------------------------------------------

    def classify_log_line(self, log_line: str, mode: str = "zero_shot") -> dict:
        """
        Classify a single log line.

        Args:
            log_line : raw BGL log line (label field must already be excluded
                       by the caller if reading from a labeled dataset)
            mode     : "zero_shot" or "few_shot"

        Returns:
            {"Label": "0"|"1", "line": <log_line>}
        """
        if mode == "few_shot":
            prompt = self.create_few_shot_prompt(log_line)
        else:
            prompt = self.create_zero_shot_prompt(log_line)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,          # deterministic — required for reproducibility
                temperature=1.0,          # override model's built-in default (0.6)
                top_p=1.0,                # override model's built-in default (0.95)
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        

        return self._parse_response(response, log_line)

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        log_lines: list,
        batch_size: int = 4,
        mode: str = "zero_shot",
    ) -> list:
        """
        Classify log lines one at a time (batch_size controls tqdm chunking only).

        Args:
            log_lines  : list of raw log line strings
            batch_size : how many lines to group per tqdm progress update
            mode       : "zero_shot" or "few_shot"
        """
        results = []

        for i in tqdm(range(0, len(log_lines), batch_size), desc=f"Processing ({mode})"):
            batch = log_lines[i:i + batch_size]

            for line in batch:
                if line.strip():
                    try:
                        result = self.classify_log_line(line, mode=mode)
                        results.append(result)
                    except Exception as e:
                        print(f"  [ERROR] Line: {line[:50]}... → {e}")
                        # Record failure as normal rather than silently dropping
                        results.append({"Label": "0", "line": line.strip()})

        return results


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_bgl_log(
    input_file: str,
    output_file: str,
    max_token: int,
    batch_size: int = 4,
    mode: str = "zero_shot",
) -> tuple:
    """
    Process a BGL log file and save results + stats.

    Args:
        input_file  : path to BGL .log file
        output_file : path for output .json (one JSON object per line)
        max_token   : maximum number of tokens to generate
        batch_size  : passed to classify_batch for progress display
        mode        : "zero_shot" or "few_shot"
    """
    if mode not in ("zero_shot", "few_shot"):
        raise ValueError(f"mode must be 'zero_shot' or 'few_shot', got: {mode!r}")

    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024

    print(f"Mode             : {mode}")
    print(f"Starting memory  : {start_memory:.2f} MB")

    if torch.cuda.is_available():
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory before: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    # Initialise classifier
    classifier = BGLAnomalyClassifier(max_token)

    if torch.cuda.is_available():
        print(f"GPU memory after : {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

    # Read log file
    print(f"\nReading: {input_file}")
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        log_lines = f.readlines()
    print(f"Total lines      : {len(log_lines)}")

    # Run classification
    results = classifier.classify_batch(log_lines, batch_size=batch_size, mode=mode)

    # Write results — one JSON object per line (same format as original)
    print(f"\nWriting results  : {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    # Statistics (identical fields to original)
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024
    memory_consumed = end_memory - start_memory
    gpu_memory = (
        torch.cuda.max_memory_allocated(0) / 1024**2
        if torch.cuda.is_available()
        else 0
    )
    time_taken = end_time - start_time
    normal_count   = sum(1 for r in results if r["Label"] == "0")
    abnormal_count = sum(1 for r in results if r["Label"] == "1")

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Mode                    : {mode}")
    print(f"Total logs processed    : {len(results)}")
    print(f"Time taken              : {time_taken:.2f} seconds")
    print(f"Average time per log    : {time_taken/len(results)*1000:.2f} ms")
    print(f"Throughput              : {len(results)/time_taken:.2f} logs/second")
    print(f"\nMemory Consumption:")
    print(f"  CPU memory consumed   : {memory_consumed:.2f} MB")
    print(f"  Peak CPU memory       : {process.memory_info().rss/1024/1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"  GPU memory used       : {gpu_memory:.2f} MB")
    print(f"\nOutput saved to         : {output_file}")
    print("=" * 60)

    stats = {
        "mode":                          mode,
        "total_logs":                    len(results),
        "max_token":                     max_token,
        "time_taken_seconds":            time_taken,
        "avg_time_per_log_ms":           time_taken / len(results) * 1000,
        "throughput_logs_per_second":    len(results) / time_taken,
        "memory_consumed_mb":            memory_consumed,
        "peak_cpu_memory_mb":            process.memory_info().rss / 1024 / 1024,
        "gpu_memory_mb":                 gpu_memory if torch.cuda.is_available() else 0,
        "normal_count":                  normal_count,
        "abnormal_count":                abnormal_count,
        "normal_percentage":             (normal_count / len(results)) * 100,
        "abnormal_percentage":           (abnormal_count / len(results)) * 100,
        "model_used":                    "meta-llama/Llama-3.2-3B-Instruct",
        "batch_size":                    batch_size,
        "gpu_available":                 torch.cuda.is_available(),
    }

    stats_file = output_file.replace(".json", "_stats.json")
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Detailed statistics     : {stats_file}")

    # Sample output
    print("\n" + "=" * 60)
    print("SAMPLE CLASSIFICATIONS (First 10)")
    print("=" * 60)
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. {result}")

    return results, stats


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BGL log anomaly classifier — zero-shot and few-shot modes"
    )
    parser.add_argument(
        "--input_file",
        default="../../Test_500_no_label_sorted.log",
        help="Path to BGL log file",
    )
    parser.add_argument(
        "--output_file",
        default="bgl_deepseek_results_original_1ab3nmt10.json",
        help="Path for output JSON file",
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompting mode: zero_shot or few_shot",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for tqdm progress grouping",
    )
    parser.add_argument(
        "--max_token",
        type=int,
        default=10,
        help="Maximum number of tokens to generate",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        exit(1)

    print(f"Processing BGL logs with meta-llama/Llama-3.2-3B-Instruct [{args.mode}]...")
    process_bgl_log(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        mode=args.mode,
        max_token=args.max_token
    )
