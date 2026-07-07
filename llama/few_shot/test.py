import argparse
import json
import os
import re
import time
from typing import Dict, List, Tuple

import psutil
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Fixed few-shot examples
# These examples are kept from the original file and should stay identical
# across models, modes, and VMs for fair comparison.
# The original BGL label column is excluded from the log text.
# ---------------------------------------------------------------------------
FEW_SHOT_EXAMPLES = [
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
            "1123914727 2005.08.12 R55-M1-N0-I:J18-U11 "
            "2005-08-12-23.32.07.413454 R55-M1-N0-I:J18-U11 RAS APP FATAL "
            "ciod: Error loading /bgl/apps/scaletest/performance/MINIBEN/"
            "mb_243_0810/allreduce.rts: invalid or missing program image, "
            "Exec format error"
        ),
        "label": "0",
    },
    {
        "log": (
            "1132236972 2005.11.17 R72-M1-N6-C:J04-U01 "
            "2005-11-17-06.16.12.504899 R72-M1-N6-C:J04-U01 RAS KERNEL INFO "
            "26741629 torus sender z- retransmission error(s) (dcr 0x02f9) "
            "detected and corrected over 268 seconds"
        ),
        "label": "0",
    },
    {
        "log": (
            "1125225358 2005.08.28 R54-M0-NC-I:J18-U01 "
            "2005-08-28-03.35.58.673640 R54-M0-NC-I:J18-U01 RAS KERNEL INFO "
            "NFS Mount failed on bglio716, slept 15 seconds, retrying (1)"
        ),
        "label": "0",
    },
]


_RULES = """Rules:
- 0 = NORMAL: informational messages, corrected hardware errors, cache parity
  corrected, DDR errors corrected, CE symbols, core file generation, alignment
  exceptions, register dumps, routine warnings, diagnostic messages,
  recovery/retry messages.
- 1 = ABNORMAL: kernel terminated, RTS panic, Lustre mount FAILED, link
  severed, connection reset, connection timeout, fatal machine-check interrupt,
  fatal hardware errors, fatal errors that stop execution.
- Focus on content, type, level, and component.
- Do NOT classify based on level alone.
- Some FATAL entries are normal recovery or diagnostic events.
- Reply with ONLY the single digit 0 or 1. Nothing else."""


def percentile(values: List[float], q: float) -> float:
    """Compute percentile q from a list of numeric values."""
    if not values:
        return 0.0

    ordered = sorted(values)
    k = (len(ordered) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(ordered) - 1)

    if f == c:
        return ordered[f]

    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def response_time_summary(results: List[Dict]) -> Dict[str, float]:
    """Compute P50/P95/P99 latency and throughput-tail metrics."""
    response_times_ms = [
        float(r.get("response_time_ms", 0.0))
        for r in results
        if float(r.get("response_time_ms", 0.0)) > 0
    ]

    per_log_tput = [
        float(r.get("per_log_throughput_logs_per_second", 0.0))
        for r in results
        if float(r.get("per_log_throughput_logs_per_second", 0.0)) > 0
    ]

    return {
        "p50_response_time_ms": percentile(response_times_ms, 50),
        "p95_response_time_ms": percentile(response_times_ms, 95),
        "p99_response_time_ms": percentile(response_times_ms, 99),
        "p50_per_log_throughput_logs_per_second": percentile(per_log_tput, 50),
        "p95_per_log_throughput_logs_per_second": percentile(per_log_tput, 95),
        "p99_per_log_throughput_logs_per_second": percentile(per_log_tput, 99),
        # For throughput, lower values are the slow tail.
        "p05_tail_throughput_logs_per_second": percentile(per_log_tput, 5),
        "p01_tail_throughput_logs_per_second": percentile(per_log_tput, 1),
    }


class BGLAnomalyClassifier:
    def __init__(
        self,
        max_token_given: int,
        model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
        max_input_length: int = 2048,
    ):
        """Initialize Meta Llama 3.2 3B Instruct for BGL classification."""
        print(f"Loading model: {model_name}")

        self.model_name = model_name
        self.max_new_tokens = max_token_given
        self.max_input_length = max_input_length
        self._chat_template_failed = False

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # Works for a single GPU/V100 setup and also for CPU fallback.
        self.device = next(self.model.parameters()).device

    # ------------------------------------------------------------------
    # Prompt builders
    # ------------------------------------------------------------------

    def create_zero_shot_prompt(self, log_line: str) -> str:
        return (
            "You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.\n\n"
            f"{_RULES}\n\n"
            f"Log line: {log_line.strip()}\n\n"
            "Classification:"
        )

    def create_few_shot_prompt(self, log_line: str) -> str:
        examples_block = ""
        for ex in FEW_SHOT_EXAMPLES:
            examples_block += (
                f"Log line: {ex['log']}\n"
                f"Classification: {ex['label']}\n\n"
            )

        return (
            "You are a BGL (Blue Gene/L) supercomputer log anomaly classifier.\n\n"
            f"{_RULES}\n\n"
            "Examples:\n\n"
            f"{examples_block}"
            "Now classify this log line:\n"
            f"Log line: {log_line.strip()}\n\n"
            "Classification:"
        )

    def _format_for_llama_instruct(self, prompt: str) -> str:
        """
        Use the Llama instruct chat template when possible.
        If the environment has old jinja2, fall back to the plain prompt.
        """
        if self._chat_template_failed or not getattr(self.tokenizer, "chat_template", None):
            return prompt

        try:
            messages = [{"role": "user", "content": prompt}]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:
            self._chat_template_failed = True
            print(
                "[WARN] Could not apply chat template. "
                "Falling back to plain prompt. "
                f"Reason: {exc}"
            )
            return prompt

    # ------------------------------------------------------------------
    # Prediction parsing
    # ------------------------------------------------------------------

    def _parse_response(self, response: str, log_line: str) -> Dict:
        cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        cleaned = cleaned.strip()

        if cleaned in ("0", "1"):
            label = cleaned
        else:
            matches = list(re.finditer(r"\b([01])\b", cleaned))
            if matches:
                label = matches[-1].group(1)
            else:
                label = "?"

        return {
            "Label": label,
            "line": log_line.strip(),
            "raw_response": response.strip(),
        }

    # ------------------------------------------------------------------
    # Single-line classification
    # ------------------------------------------------------------------

    def classify_log_line(self, log_line: str, mode: str = "few_shot") -> Dict:
        if mode == "few_shot":
            prompt = self.create_few_shot_prompt(log_line)
        else:
            prompt = self.create_zero_shot_prompt(log_line)

        formatted_prompt = self._format_for_llama_instruct(prompt)

        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        t0 = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        t1 = time.perf_counter()

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        result = self._parse_response(response, log_line)
        response_time_seconds = t1 - t0
        result["response_time_ms"] = response_time_seconds * 1000.0
        result["per_log_throughput_logs_per_second"] = (
            1.0 / response_time_seconds if response_time_seconds > 0 else 0.0
        )
        return result

    # ------------------------------------------------------------------
    # Batch classification
    # ------------------------------------------------------------------

    def classify_batch(
        self,
        log_lines: List[str],
        batch_size: int = 4,
        mode: str = "few_shot",
    ) -> List[Dict]:
        """Classify one log at a time; batch_size only controls tqdm grouping."""
        results = []

        for i in tqdm(range(0, len(log_lines), batch_size), desc=f"Processing ({mode})"):
            batch = log_lines[i:i + batch_size]

            for line in batch:
                if not line.strip():
                    continue
                try:
                    results.append(self.classify_log_line(line, mode=mode))
                except Exception as exc:
                    print(f"  [ERROR] Line: {line[:80]}... -> {exc}")
                    results.append({
                        "Label": "?",
                        "line": line.strip(),
                        "raw_response": "",
                        "response_time_ms": 0.0,
                        "per_log_throughput_logs_per_second": 0.0,
                        "error": str(exc),
                    })

        return results


def process_bgl_log(
    input_file: str,
    output_file: str,
    max_token: int,
    batch_size: int = 4,
    mode: str = "few_shot",
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct",
    max_input_length: int = 2048,
) -> Tuple[List[Dict], Dict]:
    if mode not in ("zero_shot", "few_shot"):
        raise ValueError(f"mode must be 'zero_shot' or 'few_shot', got: {mode!r}")

    process = psutil.Process(os.getpid())
    start_memory = process.memory_info().rss / 1024 / 1024

    print(f"Mode             : {mode}")
    print(f"Model            : {model_name}")
    print(f"Starting memory  : {start_memory:.2f} MB")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU              : {torch.cuda.get_device_name(0)}")
        print(f"GPU memory before: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("GPU              : not available")

    classifier = BGLAnomalyClassifier(
        max_token_given=max_token,
        model_name=model_name,
        max_input_length=max_input_length,
    )

    if torch.cuda.is_available():
        print(f"GPU memory after : {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print(f"\nReading: {input_file}")
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        log_lines = [line for line in f.readlines() if line.strip()]
    print(f"Total lines      : {len(log_lines)}")

    inference_start = time.perf_counter()
    results = classifier.classify_batch(log_lines, batch_size=batch_size, mode=mode)
    inference_end = time.perf_counter()
    time_taken = inference_end - inference_start

    print(f"\nWriting results  : {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    end_memory = process.memory_info().rss / 1024 / 1024
    memory_consumed = end_memory - start_memory
    gpu_memory = (
        torch.cuda.max_memory_allocated(0) / 1024**2
        if torch.cuda.is_available()
        else 0.0
    )

    total = len(results)
    normal_count = sum(1 for r in results if r["Label"] == "0")
    abnormal_count = sum(1 for r in results if r["Label"] == "1")
    invalid_count = sum(1 for r in results if r["Label"] not in ("0", "1"))
    latency_stats = response_time_summary(results)

    avg_time_per_log_ms = time_taken / max(1, total) * 1000.0
    throughput = total / time_taken if time_taken > 0 else 0.0

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Mode                    : {mode}")
    print(f"Model                   : {model_name}")
    print(f"Total logs processed    : {total}")
    print(f"Invalid outputs         : {invalid_count}")
    print(f"Time taken              : {time_taken:.2f} seconds")
    print(f"Average time per log    : {avg_time_per_log_ms:.2f} ms")
    print(f"Throughput              : {throughput:.2f} logs/second")
    print(f"P50 response time       : {latency_stats['p50_response_time_ms']:.2f} ms")
    print(f"P95 response time       : {latency_stats['p95_response_time_ms']:.2f} ms")
    print(f"P99 response time       : {latency_stats['p99_response_time_ms']:.2f} ms")
    print(f"P95 per-log throughput  : {latency_stats['p95_per_log_throughput_logs_per_second']:.2f} logs/s")
    print(f"P99 per-log throughput  : {latency_stats['p99_per_log_throughput_logs_per_second']:.2f} logs/s")
    print(f"P5 tail throughput      : {latency_stats['p05_tail_throughput_logs_per_second']:.2f} logs/s")
    print(f"P1 tail throughput      : {latency_stats['p01_tail_throughput_logs_per_second']:.2f} logs/s")
    print("\nMemory Consumption:")
    print(f"  CPU memory consumed   : {memory_consumed:.2f} MB")
    print(f"  Peak CPU memory       : {end_memory:.2f} MB")
    if torch.cuda.is_available():
        print(f"  GPU memory used       : {gpu_memory:.2f} MB")
    print(f"\nOutput saved to         : {output_file}")
    print("=" * 60)

    stats = {
        "mode": mode,
        "total_logs": total,
        "invalid_outputs": invalid_count,
        "max_token": max_token,
        "max_input_length": max_input_length,
        "time_taken_seconds": time_taken,
        "avg_time_per_log_ms": avg_time_per_log_ms,
        "throughput_logs_per_second": throughput,
        **latency_stats,
        "memory_consumed_mb": memory_consumed,
        "peak_cpu_memory_mb": end_memory,
        "gpu_memory_mb": gpu_memory,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "normal_percentage": (normal_count / max(1, total)) * 100.0,
        "abnormal_percentage": (abnormal_count / max(1, total)) * 100.0,
        "model_used": model_name,
        "batch_size": batch_size,
        "gpu_available": torch.cuda.is_available(),
    }

    stats_file = output_file.replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Detailed statistics     : {stats_file}")

    print("\n" + "=" * 60)
    print("SAMPLE CLASSIFICATIONS (First 10)")
    print("=" * 60)
    for i, result in enumerate(results[:10]):
        print(f"{i + 1}. {result}")

    return results, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BGL log anomaly classifier — zero-shot and few-shot Llama 3.2 Instruct"
    )
    parser.add_argument(
        "--input_file",
        default="Test_500_no_label_sorted.log",
        help="Path to BGL log file",
    )
    parser.add_argument(
        "--output_file",
        default="bgl_llama32_3b_fewshot.json",
        help="Path for output JSONL file",
    )
    parser.add_argument(
        "--mode",
        choices=["zero_shot", "few_shot"],
        default="few_shot",
        help="Prompting mode: zero_shot or few_shot",
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-3B-Instruct",
        help="Hugging Face model name",
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
        default=5,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=2048,
        help="Maximum input tokens after tokenization",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        raise SystemExit(1)

    print(f"Processing BGL logs with {args.model_name} [{args.mode}]...")
    process_bgl_log(
        input_file=args.input_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        mode=args.mode,
        max_token=args.max_token,
        model_name=args.model_name,
        max_input_length=args.max_input_length,
    )

