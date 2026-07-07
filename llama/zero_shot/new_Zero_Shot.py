"""
zero_shot_llama32_bgl_supervisor_metrics.py

Zero-shot BGL anomaly classification using a Hugging Face causal/instruct model.
Default model: meta-llama/Llama-3.2-3B-Instruct

Updates included:
- Uses Llama 3.2 3B Instruct by default instead of Qwen.
- Uses a deterministic zero-shot prompt that asks for only one label: 0 or 1.
- Adds supervisor-requested latency metrics:
    P50/P95/P99 response time per log
    P95/P99 per-log throughput
    P5/P1 tail throughput
- Saves per-log response_time_ms and per_log_throughput_logs_per_second.
- Saves run statistics as JSON.

Before running:
  huggingface-cli login
  # Make sure you have accepted access to meta-llama/Llama-3.2-3B-Instruct on Hugging Face.

Example:
  python zero_shot_llama32_bgl_supervisor_metrics.py \
    --input Test_500_no_label_sorted.log \
    --output_prefix bgl_llama32_3b_zero \
    --max_new_tokens 5 \
    --batch_size 4
"""

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


DEFAULT_MODEL = "meta-llama/Llama-3.2-3B-Instruct"


def percentile(values: List[float], q: float) -> float:
    """Compute percentile q from a list of numeric values."""
    if not values:
        return 0.0

    values = sorted(values)
    k = (len(values) - 1) * (q / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)

    if f == c:
        return values[f]

    return values[f] + (values[c] - values[f]) * (k - f)


class BGLAnomalyClassifier:
    def __init__(
        self,
        max_new_tokens: int = 5,
        model_name: str = DEFAULT_MODEL,
        max_length: int = 2048,
        show_responses: bool = False,
    ):
        """Initialize a Hugging Face instruct model for zero-shot BGL classification."""
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_length = max_length
        self.show_responses = show_responses

        print(f"Loading model: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side="left",
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def create_prompt(self, log_line: str) -> str:
        """
        Create a deterministic zero-shot prompt.

        For instruct/chat models such as Llama 3.2 Instruct, use the tokenizer's
        chat template when available. Fallback to a plain prompt otherwise.
        """
        instruction = f"""Classify this BGL supercomputer log line as 0 (normal) or 1 (abnormal).

0 NORMAL: informational messages, corrected hardware errors, warnings, retries, recovery messages, diagnostic messages.
1 ABNORMAL: kernel terminated, panic, failed mounts, severed links, connection reset or timeout, fatal hardware errors, halted execution.

Rules:
- Judge based on the log content.
- Reply with ONLY one digit: 0 or 1.
- Do not explain.

Log line: {log_line.strip()}
Label:"""

        messages = [{"role": "user", "content": instruction}]

        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        return instruction

    @staticmethod
    def extract_label(response: str) -> str:
        """Extract the first standalone 0/1 label from model output."""
        cleaned = response.strip()

        if cleaned in {"0", "1"}:
            return cleaned

        match = re.search(r"(?<!\d)([01])(?!\d)", cleaned)
        if match:
            return match.group(1)

        return "?"

    def classify_log_line(self, log_line: str) -> Dict[str, object]:
        """Classify one log line and record response-time metrics."""
        prompt = self.create_prompt(log_line)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # For device_map="auto", use the embedding/input device.
        input_device = self.model.get_input_embeddings().weight.device
        inputs = {k: v.to(input_device) for k, v in inputs.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        end = time.perf_counter()

        response_time_seconds = end - start
        response_time_ms = response_time_seconds * 1000.0
        per_log_throughput = 1.0 / response_time_seconds if response_time_seconds > 0 else 0.0

        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        if self.show_responses:
            print("Response from LLM:", repr(response))

        label = self.extract_label(response)

        return {
            "Label": label,
            "line": log_line.strip(),
            "raw_response": response.strip(),
            "response_time_ms": round(response_time_ms, 3),
            "per_log_throughput_logs_per_second": round(per_log_throughput, 4),
        }

    def classify_batch(self, log_lines: List[str], batch_size: int = 4) -> List[Dict[str, object]]:
        """
        Process logs in groups for progress display.
        Generation is still performed one line at a time for stable latency measurement.
        """
        results = []

        for i in tqdm(range(0, len(log_lines), batch_size), desc="Processing batches"):
            batch = log_lines[i:i + batch_size]

            for line in batch:
                if not line.strip():
                    continue

                try:
                    result = self.classify_log_line(line)
                except Exception as exc:
                    print(f"Error processing line: {line[:80]}... Error: {exc}")
                    result = {
                        "Label": "?",
                        "line": line.strip(),
                        "raw_response": "",
                        "response_time_ms": 0.0,
                        "per_log_throughput_logs_per_second": 0.0,
                        "error": str(exc),
                    }

                results.append(result)

        return results


def compute_latency_stats(results: List[Dict[str, object]]) -> Dict[str, float]:
    response_times_ms = [
        float(r.get("response_time_ms", 0.0))
        for r in results
        if float(r.get("response_time_ms", 0.0)) > 0
    ]
    throughputs = [
        float(r.get("per_log_throughput_logs_per_second", 0.0))
        for r in results
        if float(r.get("per_log_throughput_logs_per_second", 0.0)) > 0
    ]

    return {
        "p50_response_time_ms": percentile(response_times_ms, 50),
        "p95_response_time_ms": percentile(response_times_ms, 95),
        "p99_response_time_ms": percentile(response_times_ms, 99),
        "p50_per_log_throughput_logs_per_second": percentile(throughputs, 50),
        "p95_per_log_throughput_logs_per_second": percentile(throughputs, 95),
        "p99_per_log_throughput_logs_per_second": percentile(throughputs, 99),
        # For throughput, low values are the slow tail.
        "p05_tail_throughput_logs_per_second": percentile(throughputs, 5),
        "p01_tail_throughput_logs_per_second": percentile(throughputs, 1),
    }


def process_bgl_log(
    input_file: str,
    output_file: str,
    model_name: str,
    max_new_tokens: int,
    batch_size: int = 4,
    max_length: int = 2048,
    limit: int = 0,
    show_responses: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Process a BGL log file and save predictions plus statistics."""
    process = psutil.Process(os.getpid())
    start_time = time.perf_counter()
    start_memory_mb = process.memory_info().rss / 1024 / 1024
    start_cpu_times = process.cpu_times()
    start_cpu_seconds = start_cpu_times.user + start_cpu_times.system

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    print(f"Starting CPU memory usage: {start_memory_mb:.2f} MB")

    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory before loading: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    else:
        print("GPU available: False")

    classifier = BGLAnomalyClassifier(
        max_new_tokens=max_new_tokens,
        model_name=model_name,
        max_length=max_length,
        show_responses=show_responses,
    )

    if torch.cuda.is_available():
        print(f"GPU memory after loading model: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")

    print(f"\nReading log file: {input_file}")
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        log_lines = [line for line in f if line.strip()]

    if limit and limit > 0:
        log_lines = log_lines[:limit]

    print(f"Total log lines: {len(log_lines)}")

    results = classifier.classify_batch(log_lines, batch_size=batch_size)

    print(f"\nWriting predictions to: {output_file}")
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    end_time = time.perf_counter()
    end_memory_mb = process.memory_info().rss / 1024 / 1024
    end_cpu_times = process.cpu_times()
    end_cpu_seconds = end_cpu_times.user + end_cpu_times.system

    time_taken = end_time - start_time
    memory_consumed_mb = max(0.0, end_memory_mb - start_memory_mb)
    cpu_time_used_seconds = max(0.0, end_cpu_seconds - start_cpu_seconds)
    avg_cpu_cores_used = cpu_time_used_seconds / time_taken if time_taken > 0 else 0.0

    if torch.cuda.is_available():
        gpu_memory_mb = torch.cuda.max_memory_allocated(0) / 1024**2
    else:
        gpu_memory_mb = 0.0

    total_logs = len(results)
    normal_count = sum(1 for r in results if r.get("Label") == "0")
    abnormal_count = sum(1 for r in results if r.get("Label") == "1")
    invalid_count = sum(1 for r in results if r.get("Label") not in {"0", "1"})

    latency_stats = compute_latency_stats(results)

    stats = {
        "total_logs": total_logs,
        "max_new_tokens": max_new_tokens,
        "model_used": model_name,
        "time_taken_seconds": time_taken,
        "avg_time_per_log_ms": (time_taken / total_logs * 1000) if total_logs else 0.0,
        "throughput_logs_per_second": (total_logs / time_taken) if time_taken > 0 else 0.0,
        **latency_stats,
        "cpu_memory_consumed_mb": memory_consumed_mb,
        "peak_cpu_memory_mb": end_memory_mb,
        "gpu_memory_mb": gpu_memory_mb,
        "cpu_cores_available": os.cpu_count() or 0,
        "cpu_time_used_seconds": cpu_time_used_seconds,
        "avg_cpu_cores_used": avg_cpu_cores_used,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "invalid_count": invalid_count,
        "normal_percentage": (normal_count / total_logs * 100) if total_logs else 0.0,
        "abnormal_percentage": (abnormal_count / total_logs * 100) if total_logs else 0.0,
        "invalid_percentage": (invalid_count / total_logs * 100) if total_logs else 0.0,
        "batch_size": batch_size,
        "max_length": max_length,
        "gpu_available": torch.cuda.is_available(),
    }

    stats_file = output_file.replace(".json", "_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("PROCESSING STATISTICS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Total logs processed: {total_logs}")
    print(f"Time taken: {stats['time_taken_seconds']:.2f} seconds")
    print(f"Average time per log: {stats['avg_time_per_log_ms']:.2f} ms")
    print(f"Throughput: {stats['throughput_logs_per_second']:.2f} logs/second")
    print(f"P50 response time: {stats['p50_response_time_ms']:.2f} ms")
    print(f"P95 response time: {stats['p95_response_time_ms']:.2f} ms")
    print(f"P99 response time: {stats['p99_response_time_ms']:.2f} ms")
    print(f"P95 per-log throughput: {stats['p95_per_log_throughput_logs_per_second']:.2f} logs/s")
    print(f"P99 per-log throughput: {stats['p99_per_log_throughput_logs_per_second']:.2f} logs/s")
    print(f"P5 tail throughput: {stats['p05_tail_throughput_logs_per_second']:.2f} logs/s")
    print(f"P1 tail throughput: {stats['p01_tail_throughput_logs_per_second']:.2f} logs/s")
    print("\nMemory Consumption:")
    print(f"  - CPU memory consumed: {memory_consumed_mb:.2f} MB")
    print(f"  - Peak CPU memory: {end_memory_mb:.2f} MB")
    if torch.cuda.is_available():
        print(f"  - Peak GPU memory used: {gpu_memory_mb:.2f} MB")
    print("\nClassification counts:")
    print(f"  - Normal (0): {normal_count}")
    print(f"  - Abnormal (1): {abnormal_count}")
    print(f"  - Invalid (?): {invalid_count}")
    print(f"\nOutput saved to: {output_file}")
    print(f"Detailed statistics saved to: {stats_file}")
    print("=" * 60)

    print("\n" + "=" * 60)
    print("SAMPLE CLASSIFICATIONS (First 10)")
    print("=" * 60)
    for i, result in enumerate(results[:10]):
        print(f"{i + 1}. {result}")

    return results, stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-shot BGL anomaly classification with Llama 3.2 3B Instruct.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", default="Test_500_no_label_sorted.log", help="Input BGL log file.")
    parser.add_argument("--output_prefix", default="bgl_llama32_3b_zero", help="Prefix for output JSON files.")
    parser.add_argument("--model_name", default=DEFAULT_MODEL, help="Hugging Face model name.")
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        nargs="+",
        default=[5],
        help="One or more max_new_tokens values to test. Use 5 for strict 0/1 labels.",
    )
    parser.add_argument("--batch_size", type=int, default=4, help="Progress grouping size; generation is one log at a time.")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum input prompt length.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of lines; 0 means all.")
    parser.add_argument("--show_responses", action="store_true", help="Print raw model responses.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    for max_tokens in args.max_new_tokens:
        output_file = f"{args.output_prefix}_maxToken{max_tokens}.json"
        print(f"\nProcessing BGL logs with max_new_tokens={max_tokens} ...")
        process_bgl_log(
            input_file=args.input,
            output_file=output_file,
            model_name=args.model_name,
            max_new_tokens=max_tokens,
            batch_size=args.batch_size,
            max_length=args.max_length,
            limit=args.limit,
            show_responses=args.show_responses,
        )

