import json
import time
import psutil
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import csv

class BGLAnomalyClassifier:
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Initialize the DeepSeek-R1-Distill-Qwen-1.5B model for log classification
        """
        print(f"Loading model: {model_name}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            padding_side='left'
        )
        
        # Load model with appropriate settings for efficiency
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for memory efficiency
            device_map="auto",  # Automatically use GPU if available
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Set padding token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Recommended settings for DeepSeek-R1 models
        self.temperature = 0.4
        self.top_p = 0.95
        self.max_new_tokens = 50  # We only need a short response (0 or 1)
        
    def create_prompt(self, log_line):
        """
        Create the zero-shot prompt for log classification
        Following DeepSeek-R1 recommendations: no system prompt, all instructions in user prompt
        """
        prompt = f"""You are a BGL (Blue Gene/L) log anomaly classifier. You know what to look into a BGL log to predict if log is normal or abnormal.
        Classify the following log line as 0 (normal) or 1 (abnormal).
        < Instructions >
        - 0: Normal logs
        - 1: Abnormal logs
        </ Instructions >
        < Rules >
        - Respond ONLY with the single digit either 0 or 1, no explanations or whitespace.
        - Respond as json file with fields: {{"Label": "0" or "1", "line": "<log_line>"}}
        </ Rules >
        
        Log line: {log_line}
        """
        
        return prompt
    
    def classify_log_line(self, log_line):
        """
        Classify a single log line using the model
        """
        prompt = self.create_prompt(log_line)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        # Move to same device as model
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=False,  # Set to False for deterministic output
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print("Response from LLM: ", response)
        
        # Try to parse JSON from response
        try:
            # Look for JSON pattern in response
            import re
            json_match = re.search(r'\{[^{}]*"Label"[^{}]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                if "Label" in result:
                    return result
        except:
            pass
        
        # Fallback: try to extract 0 or 1 from response
        if "0" in response:
            label = "0"
        elif "1" in response:
            label = "1"
        else:
            label = "1"  # Default to abnormal if parsing fails
        
        return {"Label": label, "line": log_line.strip()}
    
    def classify_batch(self, log_lines, batch_size=4):
        """
        Process logs in batches for better efficiency
        """
        results = []
        
        for i in tqdm(range(0, len(log_lines), batch_size), desc="Processing batches"):
            batch = log_lines[i:i+batch_size]
            batch_results = []
            
            for line in batch:
                if line.strip():
                    try:
                        result = self.classify_log_line(line)
                        batch_results.append(result)
                    except Exception as e:
                        print(f"Error processing line: {line[:50]}... Error: {e}")
                        batch_results.append({"Label": "0", "line": line.strip()})
            
            results.extend(batch_results)
            
        return results

def process_bgl_log(input_file, output_file, batch_size=4):
    """
    Main function to process BGL.log file
    """
    # Memory and time tracking
    process = psutil.Process(os.getpid())
    start_time = time.time()
    start_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Starting memory usage: {start_memory:.2f} MB")
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory before loading: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Initialize classifier
    classifier = BGLAnomalyClassifier()
    
    if torch.cuda.is_available():
        print(f"GPU memory after loading model: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")
    
    # Read log file
    print(f"\nReading log file: {input_file}")
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        log_lines = f.readlines()
    
    print(f"Total log lines: {len(log_lines)}")
    
    # Process logs
    results = classifier.classify_batch(log_lines, batch_size=batch_size)
    
    # Write results to JSON file (one JSON object per line)
    print(f"\nWriting results to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    '''# Write results to CSV file
    print(f"\nWriting results to: {output_file}")

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "line"])  # header

        for result in results:
            writer.writerow([result["Label"], result["line"]])'''

    
    # Calculate final memory and time
    end_time = time.time()
    end_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_consumed = end_memory - start_memory
    
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.max_memory_allocated(0) / 1024**2
    else:
        gpu_memory = 0
    
    time_taken = end_time - start_time
    
    # Print statistics
    print("\n" + "="*60)
    print("PROCESSING STATISTICS")
    print("="*60)
    print(f"Total logs processed: {len(results)}")
    print(f"Time taken: {time_taken:.2f} seconds")
    print(f"Average time per log: {time_taken/len(results)*1000:.2f} ms")
    print(f"Throughput: {len(results)/time_taken:.2f} logs/second")
    print(f"\nMemory Consumption:")
    print(f"  - CPU Memory consumed: {memory_consumed:.2f} MB")
    print(f"  - Peak CPU Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if torch.cuda.is_available():
        print(f"  - GPU Memory used: {gpu_memory:.2f} MB")
    print(f"\nOutput saved to: {output_file}")
    print("="*60)
    
    # Calculate classification statistics
    normal_count = sum(1 for r in results if r["Label"] == "0")
    abnormal_count = sum(1 for r in results if r["Label"] == "1")
    
    # Save detailed statistics
    stats = {
        "total_logs": len(results),
        "time_taken_seconds": time_taken,
        "avg_time_per_log_ms": time_taken/len(results)*1000,
        "throughput_logs_per_second": len(results)/time_taken,
        "memory_consumed_mb": memory_consumed,
        "peak_cpu_memory_mb": process.memory_info().rss / 1024 / 1024,
        "gpu_memory_mb": gpu_memory if torch.cuda.is_available() else 0,
        "normal_count": normal_count,
        "abnormal_count": abnormal_count,
        "normal_percentage": (normal_count/len(results))*100,
        "abnormal_percentage": (abnormal_count/len(results))*100,
        "model_used": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "batch_size": batch_size,
        "temperature": 0.6,
        "gpu_available": torch.cuda.is_available()
    }
    
    stats_file = output_file.replace('.json', '_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\nDetailed statistics saved to: {stats_file}")
    
    # Display sample results
    print("\n" + "="*60)
    print("SAMPLE CLASSIFICATIONS (First 10)")
    print("="*60)
    for i, result in enumerate(results[:10]):
        print(f"{i+1}. {result}")
    
    return results, stats

# Main execution
if __name__ == "__main__":
    input_file = "Test_500_no_label_sorted.log"
    output_file = "bglDeepseekZero_maxToken50.json"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
    
    # Process using the main method
    print("Processing BGL logs with DeepSeek-R1-Distill-Qwen-1.5B...")
    results, stats = process_bgl_log(input_file, output_file, batch_size=4)
    
