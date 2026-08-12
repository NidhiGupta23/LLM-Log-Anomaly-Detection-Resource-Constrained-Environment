# LLM Benchmarking in Resource-Constrained Environments

## Overview

This project evaluates and compares the performance of multiple Large Language Models (LLMs) under resource-constrained environments for log analysis tasks. The goal is to understand how lightweight and distilled models perform when applied to real-world system logs and industrial datasets.

The models evaluated in this study include:

- RAPID (from DSBA-Lab)
- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen3-4B
- Qwen2.5-0.5B-Instruct
- Llama-3.2-3B-Instruct

These models are tested on below dataset commonly used in log analysis and anomaly detection:

- BGL (Blue Gene/L)

The final outcome is a comparative analysis focusing on accuracy, efficiency, and suitability for deployment in constrained environments (e.g., limited GPU/CPU, memory, or edge devices).


## Objectives

- Benchmark multiple LLM-based log analysis approaches
- Evaluate trade-offs between model size and performance
- Measure inference latency, memory usage, and throughput
- Assess accuracy in anomaly detection and log parsing tasks
- Identify models suitable for edge or low-resource deployment


## Models

### 1. RAPID
Repository: https://github.com/DSBA-Lab/RAPID/tree/main

RAPID is designed for efficient log analysis using structured reasoning and prompt-based techniques. It aims to reduce computational overhead while maintaining strong performance.

### 2. DeepSeek-R1-Distill-Qwen-1.5B
Model: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

A distilled version of DeepSeek-R1 built on Qwen architecture, optimized for efficiency and reduced resource consumption.

### 3. Qwen3-4B

A larger general-purpose LLM used as a baseline for comparison. Provides stronger reasoning capabilities at the cost of higher resource usage.

### 4. Qwen2.5-0.5B-Instruct
Model: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct

Smallest LLM model to test if size can impact the performance

### 5. Llama-3.2
Model: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct


## Datasets

### BGL (Blue Gene/L)
- High-performance computing system logs
- Contains labeled anomalies
- Widely used for benchmarking log anomaly detection


## Experimental Setup

### Environment Constraints

To simulate resource-constrained environments:

- Limited GPU memory (e.g., ≤8GB VRAM)
- CPU-only scenarios
- Reduced batch sizes
- Quantized or distilled model variants where applicable

### Metrics

The following metrics are used for evaluation:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Response time to detect anomalies
- Overall time to execute whole dataset
- Memory consumption

## Methodology

1. Preprocess datasets into a unified format
2. Apply each model to:
   - Log parsing
   - Anomaly detection
3. Standardize prompts across models where applicable
4. Run inference under identical hardware constraints
5. Collect performance and resource usage metrics
6. Compare results across models and datasets

## Results
The results for the 2 VMs can be observed in branch on 8GB_LLM and 16GB_LLM.

## References
- RAPID: https://github.com/DSBA-Lab/RAPID
- DeepSeek-R1-Distill-Qwen-1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- Llama-3.2: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
- Qwen Models: https://huggingface.co/Qwen
- Datasets:
- BGL: https://github.com/logpai/loghub/tree/master/BGL
- BGL: https://huggingface.co/datasets/logfit-project/BGL

## Author
Nidhi Gupta
https://www.linkedin.com/in/nidhi-gupta-6b8116139/
