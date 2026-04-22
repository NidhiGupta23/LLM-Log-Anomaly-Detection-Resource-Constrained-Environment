# LLM Benchmarking in Resource-Constrained Environments

## Overview

This project evaluates and compares the performance of multiple Large Language Models (LLMs) under resource-constrained environments for log analysis tasks. The goal is to understand how lightweight and distilled models perform when applied to real-world system logs and industrial datasets.

The models evaluated in this study include:

- RAPID (from DSBA-Lab)
- DeepSeek-R1-Distill-Qwen-1.5B
- Qwen3-4B
- Tiny-LLM (To do)
- LogLLM (To do)

These models are tested across multiple datasets commonly used in log analysis and anomaly detection:

- BGL (Blue Gene/L)
- HDFS (Hadoop Distributed File System)
- Edge IIoT dataset

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

### 4. LogLLM
Repository: https://github.com/guanwei49/LogLLM/blob/master/README.md

A specialized LLM framework tailored for log understanding and anomaly detection tasks.


## Datasets

### BGL (Blue Gene/L)
- High-performance computing system logs
- Contains labeled anomalies
- Widely used for benchmarking log anomaly detection

### HDFS
- Distributed system logs from Hadoop
- Structured and semi-structured logs
- Common benchmark for log parsing and anomaly detection

### Edge IIoT Dataset
- Industrial IoT logs
- Reflects real-world edge computing constraints
- Includes security and operational anomalies


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

## References
- RAPID: https://github.com/DSBA-Lab/RAPID
- DeepSeek-R1-Distill-Qwen-1.5B: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
- LogLLM: https://github.com/guanwei49/LogLLM
- Qwen Models: https://huggingface.co/Qwen
- Datasets:
- BGL: https://github.com/logpai/loghub/tree/master/BGL
- BGL: https://huggingface.co/datasets/logfit-project/BGL
- Edge-IIoT: https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot/data
- HDFS: https://github.com/logpai/loghub/tree/master/HDFS

## Author
Nidhi Gupta
https://www.linkedin.com/in/nidhi-gupta-6b8116139/
