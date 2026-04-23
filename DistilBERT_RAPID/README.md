# Log Anomaly Detection using DistilBERT with RAPID framework

## Overview
This project performs **anomaly detection on system logs (BGL dataset)** using a **pretrained language model (DistilBERT)** and **similarity-based scoring**.

Instead of training a classifier, the approach learns what **normal logs look like** and detects anomalies based on how **different** they are from normal logs.

## Source and Modification
This work is based on the original RAPID framework:

- Repository: https://github.com/DSBA-Lab/RAPID/tree/main  
- Dataset: https://huggingface.co/datasets/logfit-project/BGL  

### Modifications made
The original RAPID framework and dataset were adapted as follows:

- Modified code to run **entirely on CPU**
- Adapted the **BGL dataset format** from HuggingFace to match the RAPID framework’s expected **raw log-line format**
- Fixed preprocessing issues (e.g., header handling, robustness)

## Method Summary
The pipeline follows a **semi-supervised anomaly detection** approach:

1. Train only on **normal logs**
2. Convert logs → **language embeddings**
3. Store embeddings as a **normal memory bank**
4. For each new log:
   - Compare with normal logs using similarity
   - If too different → **anomaly**

## Dataset format
### BGL
- First token = label
  - `-` → normal log
  - others (e.g., `KERNTERM`, `APPSEV`) → anomaly

### EDGE-IIoT (to be done)

## Pipeline steps
### 1. Data Splitting

Script: `split_data.py`

Creates:
- `train` → only normal logs
- `test` → normal + abnormal

### 2. Preprocessing + Representation

Script: `preprocess_rep.py`

For each log:
- Extract timestamp using regex
- Normalize text (remove numbers, special tokens)
- Remove metadata tokens
- Deduplicate logs → unique templates
- Convert to embeddings using DistilBERT

Output:
- `*_representations.pkl`
- `*_unique_lookup_table.pkl`
- `*_label.pkl` (for test)

### 3. Anomaly Detection

Script: `ad_test_coreSet.py`

Methods:
- **KNN (distance-based)**
- **ColBERT-style max similarity**
- **Mean core-set similarity**

Process:
1. Build memory from **train (normal)** embeddings
2. Apply threshold on **test set**
3. Report metrics

## Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion Matrix
- Response time to detect anomalies
- Overall time to execute whole dataset
- Memory consumption

## Execution steps
- python split_data.py --dataset bgl --test_size 0.2 
- python preprocess_rep.py --dataset bgl --plm distilbert-base-uncased --batch_size 32 
- python ad_test_coreSet.py --dataset bgl --plm distilbert-base-uncased 



