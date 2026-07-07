#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

# Run each Python file in sequence

python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file LlamaInstructFs4nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file LlamaInstructFs4ab10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file LlamaInstructFs2ab2nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file LlamaInstructFs1ab3nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file LlamaInstructFs3ab1nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file LlamaInstructFs4nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file LlamaInstructFs4ab50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file LlamaInstructFs2ab2nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file LlamaInstructFs1ab3nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file LlamaInstructFs3ab1nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file LlamaInstructFs4nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file LlamanstructFs4nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file LlamaInstructFs4nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file LlamaInstructFs4ab25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file LlamaInstructFs4ab75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file LlamaInstructFs4ab100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file LlamaInstructFs2ab2nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file LlamaInstructFs2ab2nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file LlamaInstructFs2ab2nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file LlamaInstructFs1ab3nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file LlamaInstructFs1ab3nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file LlamaInstructFs1ab3nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file LlamaInstructFs3ab1nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file LlamaInstructFs3ab1nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file LlamaInstructFs3ab1nm100.json --max_token 100


