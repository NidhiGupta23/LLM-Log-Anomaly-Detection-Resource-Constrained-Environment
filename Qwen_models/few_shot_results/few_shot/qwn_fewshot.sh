#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

#python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file QwenInstructFs4nm10.json --max_token 10
#python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file QwenInstructFs4ab10.json --max_token 10
#python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file QwenInstructFs2ab2nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file QwenInstructFs1ab3nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file QwenInstructFs3ab1nm10.json --max_token 10
python3 Qwen_4B_fewShotBgl4nm.py --output_file Qwen4BFs4nm10.json --max_token 10
python3 Qwen_4B_fewShotBgl4ab.py  --output_file Qwen4BFs4ab10.json --max_token 10
python3 Qwen_4B_fewShotBgl2ab2nm.py --output_file Qwen4BFs2ab2nm10.json --max_token 10
python3 Qwen_4B_fewShotBgl1ab3nm.py --output_file Qwen4BFs1ab3nm10.json --max_token 10
python3 Qwen_4B_fewShotBgl3ab1nm.py --output_file Qwen4BFs3ab1nm10.json --max_token 10
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file QwenInstructFs4nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file QwenInstructFs4nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file QwenInstructFs4nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl4nm.py --output_file QwenInstructFs4nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file QwenInstructFs4ab25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file QwenInstructFs4ab50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file QwenInstructFs4ab75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl4ab.py  --output_file QwenInstructFs4ab100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file QwenInstructFs2ab2nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file QwenInstructFs2ab2nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file QwenInstructFs2ab2nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl2ab2nm.py --output_file QwenInstructFs2ab2nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file QwenInstructFs1ab3nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file QwenInstructFs1ab3nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file QwenInstructFs1ab3nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl1ab3nm.py --output_file QwenInstructFs1ab3nm100.json --max_token 100
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file QwenInstructFs3ab1nm25.json --max_token 25
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file QwenInstructFs3ab1nm50.json --max_token 50
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file QwenInstructFs3ab1nm75.json --max_token 75
python3 Qwen_Instrcut_fewShotBgl3ab1nm.py --output_file QwenInstructFs3ab1nm100.json --max_token 100

rm -rf /root/.cache/huggingface
rm -rf /root/.cache/pip
apt clean
apt autoremove -y
journalctl --vacuum-time=3d


python3 Qwen_4B_fewShotBgl4nm.py --output_file Qwen4BFs4nm25.json --max_token 25
python3 Qwen_4B_fewShotBgl4nm.py --output_file Qwen4BFs4nm50.json --max_token 50
python3 Qwen_4B_fewShotBgl4nm.py --output_file Qwen4BFs4nm75.json --max_token 75
python3 Qwen_4B_fewShotBgl4nm.py --output_file Qwen4BFs4nm100.json --max_token 100
python3 Qwen_4B_fewShotBgl4ab.py  --output_file Qwen4BFs4ab25.json --max_token 25
python3 Qwen_4B_fewShotBgl4ab.py  --output_file Qwen4BFs4ab50.json --max_token 50
python3 Qwen_4B_fewShotBgl4ab.py  --output_file Qwen4BFs4ab75.json --max_token 75
python3 Qwen_4B_fewShotBgl4ab.py  --output_file Qwen4BFs4ab100.json --max_token 100
python3 Qwen_4B_fewShotBgl2ab2nm.py --output_file Qwen4BFs2ab2nm25.json --max_token 25
python3 Qwen_4B_fewShotBgl2ab2nm.py --output_file Qwen4BFs2ab2nm50.json --max_token 50
python3 Qwen_4B_fewShotBgl2ab2nm.py --output_file Qwen4BFs2ab2nm75.json --max_token 75
python3 Qwen_4B_fewShotBgl2ab2nm.py --output_file Qwen4BFs2ab2nm100.json --max_token 100
python3 Qwen_4B_fewShotBgl1ab3nm.py --output_file Qwen4BFs1ab3nm25.json --max_token 25
python3 Qwen_4B_fewShotBgl1ab3nm.py --output_file Qwen4BFs1ab3nm50.json --max_token 50
python3 Qwen_4B_fewShotBgl1ab3nm.py --output_file Qwen4BFs1ab3nm75.json --max_token 75
python3 Qwen_4B_fewShotBgl1ab3nm.py --output_file Qwen4BFs1ab3nm100.json --max_token 100
python3 Qwen_4B_fewShotBgl3ab1nm.py --output_file Qwen4BFs3ab1nm25.json --max_token 25
python3 Qwen_4B_fewShotBgl3ab1nm.py --output_file Qwen4BFs3ab1nm50.json --max_token 50
python3 Qwen_4B_fewShotBgl3ab1nm.py --output_file Qwen4BFs3ab1nm75.json --max_token 75
python3 Qwen_4B_fewShotBgl3ab1nm.py --output_file Qwen4BFs3ab1nm100.json --max_token 100
