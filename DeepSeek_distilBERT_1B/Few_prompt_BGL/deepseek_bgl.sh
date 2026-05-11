#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

# Run each Python file in sequence

python few_shot_bgl_4normt10.py
python few_shot_bgl_4abnmt10.py
python few_shot_bgl_2ab2nmt10.py
python few_shot_bgl_3ab1nmt10.py
python few_shot_bgl_1ab3nmt10.py
python few_shot_bgl_4normt50.py
python few_shot_bgl_4abnmt50.py
python few_shot_bgl_2ab2nmt50.py
python few_shot_bgl_3ab1nmt50.py
python few_shot_bgl_1ab3nmt50.py
python few_shot_bgl_4normt75.py
python few_shot_bgl_4abnmt75.py
python few_shot_bgl_2ab2nmt75.py
python few_shot_bgl_3ab1nmt75.py 
python few_shot_bgl_1ab3nmt75.py
python few_shot_bgl_4normt100.py
python few_shot_bgl_4abnmt100.py
python few_shot_bgl_2ab2nmt100.py
python few_shot_bgl_3ab1nmt100.py
python few_shot_bgl_1ab3nmt100.py 
python few_shot_bgl_4normt512.py
python few_shot_bgl_4abnmt512.py
python few_shot_bgl_2ab2nmt512.py 
python few_shot_bgl_3ab1nmt512.py
python few_shot_bgl_1ab3nmt512.py

echo "All scripts executed successfully."
