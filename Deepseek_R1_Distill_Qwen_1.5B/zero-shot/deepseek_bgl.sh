#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

# Run each Python file in sequence

python3 zero_prompt_bgl_maxToken10.py
python3 zero_prompt_bgl_maxToken25.py
python3 zero_prompt_bgl_maxToken50.py
python3 zero_prompt_bgl_maxToken75.py
python3 zero_prompt_bgl_maxToken100.py

echo "All scripts executed successfully."
