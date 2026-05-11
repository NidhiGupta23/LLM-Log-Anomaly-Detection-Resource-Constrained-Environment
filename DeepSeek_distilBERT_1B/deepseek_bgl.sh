#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

# Run each Python file in sequence

python zero_prompt_bgl_maxToken10.py
python zero_prompt_bgl_maxToken25.py
python zero_prompt_bgl_maxToken50.py
python zero_prompt_bgl_maxToken75.py

echo "All scripts executed successfully."
