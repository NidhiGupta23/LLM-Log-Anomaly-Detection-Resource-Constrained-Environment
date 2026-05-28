#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

python3 EvalBGLF16_updated_token.py  --gguf bgl_1.5b.f16.gguf   --test Test_500_no_label_sorted.log --output claude500logs_modeFull_v1.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf bgl_1.5b.f16.gguf   --test Test_500_no_label_sorted.log --output claude500logs_modeExtended_v1.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf bgl_1.5b.f16.gguf   --test Test_500_no_label_sorted.log --output claude500logs_modeMinimal_v1.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf bgl_1.5b.f16.gguf   --test Test_500_no_label_sorted.log --output claude500logs_modeOriginal_v1.json
