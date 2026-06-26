#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

python3 EvalBGLF16_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_model_qwenInst.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeFull_v1.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_model_qwenInst.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeExtended_v1.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_model_qwenInst.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeMinimal_v1.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_model_qwenInst.f16.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeOriginal_v1.json
