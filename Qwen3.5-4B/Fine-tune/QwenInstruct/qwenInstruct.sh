#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

python ../../../Deepseek_R1_Distill_Qwen_1.5B/fine_tuning/EvalBGLQ8_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_modelQwenInstr_Q8_0.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/fine_tuning/Test_500_no_label_sorted.log --output QwenInstructQ8gguf500logs_modeFull_v1.json --mode full
python EvalBGLQ8_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_modelQwenInstr_Q8_0.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/fine_tuning/Test_500_no_label_sorted.log --output QwenInstructQ8gguf500logs_modeExtended_v1.json --mode extended
python EvalBGLQ8_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_modelQwenInstr_Q8_0.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/fine_tuning/Test_500_no_label_sorted.log --output QwenInstrucQ8gguf500logs_modeMinimal_v1.json --mode minimal
python EvalBGLQ8.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct/snapshots/e89c352af92642226385c3160c9bddf4b481151c/bgl_modelQwenInstr_Q8_0.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/fine_tuning/Test_500_no_label_sorted.log --output QwenInstructQ8gguf500logs_modeOriginal_v1.json
