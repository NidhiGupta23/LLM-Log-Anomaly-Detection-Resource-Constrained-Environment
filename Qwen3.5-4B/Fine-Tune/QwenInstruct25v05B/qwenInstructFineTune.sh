#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e


python3 EvalBGLF16_updated_token.py  --gguf   /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeFull_v1.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeExtended_v1.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeMinimal_v1.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeOriginal_v1.json



python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4_K_M500logs_modeFull_v1.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeExtended_v1.json --mode extended                                                                                                                                                                                
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeMinimal_v1.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeOriginal_v1.json


python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeFull_v2.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeExtended_v1.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeMinimal_v1.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeOriginal_v1.json



python3 EvalBGLF16_updated_token.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeFull_v2.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeExtended_v2.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeMinimal_v2.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b.f16.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstruct16gguf500logs_modeOriginal_v2.json


python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4_K_M500logs_modeFull_v2.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeExtended_v2.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeMinimal_v2.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ4500logs_modeOriginal_v2.json


python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeFull_v2.json --mode full
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeExtended_v2.json --mode extended
python3 EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q4_K_M.gguf   --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeMinimal_v2.json --mode minimal
python3 EvaluationBGL1_5B_F16.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf  --test ../../../Deepseek_R1_Distill_Qwen_1.5B/Fine_tune/Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeOriginal_v2.json






