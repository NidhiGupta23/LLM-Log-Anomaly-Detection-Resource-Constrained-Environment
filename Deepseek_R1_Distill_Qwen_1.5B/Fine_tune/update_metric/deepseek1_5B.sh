python3 EvalDeepseek.py  --gguf   /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf   --test ../../Test_500_no_label_sorted.log --output deepseek16gguf500_v1.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf  --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeExtended_v1.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf   --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeMinimal_v1.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf  --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeOriginall_v1.json



python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf --test ../Test_500_no_label_sorted.log --output DeepseekQ4_K_M500logs_modeFull_v1.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeExtended_v1.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf   --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeMinimal_v1.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeOriginal_v1.json


python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeFull_v2.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeExtended_v1.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf   --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeMinimal_v1.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeOriginal_v1.json



python3 EvalDeepseek.py  --gguf  /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf   --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeFull_v2.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf  --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeExtended_v2.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf   --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeMinimal_v2.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b.f16.gguf  --test ../Test_500_no_label_sorted.log --output Deepseek16gguf500logs_modeOriginall_v2.json


python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf --test ../Test_500_no_label_sorted.log --output DeepseekQ4_K_M500logs_modeFull_v2.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeExtended_v2.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf   --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeMinimal_v2.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q4_K_M.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ4500logs_modeOriginal_v2.json


python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeFull_v2.json --mode full
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeExtended_v2.json --mode extended
python3 EvalDeepseek.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf   --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeMinimal_v2.json --mode minimal
python3 EvalDeepSeek2column.py   --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/bgl_deepseek1_5b_Q8_0.gguf  --test ../Test_500_no_label_sorted.log --output DeepseekQ8500logs_modeOriginal_v2.json

python3 ../../../Qwen3.5-4B/Fine-Tune/QwenInstruct25v05B/EvalBGLF16_updated_token.py --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf   --test ../Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeMinimal_v2.json --mode minimal
python3 ../../../Qwen3.5-4B/Fine-Tune/QwenInstruct25v05B/EvalBGLF16_updated_token.py  --gguf /root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/bgl_1.5b_Q8_0.gguf   --test ../Test_500_no_label_sorted.log --output QwenInstructQ8500logs_modeMinimal_v1.json --mode minimal

