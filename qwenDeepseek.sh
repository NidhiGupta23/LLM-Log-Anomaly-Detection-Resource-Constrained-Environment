#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e


MODEL_DIR="/root/.cache/huggingface/hub/models--NidhiGupta23--DeepSeek1_5B/snapshots/729d903ec5e24d0e2ed1ef11dbd4527da13d0b1b/"
GGUF1="bgl_deepseek1_5b.f16.gguf"
GGUF2="bgl_deepseek1_5b_Q4_K_M.gguf"
GGUF3="bgl_deepseek1_5b_Q8_0.gguf"
FileStarting="DeepseekR1distil"
File1="EvalBGLF16_updated_token.py"
File2="EvaluationBGLF16Original.py"
TestFile="Test_500_no_label_sorted.log"
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeOriginal_v2.json


python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeOriginal_v2.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeOriginal_v2.json


MODEL_DIR="/root/.cache/huggingface/hub/models--NidhiGupta23--QwenInstruct25v05B/snapshots/132d2f74c8b824e9242a2b265c16e5ec49e3c275/"
GGUF1="bgl_1.5b.f16.gguf"
GGUF2="bgl_1.5b_Q4_K_M.gguf"
GGUF3="bgl_1.5b_Q8_0.gguf"
FileStarting="QwenInstruct"
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf500logs_modeOriginal_v2.json


python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q4500logs_modeOriginal_v2.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_mode2column_v1.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeOriginal_v1.json

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_mode2column_v2.json --mode 2column
python3 "$File2" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q8500logs_modeOriginal_v2.json
