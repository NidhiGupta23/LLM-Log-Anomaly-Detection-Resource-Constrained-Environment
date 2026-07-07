#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e
MODEL_DIR="/root/.cache/huggingface/hub/models--NidhiGupta23--Llama-3.2-3B-Instruct/snapshots/542e71437dfbed201819b59e1c43ded52b55bcec/"
GGUF1="bgl_llama3_3b.f16.gguf"
GGUF2="bgl_llama3_3b_Q4_K_M.gguf"
GGUF3="bgl_llama3_3b_Q8_0.gguf"
FileStarting="LlamaInstruct"
File1="../EvalBGLF16_updated_token.py"
TestFile="../../../Test_5000_no_label_sorted.log"
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_mode2column_v1.json --mode 2column

python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF1" --test "$TestFile" --output "$FileStarting"16gguf5000logs_mode2column_v2.json --mode 2column

python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_mode2column_v1.json --mode 2column

python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF2" --test "$TestFile" --output "$FileStarting"Q45000logs_mode2column_v2.json --mode 2column

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeFull_v1.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeExtended_v1.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeMinimal_v1.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_mode2column_v1.json --mode 2column

python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeFull_v2.json --mode full
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeExtended_v2.json --mode extended
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_modeMinimal_v2.json --mode minimal
python3 "$File1" --gguf "$MODEL_DIR/$GGUF3" --test "$TestFile" --output "$FileStarting"Q85000logs_mode2column_v2.json --mode 2column
