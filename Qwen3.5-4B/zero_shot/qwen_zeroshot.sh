#!/bin/bash
# This script runs Python files sequentially

# Exit immediately if a command fails
set -e

# Run each Python file in sequence
rm -rf /root/.cache/huggingface 
rm -rf /root/.cache/pip 
apt clean 
apt autoremove -y 
journalctl --vacuum-time=3d 

python3 zero_shot.py

rm -rf /root/.cache/huggingface
rm -rf /root/.cache/pip
apt clean
apt autoremove -y
journalctl --vacuum-time=3d

python3 zero_shot_instruct.py 
