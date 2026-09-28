#!/bin/bash

#SBATCH --job-name=lora_install_hf
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=install_hf_stack_%j.out
#SBATCH --error=install_hf_stack_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== INSTALL PROJECT ==="
python -m pip install -e .

echo
echo "=== VERSION CHECK ==="
python - <<'PY'
import torch
import transformers
import datasets
import accelerate
import peft
import bitsandbytes as bnb

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("datasets:", datasets.__version__)
print("accelerate:", accelerate.__version__)
print("peft:", peft.__version__)
print("bitsandbytes:", bnb.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
PY

echo
echo "=== PIP CHECK ==="
pip check