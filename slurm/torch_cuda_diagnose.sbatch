#!/bin/bash

#SBATCH --job-name=lora_cuda_diag
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=torch_cuda_diagnose_%j.out
#SBATCH --error=torch_cuda_diagnose_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

echo "=== HOST ==="
hostname

echo
echo "=== SLURM GPU VARIABLES ==="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_GPUS=${SLURM_GPUS}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"

echo
echo "=== NVIDIA-SMI ==="
nvidia-smi

echo
echo "=== PYTORCH ==="
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

print("\n=== EXPLICIT CUDA INIT ===")
try:
    torch.cuda.init()
    print("CUDA init: OK")
    print("GPU:", torch.cuda.get_device_name(0))
except Exception as exc:
    print("CUDA init FAILED:")
    print(type(exc).__name__, str(exc))
PY

echo
echo "=== PIP CHECK ==="
pip check