#!/bin/bash

#SBATCH --job-name=lora_hw_check
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=hardware_check_%j.out
#SBATCH --error=hardware_check_%j.err

set -e

echo "=== HOST ==="
hostname

echo
echo "=== PYTHON AVAILABLE ==="
python3 --version
which python3

echo
echo "=== NVIDIA / GPU ==="
nvidia-smi

echo
echo "=== GPU MEMORY ==="
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv

echo
echo "=== SYSTEM RAM ==="
free -h

echo
echo "=== CPU ==="
lscpu | grep -E '^CPU\(s\)|Model name'
EOF