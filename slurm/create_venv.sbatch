#!/bin/bash

#SBATCH --job-name=lora_create_venv
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=create_venv_%j.out
#SBATCH --error=create_venv_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "=== HOST ==="
hostname

echo
echo "=== SYSTEM PYTHON ==="
python3 --version
which python3

echo
echo "=== CREATE VENV ==="
python3 -m venv --copies .venv

echo
echo "=== ACTIVATE VENV ==="
source .venv/bin/activate

echo
echo "=== VENV PYTHON ==="
python --version
which python

echo
echo "=== UPGRADE PACKAGING TOOLS ==="
python -m pip install --upgrade pip setuptools wheel

echo
echo "=== PIP CHECK ==="
pip check