#!/bin/bash

#SBATCH --job-name=qlora_adapter
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_qlora_adapter_%j.out
#SBATCH --error=test_qlora_adapter_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/test_qlora_adapter.py