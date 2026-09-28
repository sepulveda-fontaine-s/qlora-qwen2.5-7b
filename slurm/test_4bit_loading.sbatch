#!/bin/bash

#SBATCH --job-name=qlora_4bit_load
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=test_4bit_loading_%j.out
#SBATCH --error=test_4bit_loading_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/test_4bit_loading.py