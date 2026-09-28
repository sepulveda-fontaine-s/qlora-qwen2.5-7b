#!/bin/bash

#SBATCH --job-name=lora_overlength
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=analyze_overlength_%j.out
#SBATCH --error=analyze_overlength_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/analyze_overlength.py