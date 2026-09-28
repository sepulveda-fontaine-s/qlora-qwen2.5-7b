#!/bin/bash

#SBATCH --job-name=lora_download_model
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=download_model_%j.out
#SBATCH --error=download_model_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/download_model.py