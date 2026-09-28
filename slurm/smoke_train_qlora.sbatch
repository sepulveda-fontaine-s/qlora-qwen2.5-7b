#!/bin/bash

#SBATCH --job-name=qlora_smoke_train
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=smoke_train_qlora_%j.out
#SBATCH --error=smoke_train_qlora_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/smoke_train_qlora.py