#!/bin/bash

#SBATCH --job-name=qlora_r16_train
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=train_qlora_r16_%j.out
#SBATCH --error=train_qlora_r16_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -u scripts/train_qlora.py