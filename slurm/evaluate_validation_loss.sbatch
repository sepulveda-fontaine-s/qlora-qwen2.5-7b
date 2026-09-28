#!/bin/bash

#SBATCH --job-name=qlora_eval_loss
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=evaluate_validation_loss_%j.out
#SBATCH --error=evaluate_validation_loss_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -u scripts/evaluate_validation_loss.py