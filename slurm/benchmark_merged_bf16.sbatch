#!/bin/bash

#SBATCH --job-name=merged_bf16_bench
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=benchmark_merged_bf16_%j.out
#SBATCH --error=benchmark_merged_bf16_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -u scripts/benchmark_merged_bf16.py