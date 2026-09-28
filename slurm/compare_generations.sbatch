#!/bin/bash

#SBATCH --job-name=qlora_gen_compare
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=compare_generations_%j.out
#SBATCH --error=compare_generations_%j.err
#SBATCH --mail-type=END,FAIL

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONUNBUFFERED=1

python -u scripts/compare_generations.py