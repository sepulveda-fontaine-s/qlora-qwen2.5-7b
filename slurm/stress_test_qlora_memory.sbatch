#!/bin/bash

#SBATCH --job-name=qlora_memory_test
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=72G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=stress_test_qlora_memory_%j.out
#SBATCH --error=stress_test_qlora_memory_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python scripts/stress_test_qlora_memory.py