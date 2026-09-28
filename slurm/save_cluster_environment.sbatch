#!/bin/bash

#SBATCH --job-name=save_env
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=save_cluster_environment_%j.out
#SBATCH --error=save_cluster_environment_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

source .venv/bin/activate

{
    echo "=== DATE ==="
    date

    echo
    echo "=== HOST ==="
    hostname

    echo
    echo "=== PYTHON ==="
    python --version
    which python

    echo
    echo "=== PIP ==="
    pip --version

    echo
    echo "=== PIP FREEZE ==="
    pip freeze

    echo
    echo "=== PIP CHECK ==="
    pip check

} > cluster_environment.txt

echo
echo "=== RESULT ==="
ls -lh cluster_environment.txt
echo "Environment capture: SUCCESS"