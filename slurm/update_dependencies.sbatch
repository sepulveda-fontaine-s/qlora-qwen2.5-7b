#!/bin/bash

#SBATCH --job-name=lora_update_deps
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --output=update_dependencies_%j.out
#SBATCH --error=update_dependencies_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

python -m pip install -e .

python - <<'PY'
import yaml
print("PyYAML:", yaml.__version__)
PY

pip check