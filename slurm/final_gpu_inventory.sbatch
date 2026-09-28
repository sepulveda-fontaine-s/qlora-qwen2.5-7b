#!/bin/bash

#SBATCH --job-name=final_inventory
#SBATCH --partition=GPU
#SBATCH --qos=gpu
#SBATCH --nodelist=g-0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=final_gpu_inventory_%j.out
#SBATCH --error=final_gpu_inventory_%j.err

set -e

cd "$SLURM_SUBMIT_DIR"

echo "=== PROJECT SIZE ==="
du -sh .

echo
echo "=== MAIN DIRECTORIES ==="
du -sh \
    data \
    results \
    configs \
    scripts \
    slurm \
    2>/dev/null

echo
echo "=== REQUIRED ARTIFACTS ==="

check_path () {
    if [ -e "$1" ]; then
        echo "OK   $1"
    else
        echo "MISS $1"
    fi
}

check_path "data/raw/models/Qwen2.5-7B"
check_path "data/processed/dolly_15k"
check_path "data/processed/dolly_15k_max1024"
check_path "data/processed/dolly_15k_tokenized"

check_path "results/qlora_r8/final_adapter"
check_path "results/qlora_r16/final_adapter"

check_path "results/generation_comparison/base_vs_r8_vs_r16.jsonl"

check_path "results/inference_benchmark/base_vs_qlora_r8.json"
check_path "results/inference_benchmark/base_bf16.json"
check_path "results/inference_benchmark/peft_bf16.json"
check_path "results/inference_benchmark/merged_bf16.json"

check_path "configs/qlora_r8.yaml"
check_path "configs/qlora_r16.yaml"

check_path "cluster_environment.txt"

echo
echo "=== ADAPTER SIZES ==="
du -sh \
    results/qlora_r8/final_adapter \
    results/qlora_r16/final_adapter

echo
echo "=== RESULT ==="
echo "Final GPU inventory: SUCCESS"