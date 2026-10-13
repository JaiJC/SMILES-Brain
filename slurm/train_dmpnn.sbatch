#!/bin/bash
#SBATCH --job-name=sb-dmpnn
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail
mkdir -p slurm/logs
CONFIG="${1:-configs/affinity_dmpnn.yaml}"

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-none}  config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python scripts/train_affinity.py --config "$CONFIG"
