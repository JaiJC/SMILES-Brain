#!/bin/bash
#SBATCH --job-name=sb-affinity
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
# Train the Stage-1 affinity model on an HPC node.
# Usage:  sbatch slurm/train_affinity.sbatch [config]
# Edit the module/conda lines below to match your cluster.

set -euo pipefail
mkdir -p slurm/logs

CONFIG="${1:-configs/affinity_baseline.yaml}"

# --- cluster environment (EDIT ME) ---
# module load cuda/12.1
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate smiles-brain
# --------------------------------------

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-none}  config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true

python scripts/train_affinity.py --config "$CONFIG"
echo "done"
