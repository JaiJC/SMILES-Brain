#!/bin/bash
#SBATCH --job-name=sb-dmpnn
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

# Activate the project environment before submitting, or add the cluster-specific
# module/conda commands here. The preflight below fails before training if CUDA or
# Chemprop is unavailable, rather than silently running a CPU job.
# module load cuda/12.1
# source "$(conda info --base)/etc/profile.d/conda.sh"
# conda activate smilesbrain

mkdir -p slurm/logs
CONFIG="${1:-configs/affinity_dmpnn_slurm.yaml}"

echo "node: $(hostname)  gpu: ${CUDA_VISIBLE_DEVICES:-none}  config: $CONFIG"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
python - <<'PY'
import sys
import torch
import chemprop

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to run the production D-MPNN job.")
if not str(getattr(chemprop, "__version__", "")).startswith("1.7"):
    raise SystemExit("Chemprop 1.7.x is required by the D-MPNN adapter.")
print("python:", sys.executable)
print("torch:", torch.__version__)
print("chemprop:", chemprop.__version__)
print("cuda:", torch.cuda.get_device_name(0))
PY
python scripts/train_affinity.py --config "$CONFIG"
