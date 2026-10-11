#!/bin/bash
#SBATCH --job-name=sb-fetch
#SBATCH --output=slurm/logs/%x-%j.out
#SBATCH --error=slurm/logs/%x-%j.err
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
# Download ChEMBL affinities + Hansen atlas and assemble the label table.
# Run on a node with outbound internet (often a login/data-mover node, NOT a compute node).
# If compute nodes are offline, run this interactively on a login node instead of via sbatch.

set -euo pipefail
mkdir -p slurm/logs

# source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate smiles-brain

python -m smilesbrain.data.fetch_atlas
python -m smilesbrain.data.fetch_affinity
python -m smilesbrain.data.assemble
echo "data ready"
