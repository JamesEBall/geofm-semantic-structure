#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --job-name=🌍🤖gpucheck
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nice=1000
set -euo pipefail
release="${1:?}"
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
export MEGAEARTH_ROOT="$VOCAB_ROOT"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$release/code"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$release"
sha256sum --check RELEASE.sha256 >/dev/null
python -u validation/gpu_adapters.py
