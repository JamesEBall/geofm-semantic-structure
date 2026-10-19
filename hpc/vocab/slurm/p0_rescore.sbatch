#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=🌍🤖p0
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --nice=1000
set -euo pipefail
release="${1:?Pass immutable release directory}"
out="${2:?Pass new output directory}"
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$release/code"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$release"
sha256sum --check RELEASE.sha256
python -u code/p0/rescore_archived.py --base "$(dirname "$VOCAB_ROOT")" --out "$out" --bootstrap 1000 --seed 51426
