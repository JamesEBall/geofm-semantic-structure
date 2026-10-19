#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=🌍🤖qualcheck
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --nice=1000
set -euo pipefail
release="$(readlink -f "${1:?}")"
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
export MEGAEARTH_ROOT="$VOCAB_ROOT"
export MEGAEARTH_THREADS="${SLURM_CPUS_PER_TASK:-8}"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$release/code"
cd "$release"
sha256sum --check RELEASE.sha256 >/dev/null
python -u validation/qualify_smoke.py "$release" full
