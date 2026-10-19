#!/bin/bash
#SBATCH --partition=short
#SBATCH --job-name=🌍🤖mantelx
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --nice=1000
#SBATCH --array=0-9
# Supplementary Mantel null simulations (descriptive): 10 tasks x 100 simulations, ids 200..1199.
set -euo pipefail
release="$(readlink -f "${1:?}")"
outdir="$(readlink -f "${2:?}")"
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
export MEGAEARTH_ROOT="$VOCAB_ROOT"
export MEGAEARTH_THREADS="${SLURM_CPUS_PER_TASK:-4}"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$release/code"
cd "$release"
sha256sum --check RELEASE.sha256 >/dev/null
start=$((200 + SLURM_ARRAY_TASK_ID * 100))
python -u "$outdir/mantel_extra.py" "$release" "$start" 100 "$outdir/mantel-$start.json"
