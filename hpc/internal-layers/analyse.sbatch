#!/bin/bash -l
#SBATCH --job-name=geofm-internal-analysis
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${FRESH_ROOT:?Set FRESH_ROOT}" "${INTERNAL_ROOT:?Set INTERNAL_ROOT}"
(cd "$INTERNAL_ROOT" && sha256sum --check SHA256SUMS)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
srun "$FRESH_ROOT/.venv/bin/python" "$INTERNAL_ROOT/code/analyse.py" --fresh "$FRESH_ROOT" --output "$INTERNAL_ROOT/results"
