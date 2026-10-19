#!/bin/bash -l
#SBATCH --job-name=geofm-internal-uncertainty
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SUPPLEMENT_ROOT:?}" "${STUDY_ROOT:?}" "${INTERNAL_RESULTS:?}"
(cd "$SUPPLEMENT_ROOT" && sha256sum --check SHA256SUMS)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
srun "$STUDY_ROOT/.venv/bin/python" "$SUPPLEMENT_ROOT/code/uncertainty.py" intervention --root "$STUDY_ROOT" --output "$SUPPLEMENT_ROOT/results/internal" --internal "$INTERNAL_RESULTS"
