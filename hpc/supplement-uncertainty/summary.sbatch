#!/bin/bash -l
#SBATCH --job-name=geofm-fit-summary
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${SUPPLEMENT_ROOT:?}" "${STUDY_ROOT:?}" "${STUDY_NAME:?}"
(cd "$SUPPLEMENT_ROOT" && sha256sum --check SHA256SUMS)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun "$STUDY_ROOT/.venv/bin/python" "$SUPPLEMENT_ROOT/code/uncertainty.py" summarise --root "$STUDY_ROOT" --output "$SUPPLEMENT_ROOT/results/$STUDY_NAME"
