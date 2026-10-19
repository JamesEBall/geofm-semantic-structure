#!/bin/bash -l
#SBATCH --job-name=geofm-fit-stability
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-19%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${SUPPLEMENT_ROOT:?}" "${STUDY_ROOT:?}" "${STUDY_NAME:?}"
(cd "$SUPPLEMENT_ROOT" && sha256sum --check SHA256SUMS)
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
srun "$STUDY_ROOT/.venv/bin/python" "$SUPPLEMENT_ROOT/code/uncertainty.py" fit --root "$STUDY_ROOT" --output "$SUPPLEMENT_ROOT/results/$STUDY_NAME" --replicate "$SLURM_ARRAY_TASK_ID"
