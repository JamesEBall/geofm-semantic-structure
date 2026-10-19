#!/bin/bash -l
#SBATCH --job-name=geofm-extra-data
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-37%32
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${GEOFM_PROXY_HELPER:?Set the authenticated proxy helper path}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python code/freeze.py --check
exec .venv/bin/python "$GEOFM_PROXY_HELPER" .venv/bin/python -u support/acquire_extra.py "$SLURM_ARRAY_TASK_ID"
