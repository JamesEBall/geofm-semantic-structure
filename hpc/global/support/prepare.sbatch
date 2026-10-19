#!/bin/bash -l
#SBATCH --job-name=geofm-support-map
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
: "${GEOFM_PROXY_HELPER:?Set the authenticated proxy helper path}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python support/test_support.py
srun .venv/bin/python "$GEOFM_PROXY_HELPER" .venv/bin/python -u support/select_support.py --workers 32
.venv/bin/python support/commit_inventory.py
