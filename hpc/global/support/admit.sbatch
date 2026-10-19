#!/bin/bash -l
#SBATCH --job-name=geofm-extra-gate
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
sha256sum --check staging/SHA256SUMS
.venv/bin/python code/freeze.py --check
srun .venv/bin/python staging/verify_staged.py
srun .venv/bin/python -u code/aggregate.py
.venv/bin/python code/freeze.py --check --cohort
