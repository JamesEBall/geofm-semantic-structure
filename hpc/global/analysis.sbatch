#!/bin/bash -l
#SBATCH --job-name=geofm-global-analysis
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
.venv/bin/python code/freeze.py --check --cohort
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun .venv/bin/python -u code/analyse.py --root "$PWD" --bootstrap 1000
