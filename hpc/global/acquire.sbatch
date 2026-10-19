#!/bin/bash -l
#SBATCH --job-name=geofm-global-acquire
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --array=0-119%4
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -c 'import json; assert json.load(open("results/pilot-report.json"))["passed"]'
srun .venv/bin/python -u code/acquire.py acquire --block "$SLURM_ARRAY_TASK_ID" --limit 200
