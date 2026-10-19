#!/bin/bash -l
#SBATCH --job-name=geofm-rich-controls
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-15%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
SUPPLEMENT="$(cd ../supplement-controls && pwd)"
(cd "$SUPPLEMENT" && sha256sum -c SHA256SUMS)
model=$(.venv/bin/python -c 'import sys;sys.path.insert(0,sys.argv[1]);from controls import MODELS;print(MODELS[int(sys.argv[2])])' "$SUPPLEMENT/code" "$SLURM_ARRAY_TASK_ID")
srun .venv/bin/python -u "$SUPPLEMENT/code/controls.py" --root "$1" --out "$2" --model "$model"
