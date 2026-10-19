#!/bin/bash -l
#SBATCH --job-name=geofm-global-encode
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-11%2
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${FRESH_ROOT:?Set absolute FRESH_ROOT}"
.venv/bin/python code/freeze.py --check --cohort
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
models=(clay prithvi prithvi100 random_clay_101 random_prithvi_101 random_prithvi100_101 random_clay_202 random_prithvi_202 random_prithvi100_202 random_clay_303 random_prithvi_303 random_prithvi100_303)
model="${models[$SLURM_ARRAY_TASK_ID]}"
batch=$(.venv/bin/python -c 'import json,sys; m=sys.argv[1]; a=m.split("_")[1] if m.startswith("random_") else m; print(json.load(open(sys.argv[2]))[a]["batch_size"])' "$model" "$FRESH_ROOT/results/batch-sizes.json")
srun .venv/bin/python -u code/extract.py --root "$PWD" --model "$model" --batch-size "$batch"
