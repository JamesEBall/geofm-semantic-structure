#!/bin/bash -l
#SBATCH --job-name=geofm-internal
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-2%1
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
: "${FRESH_ROOT:?Set FRESH_ROOT}" "${INTERNAL_ROOT:?Set INTERNAL_ROOT}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
(cd "$INTERNAL_ROOT" && sha256sum --check SHA256SUMS)
models=(clay prithvi prithvi100)
srun "$FRESH_ROOT/.venv/bin/python" -u "$INTERNAL_ROOT/code/run.py" --fresh "$FRESH_ROOT" --output "$INTERNAL_ROOT/results" --model "${models[$SLURM_ARRAY_TASK_ID]}" --batch 32
