#!/bin/bash -l
#SBATCH --job-name=geofm-cuda-gate
#SBATCH --account=default
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
.venv/bin/python code/freeze.py --check
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi
test -s data/text.npz
.venv/bin/python -u code/cpu_cuda_parity.py --root "$PWD" --models clay prithvi prithvi100 --samples 4 --threads 4
for model in clay prithvi prithvi100; do
 for batch in 32 128; do
  .venv/bin/python -u code/extract.py --root "$PWD" --model "$model" --batch-size "$batch" --limit 256
  mv "features/${model}-diagnostic-256" "features/${model}-diagnostic-256-b${batch}"
 done
done
.venv/bin/python -u code/select_batches.py
