#!/bin/bash -l
#SBATCH --job-name=geofm-output-intervention
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
# Submit from fresh root, with the mechanistic directory as its sibling.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python code/freeze.py --check
(cd ../mechanistic && sha256sum --check SHA256SUMS)
srun .venv/bin/python -u ../mechanistic/code/intervene.py --fresh-root "$PWD" --out ../mechanistic/results
