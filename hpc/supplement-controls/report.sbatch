#!/bin/bash -l
#SBATCH --job-name=geofm-rich-controls-report
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
SUPPLEMENT="$(cd ../supplement-controls && pwd)"
(cd "$SUPPLEMENT" && sha256sum -c SHA256SUMS)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
srun .venv/bin/python "$SUPPLEMENT/code/report.py" --out "$1"
