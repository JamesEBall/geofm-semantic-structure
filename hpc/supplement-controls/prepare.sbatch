#!/bin/bash -l
#SBATCH --job-name=geofm-rich-controls-prep
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
# Supply ROOT OUT as positional arguments. Submit from fresh-01 for its venv.
SUPPLEMENT="$(cd ../supplement-controls && pwd)"
(cd "$SUPPLEMENT" && sha256sum -c SHA256SUMS)
srun .venv/bin/python "$SUPPLEMENT/code/controls.py" --root "$1" --out "$2" --prepare
