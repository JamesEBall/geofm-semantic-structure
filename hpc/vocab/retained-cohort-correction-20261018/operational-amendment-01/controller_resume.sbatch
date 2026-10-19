#!/bin/bash
#SBATCH --account=default
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:30:00
set -euo pipefail
amendment="$1"
base="$2"
source_run="$3"
output="$4"
handover="$5"
: "${MEGAEARTH_ROOT:?}"
source "$MEGAEARTH_ROOT/env/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -u "$amendment/controller_resume.py" --base-release "$base" --manifest "$amendment/manifest.json" --handover "$handover" --source-run "$source_run" --output "$output" --resume
