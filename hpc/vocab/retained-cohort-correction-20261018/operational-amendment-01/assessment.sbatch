#!/bin/bash
#SBATCH --account=default
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:30:00
set -euo pipefail
amendment="$1"
base="$2"
source_run="$3"
output="$4"
controller="$5"
: "${MEGAEARTH_ROOT:?}"
source "$MEGAEARTH_ROOT/env/activate.sh"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
exec python -u "$amendment/assessment.py" --base-release "$base" --manifest "$amendment/manifest.json" --source-run "$source_run" --output "$output" --controller-job-id "$controller"
