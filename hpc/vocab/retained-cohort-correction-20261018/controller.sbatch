#!/bin/bash
# Author: James Edward Ball. The controller submits work only with explicit --submit.
#SBATCH --account=default
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --partition=short
#SBATCH --time=23:30:00
set -euo pipefail
release="$1"
source_run="$2"
output="$3"
shift 3
: "${MEGAEARTH_ROOT:?Set the source cache/environment root explicitly}"
export VOCAB_ROOT="$MEGAEARTH_ROOT"
source "$VOCAB_ROOT/env/activate.sh"
exec python -u "$release/retained-cohort-correction-20261018/controller.py" \
    --source-run "$source_run" --output "$output" --submit "$@"
