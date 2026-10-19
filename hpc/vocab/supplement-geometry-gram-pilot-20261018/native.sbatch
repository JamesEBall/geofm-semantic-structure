#!/bin/bash
#SBATCH --job-name=megaearth-native-diag
#SBATCH --partition=himem
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --nice=1000
set -euo pipefail
export VOCAB_ROOT=/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01
export VOCAB_THREADS=4
source "$VOCAB_ROOT/env/activate.sh"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
python "$GRAM_PILOT_RELEASE/native_diagnostics.py" --root "$VOCAB_ROOT/runs/retained-cohort-20261018" --output "$GRAM_PILOT_OUTPUT/native-diagnostics.json" --manifest-sha256 d3407b985ccb43b25e41a0d5752295af5244ef16515b5008e391b3b33143e8b4
