#!/bin/bash
# Author: James Edward Ball.
# Supply resources explicitly with sbatch; inference requires --cpus-per-task=8 --mem=160G.
#SBATCH --account=default
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
set -euo pipefail
release="$1"
source_run="$2"
output="$3"
action="$4"
stage="$5"
phase="$6"
: "${MEGAEARTH_ROOT:?Set the source cache/environment root explicitly}"
export VOCAB_ROOT="$MEGAEARTH_ROOT"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$release/code"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:?}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export POLARS_MAX_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONHASHSEED=0
extra=()
if [[ "$action" == unit ]]; then
    extra=(--unit-index "${SLURM_ARRAY_TASK_ID:?Unit action requires a job array}")
fi
child=''
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
amendment="${7:?Supply the immutable operational amendment path}"
python -u "$amendment/amended_worker.py" --base-release "$release" --manifest "$amendment/manifest.json" -- "$action" \
    --source-run "$source_run" --output "$output" --stage "$stage" --phase "$phase" \
    --threads "$SLURM_CPUS_PER_TASK" "${extra[@]}" &
child=$!
set +e
wait "$child"; result=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; result=$?; done
exit "$result"
