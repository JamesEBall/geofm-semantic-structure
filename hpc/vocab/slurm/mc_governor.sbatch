#!/bin/bash
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=23:30:00
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
set -euo pipefail
MEGAEARTH_CODE="$1"; profile="${2:-full}"
export MEGAEARTH_CODE VOCAB_ROOT="${MEGAEARTH_ROOT:?}"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$MEGAEARTH_CODE/code${PYTHONPATH:+:$PYTHONPATH}"
timed_out=0; child=''
trap 'timed_out=1; [[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1
trap '[[ -z "$child" ]] || kill -TERM "$child" 2>/dev/null || true' TERM
python -u -m mc.cli --profile "$profile" governor &
child=$!
set +e
wait "$child"; rc=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; rc=$?; done
set -e
# 75 = yielded (time limit or flag). Only the time-limit signal earns an automatic successor; a
# STOP/PAUSE flag must be cleared by a person. The successor queues behind any maintenance reservation.
if [[ "$rc" -eq 75 && "$timed_out" -eq 1 ]]; then
  echo "governor: time limit reached; resubmitting"
  sbatch --parsable --job-name="${SLURM_JOB_NAME:-🌍🤖governor}" --output="$VOCAB_ROOT/logs/governor-%j.log" \
    --export=ALL "$MEGAEARTH_CODE/slurm/mc_governor.sbatch" "$MEGAEARTH_CODE" "$profile"
fi
exit "$rc"
