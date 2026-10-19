#!/bin/bash
#SBATCH --account=default
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --open-mode=append
set -euo pipefail
MEGAEARTH_CODE="$1"; pool="$2"; cpus="$3"; mem="$4"; gpus="$5"; profile="$6"
export MEGAEARTH_CODE VOCAB_ROOT="${MEGAEARTH_ROOT:?}"
source "$VOCAB_ROOT/env/activate.sh"
export PYTHONPATH="$MEGAEARTH_CODE/code${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$pool" == net ]]; then
  source "$MEGAEARTH_CODE/lib/tunnel.sh"
  tunnel_start || exit 75
fi
child=''
trap '[[ -z "$child" ]] || kill -USR1 "$child" 2>/dev/null || true' USR1 TERM
python -u -m mc.cli --profile "$profile" worker --pool "$pool" --cpus "$cpus" --mem-gb "$mem" --gpus "$gpus" &
child=$!
set +e
wait "$child"; result=$?
while kill -0 "$child" 2>/dev/null; do wait "$child"; result=$?; done
exit "$result"
