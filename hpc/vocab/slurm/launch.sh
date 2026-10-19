#!/bin/bash
# Submit one governor from a checksum-verified immutable release.
set -euo pipefail
release="$(cd "$(dirname "$0")/.." && pwd)"
profile="${1:-full}"
export MEGAEARTH_ROOT="${MEGAEARTH_ROOT:-/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01}"
export VOCAB_ROOT="$MEGAEARTH_ROOT"
source "$VOCAB_ROOT/env/activate.sh"
cd "$release"
sha256sum --check RELEASE.sha256 >/dev/null
export PYTHONPATH="$release/code"
export MEGAEARTH_RELEASE="$(basename "$release")"
python -m mc.cli --profile "$profile" dag
mkdir -p "$MEGAEARTH_ROOT/logs"
sbatch --parsable --job-name="🌍🤖governor" --output="$MEGAEARTH_ROOT/logs/governor-%j.log" \
  --export=ALL "$release/slurm/mc_governor.sbatch" "$release" "$profile"
