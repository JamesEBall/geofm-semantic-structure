#!/bin/bash
# Local or Slurm shell regression checks; no downloads and no package imports.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_ROOT="$(mktemp -d /tmp/megaearth-cache-test.XXXXXX)"
trap 'rm -rf -- "$TEST_ROOT"' EXIT
export TEST_ROOT
bash -eu -o pipefail -s -- "$HERE/cache_dirs.sh" <<'CHILD'
helper=$1
set -- sentinel-one sentinel-two
export VOCAB_ROOT="$TEST_ROOT/root" VOCAB_LOCAL_TMP_BASE="$TEST_ROOT"
export SLURM_JOB_ID=987654321 VOCAB_TMPDIR="" VOCAB_CACHE_DIR=""
export HF_HOME="$HOME/.cache/test-hf" PIP_CACHE_DIR="$HOME/.cache/test-pip"
export PYTHONPYCACHEPREFIX="$HOME/.cache/test-pycache"
trap 'printf done > "$TEST_ROOT/prior-exit-trap"' EXIT
source "$helper"
[[ "$1" == sentinel-one && "$2" == sentinel-two ]]
[[ "$HF_HOME" == "$VOCAB_ROOT/cache/huggingface" ]]
[[ "$PIP_CACHE_DIR" == "$VOCAB_ROOT/cache/pip" ]]
[[ "$PYTHONPYCACHEPREFIX" == "$VOCAB_ROOT/env/pycache" ]]
[[ "$TMPDIR" == "$TEST_ROOT/vocab-$SLURM_JOB_ID" && -d "$TMPDIR" ]]
printf '%s' "$TMPDIR" > "$TEST_ROOT/created-tmp"
source "$helper"
[[ "$1" == sentinel-one && "$2" == sentinel-two ]]
CHILD
[[ -f "$TEST_ROOT/prior-exit-trap" ]]
[[ ! -e "$(cat "$TEST_ROOT/created-tmp")" ]]
printf '%s\n' 'cache_dirs: home redirects, arguments, existing trap, repeat sourcing and cleanup PASS'
