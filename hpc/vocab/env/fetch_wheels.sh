#!/bin/bash
# LOGIN NODE ONLY: pure download, no install, no venv, no python processing.
# Fills $VOCAB_ROOT/env/wheelhouse with cp312 manylinux wheels for requirements.in (and
# their dependencies), resolved under fresh-01's exact versions, and writes
# env/constraints-fresh01.txt from fresh-01's *.dist-info names.
# Usage (from vocab-01 after staging hpc/vocab/env -> vocab-01/env):  bash env/fetch_wheels.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${VOCAB_ROOT:-$(dirname "$HERE")}"
export VOCAB_ROOT="$ROOT"
source "$HERE/cache_dirs.sh"
FRESH="${VOCAB_FRESH_VENV:-/mnt/scratch/users/40021461/geofm-semantic-expansion/fresh-01/.venv}"
WH="$ROOT/env/wheelhouse"

BASE_PY="$(sed -n 's/^executable *= *//p' "$FRESH/pyvenv.cfg")"
PYVER="$(sed -n 's/^version *= *//p' "$FRESH/pyvenv.cfg")"
MM="${PYVER%.*}"
TAG="cp${MM/./}"
FRESH_SP="$FRESH/lib/python$MM/site-packages"
[[ -x "$BASE_PY" && -d "$FRESH_SP" ]] || { echo "fetch_wheels: bad fresh-01 venv $FRESH" >&2; exit 2; }
echo "base interpreter $BASE_PY ($PYVER, $TAG); fresh-01 site-packages $FRESH_SP"

(cd "$FRESH_SP" && ls -1d -- *.dist-info) \
  | sed -E 's/^([^-]+)-(.+)\.dist-info$/\1==\2/' | LC_ALL=C sort -f >"$HERE/constraints-fresh01.txt"
echo "constraints: $(wc -l <"$HERE/constraints-fresh01.txt") pins -> $HERE/constraints-fresh01.txt"

platforms=()
for p in manylinux_2_28 manylinux_2_27 manylinux_2_24 manylinux_2_17 manylinux2014 manylinux2010 manylinux1; do
  platforms+=(--platform "${p}_x86_64")
done

# Fresh download dir, swapped in only on success, so the wheelhouse is exactly the resolution
# of requirements.in (no stale wheels from earlier pins).
rm -rf "$WH.new"
mkdir -p "$WH.new"
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 \
PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1 \
  "$BASE_PY" -m pip download --dest "$WH.new" --only-binary=:all: \
    --implementation cp --python-version "$MM" --abi "$TAG" "${platforms[@]}" \
    -c "$HERE/constraints-fresh01.txt" -r "$HERE/requirements.in"
rm -rf "$WH.old"
if [[ -d "$WH" ]]; then
  mv "$WH" "$WH.old"
fi
mv "$WH.new" "$WH"
rm -rf "$WH.old"

echo "wheelhouse $WH:"
ls -1 "$WH"
