#!/bin/bash
# Slurm only (needs the SOCKS tunnel, see lib/tunnel.sh): build the conda-forge prefix env
# $VOCAB_ROOT/env/osm with the C++ osmium-tool (+ pyosmium) using the static micromamba in
# env/bin (tarball downloaded on the login node). Reproduces env/osm-lock.txt exactly when
# present; set VOCAB_OSM_RESOLVE=1 to re-solve "osmium-tool=1.16 pyosmium" and rewrite it.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
  echo "build_osm_env: run inside Slurm" >&2
  exit 2
fi
: "${VOCAB_PROXY:?build_osm_env: start the tunnel first (tunnel_start)}"
source "$HERE/cache_dirs.sh"

MM="$HERE/bin/micromamba"
if [[ ! -x "$MM" ]]; then
  tar -xjf "$HERE/bin/micromamba-linux-64.tar.bz2" -C "$HERE" bin/micromamba
fi
echo "micromamba $("$MM" --version)"

export MAMBA_NO_BANNER=1
PREFIX="$HERE/osm"
LOCK="$HERE/osm-lock.txt"
case "$PREFIX" in */env/osm) rm -rf "$PREFIX" ;; *) exit 2 ;; esac

t0=$SECONDS
if [[ -f "$LOCK" && "${VOCAB_OSM_RESOLVE:-0}" != 1 ]]; then
  echo "creating $PREFIX from $LOCK"
  "$MM" create -y --no-rc -p "$PREFIX" --file "$LOCK"
else
  echo "solving osmium-tool=1.16 pyosmium from conda-forge"
  "$MM" create -y --no-rc -p "$PREFIX" --override-channels -c conda-forge \
    --strict-channel-priority osmium-tool=1.16 pyosmium
  "$MM" env export --no-rc -p "$PREFIX" --explicit --md5 >"$LOCK"
fi
echo "osm env built in $(( SECONDS - t0 )) s"
"$PREFIX/bin/osmium" --version | head -3
"$PREFIX/bin/python" -c 'import osmium.version as v; print("pyosmium", v.pyosmium_release, "libosmium", v.libosmium_version)'
