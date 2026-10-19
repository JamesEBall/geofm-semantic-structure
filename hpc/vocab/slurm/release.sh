#!/bin/bash
# Build an immutable, checksum-verified MegaEarth release from hpc/vocab and (optionally) deploy it.
#
#   slurm/release.sh build            # -> output/megaearth-<UTC stamp>/ with RELEASE.sha256
#   slurm/release.sh deploy [DIR]     # rsync a built release to the HPC releases directory
#   slurm/release.sh verify [DIR]     # re-check RELEASE.sha256 on the HPC copy
#
# Author: James Edward Ball. Deployment uses the login node only for file transfer; nothing is
# executed there. Launch the governor separately with: slurm/launch.sh <profile> (from the release).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"          # hpc/vocab
REPO="$(cd "$HERE/../.." && pwd)"
OUT="$REPO/output"
HPC_ROOT="${MEGAEARTH_ROOT:-/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01}"
# SSH target: the ~/.ssh/config alias `enucc-gw` (ProxyJump through the gateway, persistent control
# socket under ~/.ssh/sockets). Override with MEGAEARTH_SSH_TARGET; MEGAEARTH_SSH_SOCKET adds -S.
TARGET="${MEGAEARTH_SSH_TARGET:-enucc-gw}"
SOCKET="${MEGAEARTH_SSH_SOCKET:-}"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 ${SOCKET:+-S "$SOCKET"} "$TARGET")
RSYNC_SSH="ssh -o BatchMode=yes${SOCKET:+ -S $SOCKET}"

usage() { sed -n '2,9p' "$0"; exit 2; }

build() {
  local stamp dest
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  dest="$OUT/megaearth-$stamp"
  mkdir -p "$OUT"
  # Exclude caches, test work directories and the local venv; tests ship so the HPC can run them.
  rsync -a --exclude='__pycache__' --exclude='.pytest_cache' --exclude='*.pyc' --exclude='.DS_Store' \
        --exclude='paper-drafts' --exclude='work-*' --exclude='env/venv' --exclude='env/osm' --exclude='env/wheelhouse*' \
        --exclude='env/mamba-root' --exclude='env/bin' "$HERE/" "$dest/"
  (cd "$dest" && find . -type f ! -name RELEASE.sha256 -print0 | sort -z |
     xargs -0 shasum -a 256 | sed 's#  \./#  #' > RELEASE.sha256)
  (cd "$dest" && shasum -a 256 --check --quiet RELEASE.sha256)
  python3 -c 'import json,sys; print(json.dumps({"local": sys.argv[1], "name": sys.argv[2]}))' \
     "$dest" "$(basename "$dest")" > "$OUT/megaearth-deploy-current.json"
  echo "$dest"
}

current() {
  python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["local"])' "$OUT/megaearth-deploy-current.json"
}

deploy() {
  local dir="${1:-$(current)}" name
  name="$(basename "$dir")"
  [[ -f "$dir/RELEASE.sha256" ]] || { echo "release.sh: $dir has no RELEASE.sha256 (run build first)" >&2; exit 1; }
  "${SSH[@]}" "mkdir -p '$HPC_ROOT/releases' '$HPC_ROOT/logs'"
  # A release is immutable: refuse to overwrite one that already exists remotely.
  if "${SSH[@]}" "test -e '$HPC_ROOT/releases/$name'"; then
    echo "release.sh: $name already exists on the HPC; build a new release instead" >&2; exit 1
  fi
  rsync -a -e "$RSYNC_SSH" "$dir" "$TARGET:$HPC_ROOT/releases/"
  "${SSH[@]}" "chmod -R u+rwX,go+rX '$HPC_ROOT/releases/$name' && chmod +x '$HPC_ROOT/releases/$name'/slurm/*.sh '$HPC_ROOT/releases/$name'/slurm/*.sbatch '$HPC_ROOT/releases/$name'/env/*.sh 2>/dev/null; cd '$HPC_ROOT/releases/$name' && sha256sum --check --quiet RELEASE.sha256 && echo deployed: $HPC_ROOT/releases/$name"
}

verify() {
  local dir="${1:-$(current)}" name
  name="$(basename "$dir")"
  "${SSH[@]}" "cd '$HPC_ROOT/releases/$name' && sha256sum --check --quiet RELEASE.sha256 && echo verified: $name"
}

case "${1:-}" in
  build) build ;;
  deploy) deploy "${2:-}" ;;
  verify) verify "${2:-}" ;;
  *) usage ;;
esac
