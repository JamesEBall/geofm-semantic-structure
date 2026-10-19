#!/bin/bash
#SBATCH --job-name=🌍🤖envbuild
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nice=10000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/env-build-%j.out
# Builds the vocab environments. Submit from vocab-01 after staging hpc/vocab/{env,lib,slurm}
# and running `bash env/fetch_wheels.sh` on the login node:
#   cd vocab-01 && sbatch slurm/env_build.sbatch [venv] [osm] [smoke] [verify]   (default: all three)
#   venv : env/venv, OFFLINE from env/wheelhouse, layered on fresh-01 via a .pth file,
#          then env/verify_env.py (imports, versions, manifest.json, lock.txt, hashes)
#   osm  : env/osm conda-forge prefix (osmium-tool, pyosmium) via micromamba + SOCKS tunnel
#   smoke: env/osmium_smoke.py on smoke/raw/luxembourg-latest.osm.pbf -> env/osmium-smoke.json
set -euo pipefail

ROOT="${VOCAB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export VOCAB_ROOT="$ROOT"
ENV_DIR="$ROOT/env"
FRESH="${VOCAB_FRESH_VENV:-/mnt/scratch/users/40021461/geofm-semantic-expansion/fresh-01/.venv}"
VENV="$ENV_DIR/venv"
WH="$ENV_DIR/wheelhouse"
PARTS="${*:-venv osm smoke}"

if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
  echo "env_build: submit with sbatch; never run on a login node" >&2
  exit 2
fi
for f in "$ENV_DIR/requirements.in" "$ENV_DIR/verify_env.py" "$ENV_DIR/activate.sh" "$ROOT/lib/tunnel.sh" "$FRESH/pyvenv.cfg"; do
  [[ -f "$f" ]] || { echo "env_build: missing $f" >&2; exit 2; }
done
source "$ENV_DIR/cache_dirs.sh"
echo "env_build: host=$(hostname -s) job=$SLURM_JOB_ID parts=[$PARTS] root=$ROOT"

if [[ " $PARTS " == *" venv "* ]]; then
  [[ -f "$ENV_DIR/constraints-fresh01.txt" && -n "$(ls -A "$WH" 2>/dev/null)" ]] || {
    echo "env_build: empty wheelhouse; run 'bash env/fetch_wheels.sh' on the login node" >&2; exit 2; }
  BASE_PY="$(sed -n 's/^executable *= *//p' "$FRESH/pyvenv.cfg")"
  PYVER="$(sed -n 's/^version *= *//p' "$FRESH/pyvenv.cfg")"
  FRESH_SP="$FRESH/lib/python${PYVER%.*}/site-packages"
  echo "base interpreter: $BASE_PY ($("$BASE_PY" -c 'import sys; print(sys.version.split()[0])'))"
  if [[ -f "$ENV_DIR/wheels.sha256" ]]; then
    (cd "$WH" && sha256sum -c --quiet "$ENV_DIR/wheels.sha256") || {
      echo "env_build: wheelhouse does not match env/wheels.sha256 (pins changed? delete it to re-lock)" >&2; exit 1; }
    echo "wheelhouse matches wheels.sha256"
  fi

  export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_INPUT=1
  export PYTHONPYCACHEPREFIX="$ENV_DIR/pycache"
  unset PYTHONPATH PYTHONHOME PIP_INDEX_URL PIP_EXTRA_INDEX_URL PIP_FIND_LINKS
  case "$VENV" in */env/venv) rm -rf "$VENV" ;; *) exit 2 ;; esac
  "$BASE_PY" -m venv "$VENV"
  SP="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
  printf '%s\n' "$FRESH_SP" >"$SP/_vocab_fresh01.pth"
  "$VENV/bin/python" -m pip install --no-index --find-links "$WH" --only-binary=:all: \
    -c "$ENV_DIR/constraints-fresh01.txt" -r "$ENV_DIR/requirements.in"
  "$VENV/bin/python" -m pip check
  # shellcheck source=/dev/null
  source "$ENV_DIR/activate.sh"
  python -u "$ENV_DIR/verify_env.py" --env-dir "$ENV_DIR" --fresh-venv "$FRESH" --wheelhouse "$WH"
fi

if [[ " $PARTS " == *" osm "* ]]; then
  # shellcheck source=/dev/null
  source "$ROOT/lib/tunnel.sh"
  VOCAB_TUNNEL_JITTER=0 tunnel_start || { echo "env_build: tunnel_start failed" >&2; exit 75; }
  tunnel_check || { echo "env_build: tunnel_check failed" >&2; exit 75; }
  bash "$ENV_DIR/build_osm_env.sh"
  tunnel_stop
fi

if [[ " $PARTS " == *" smoke "* ]]; then
  # shellcheck source=/dev/null
  source "$ENV_DIR/activate.sh"
  python -u "$ENV_DIR/osmium_smoke.py" --pbf "$ROOT/smoke/raw/luxembourg-latest.osm.pbf" \
    --osmium "${OSMIUM_BIN:-}" --workdir "$ROOT/smoke/osmium-selftest" --out "$ENV_DIR/osmium-smoke.json"
fi
if [[ " $PARTS " == *" verify "* ]]; then
  source "$ENV_DIR/activate.sh"
  REPORT_DIR="$ENV_DIR/records/validation-$SLURM_JOB_ID"
  mkdir -p "$REPORT_DIR"
  bash "$ENV_DIR/test_cache_dirs.sh"
  python -m pip check
  python -u "$ENV_DIR/verify_env.py" --env-dir "$REPORT_DIR" --fresh-venv "$FRESH" --wheelhouse "$WH"
fi
echo "env_build: done"
