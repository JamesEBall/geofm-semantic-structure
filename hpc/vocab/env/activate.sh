# shellcheck shell=bash
# Activate the vocab venv (env/venv, layered on fresh-01) with sane job defaults.
# Usage inside Slurm jobs (safe under `set -euo pipefail`):
#   source "$VOCAB_ROOT/env/activate.sh"
# Overrides: VOCAB_THREADS (default $SLURM_CPUS_PER_TASK, else 1); any GDAL_*/HF_* variable
# exported before sourcing is kept.

VOCAB_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VOCAB_ENV_DIR
export VOCAB_ROOT="${VOCAB_ROOT:-$(dirname "$VOCAB_ENV_DIR")}"
# Keep caches and temporary files off the quota-limited home directory.
source "$VOCAB_ENV_DIR/cache_dirs.sh"
export VOCAB_VENV="$VOCAB_ENV_DIR/venv"
if [[ ! -x "$VOCAB_VENV/bin/python" ]]; then
  echo "activate.sh: no venv at $VOCAB_VENV (run: sbatch slurm/env_build.sbatch)" >&2
  return 1 2>/dev/null || exit 1
fi

export VIRTUAL_ENV="$VOCAB_VENV"
case ":${PATH:-}:" in
  *":$VOCAB_VENV/bin:"*) ;;
  *) export PATH="$VOCAB_VENV/bin:${PATH:-/usr/bin:/bin}" ;;
esac
unset PYTHONHOME
export VOCAB_PYTHON="$VOCAB_VENV/bin/python"
export PYTHONNOUSERSITE=1
# Keep fresh-01 read-only: bytecode for every module goes under env/pycache.
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$VOCAB_ENV_DIR/pycache}"
export PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
# C++ osmium-tool from the conda-forge prefix env/osm. Its bin/ is NOT put on PATH (it has
# its own python); call "$OSMIUM_BIN" explicitly.
export VOCAB_OSM_PREFIX="$VOCAB_ENV_DIR/osm"
export OSMIUM_BIN="${OSMIUM_BIN:-$VOCAB_OSM_PREFIX/bin/osmium}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}" HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

_vocab_threads="${VOCAB_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="$_vocab_threads" OPENBLAS_NUM_THREADS="$_vocab_threads"
export MKL_NUM_THREADS="$_vocab_threads" NUMEXPR_NUM_THREADS="$_vocab_threads"
export VECLIB_MAXIMUM_THREADS="$_vocab_threads" RAYON_NUM_THREADS="$_vocab_threads"
unset _vocab_threads

# Cloud-optimised GeoTIFF reads over HTTP (rasterio/GDAL). Proxy variables come from
# lib/tunnel.sh (tunnel_start), not from here.
export GDAL_DISABLE_READDIR_ON_OPEN="${GDAL_DISABLE_READDIR_ON_OPEN:-EMPTY_DIR}"
export CPL_VSIL_CURL_ALLOWED_EXTENSIONS="${CPL_VSIL_CURL_ALLOWED_EXTENSIONS:-.tif,.tiff,.TIF}"
export GDAL_HTTP_MULTIRANGE="${GDAL_HTTP_MULTIRANGE:-YES}"
export GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="${GDAL_HTTP_MERGE_CONSECUTIVE_RANGES:-YES}"
export GDAL_HTTP_MAX_RETRY="${GDAL_HTTP_MAX_RETRY:-5}" GDAL_HTTP_RETRY_DELAY="${GDAL_HTTP_RETRY_DELAY:-1}"
export GDAL_HTTP_TIMEOUT="${GDAL_HTTP_TIMEOUT:-60}" GDAL_HTTP_CONNECTTIMEOUT="${GDAL_HTTP_CONNECTTIMEOUT:-20}"
export VSI_CACHE="${VSI_CACHE:-TRUE}"
return 0 2>/dev/null || true
