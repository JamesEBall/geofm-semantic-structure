# shellcheck shell=bash
# Home-quota guard: keep every cache/config/temp dir of the vocab tooling OFF /users/$USER
# (50 GB soft / 75 GB hard, 150k files soft). Sourced by activate.sh, build_osm_env.sh,
# fetch_wheels.sh and slurm/env_build.sbatch; safe under `set -euo pipefail`; no venv needed.
#   persistent caches -> $VOCAB_CACHE_DIR (default $VOCAB_ROOT/cache, Lustre scratch, no quota)
#   TMPDIR/CPL_TMPDIR -> node-local /tmp/vocab-$SLURM_JOB_ID inside Slurm jobs (removed at exit
#                        by the shell that created it; VOCAB_TMP_NO_TRAP=1 disables), and
#                        $VOCAB_CACHE_DIR/tmp on the login node.
# A variable that is already set is kept unless it resolves under $HOME. VOCAB_TMPDIR forces
# the temp dir (never auto-removed). TRANSFORMERS_CACHE is deliberately NOT set: transformers
# 4.57 emits a FutureWarning on every import when it is; it defaults to HF_HUB_CACHE anyway.

_vocab_cd_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VOCAB_ROOT="${VOCAB_ROOT:-$(dirname "$_vocab_cd_here")}"
_vocab_home="$(realpath -m "${HOME:-/nonexistent}" 2>/dev/null || printf '%s' "${HOME:-/nonexistent}")"

# 0 if $1 is empty or resolves to $HOME or below it.
_vocab_under_home() {
  local p
  [[ -n "${1:-}" ]] || return 0
  p="$(realpath -m "$1" 2>/dev/null || printf '%s' "$1")"
  [[ "$p" == "$_vocab_home" || "$p" == "$_vocab_home"/* ]]
}
# Print $1 if set and off-home, else $2.
_vocab_pick() {
  if [[ -n "${1:-}" ]] && ! _vocab_under_home "$1"; then printf '%s' "$1"; else printf '%s' "$2"; fi
}

VOCAB_CACHE_DIR="$(_vocab_pick "${VOCAB_CACHE_DIR:-}" "$VOCAB_ROOT/cache")"
_c="$VOCAB_CACHE_DIR"
export VOCAB_CACHE_DIR
PIP_CACHE_DIR="$(_vocab_pick "${PIP_CACHE_DIR:-}" "$_c/pip")"
UV_CACHE_DIR="$(_vocab_pick "${UV_CACHE_DIR:-}" "$_c/uv")"
HF_HOME="$(_vocab_pick "${HF_HOME:-}" "$_c/huggingface")"
HF_HUB_CACHE="$(_vocab_pick "${HF_HUB_CACHE:-}" "$HF_HOME/hub")"
HF_XET_CACHE="$(_vocab_pick "${HF_XET_CACHE:-}" "$HF_HOME/xet")"
HF_ASSETS_CACHE="$(_vocab_pick "${HF_ASSETS_CACHE:-}" "$HF_HOME/assets")"
HF_DATASETS_CACHE="$(_vocab_pick "${HF_DATASETS_CACHE:-}" "$HF_HOME/datasets")"
TORCH_HOME="$(_vocab_pick "${TORCH_HOME:-}" "$_c/torch")"
XDG_CACHE_HOME="$(_vocab_pick "${XDG_CACHE_HOME:-}" "$_c/xdg/cache")"
XDG_CONFIG_HOME="$(_vocab_pick "${XDG_CONFIG_HOME:-}" "$_c/xdg/config")"
XDG_DATA_HOME="$(_vocab_pick "${XDG_DATA_HOME:-}" "$_c/xdg/data")"
XDG_STATE_HOME="$(_vocab_pick "${XDG_STATE_HOME:-}" "$_c/xdg/state")"
MPLCONFIGDIR="$(_vocab_pick "${MPLCONFIGDIR:-}" "$_c/matplotlib")"
NUMBA_CACHE_DIR="$(_vocab_pick "${NUMBA_CACHE_DIR:-}" "$_c/numba")"
MAMBA_ROOT_PREFIX="$(_vocab_pick "${MAMBA_ROOT_PREFIX:-}" "$_c/mamba")"
CONDA_PKGS_DIRS="$(_vocab_pick "${CONDA_PKGS_DIRS:-}" "$MAMBA_ROOT_PREFIX/pkgs")"
PROJ_USER_WRITABLE_DIRECTORY="$(_vocab_pick "${PROJ_USER_WRITABLE_DIRECTORY:-}" "$_c/proj")"
HYPOTHESIS_STORAGE_DIRECTORY="$(_vocab_pick "${HYPOTHESIS_STORAGE_DIRECTORY:-}" "$_c/hypothesis")"
CUDA_CACHE_PATH="$(_vocab_pick "${CUDA_CACHE_PATH:-}" "$_c/nv-compute-cache")"
TRITON_CACHE_DIR="$(_vocab_pick "${TRITON_CACHE_DIR:-}" "$_c/triton")"
PYTHONPYCACHEPREFIX="$(_vocab_pick "${PYTHONPYCACHEPREFIX:-}" "$VOCAB_ROOT/env/pycache")"
export PIP_CACHE_DIR UV_CACHE_DIR HF_HOME HF_HUB_CACHE HF_XET_CACHE HF_ASSETS_CACHE \
  HF_DATASETS_CACHE TORCH_HOME XDG_CACHE_HOME XDG_CONFIG_HOME XDG_DATA_HOME XDG_STATE_HOME \
  MPLCONFIGDIR NUMBA_CACHE_DIR MAMBA_ROOT_PREFIX CONDA_PKGS_DIRS PROJ_USER_WRITABLE_DIRECTORY \
  HYPOTHESIS_STORAGE_DIRECTORY CUDA_CACHE_PATH TRITON_CACHE_DIR PYTHONPYCACHEPREFIX
mkdir -p "$PIP_CACHE_DIR" "$UV_CACHE_DIR" "$HF_HUB_CACHE" "$TORCH_HOME" "$XDG_CACHE_HOME" \
  "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" \
  "$CONDA_PKGS_DIRS" "$PROJ_USER_WRITABLE_DIRECTORY" "$HYPOTHESIS_STORAGE_DIRECTORY" \
  "$PYTHONPYCACHEPREFIX" "$_c/tmp"

# --- temp dir: node-local inside jobs, scratch on the login node --------------------------
_vocab_in_job=0
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  case "$(hostname -s 2>/dev/null || true)" in login*) ;; *) _vocab_in_job=1 ;; esac
fi
_vocab_tmp_new=""
if [[ -n "${VOCAB_TMPDIR:-}" ]] && ! _vocab_under_home "$VOCAB_TMPDIR"; then
  mkdir -p "$VOCAB_TMPDIR"   # caller-provided or inherited from a parent: keep, never remove
elif [[ "$_vocab_in_job" == 1 ]]; then
  VOCAB_TMPDIR="${VOCAB_LOCAL_TMP_BASE:-/tmp}/vocab-$SLURM_JOB_ID"
  if mkdir -m 700 "$VOCAB_TMPDIR" 2>/dev/null; then
    _vocab_tmp_new="$VOCAB_TMPDIR"
  elif [[ ! -d "$VOCAB_TMPDIR" || ! -w "$VOCAB_TMPDIR" ]]; then
    VOCAB_TMPDIR="$_c/tmp/job-$SLURM_JOB_ID"
    mkdir -p "$VOCAB_TMPDIR"
  fi
else
  VOCAB_TMPDIR="$_c/tmp"
fi
export VOCAB_TMPDIR TMPDIR="$VOCAB_TMPDIR"
CPL_TMPDIR="$(_vocab_pick "${CPL_TMPDIR:-}" "$VOCAB_TMPDIR")"
export CPL_TMPDIR

# Remove the node-local temp dir this shell created (only /…/vocab-<jobid> paths).
vocab_tmp_cleanup() {
  local d="${_VOCAB_TMP_OWNED:-}"
  case "$d" in */vocab-[0-9]*) [[ -d "$d" ]] && rm -rf -- "$d" ;; esac
  _VOCAB_TMP_OWNED=""
  return 0
}
if [[ -n "$_vocab_tmp_new" ]]; then
  _VOCAB_TMP_OWNED="$_vocab_tmp_new"
  if [[ "${VOCAB_TMP_NO_TRAP:-0}" != 1 ]]; then
    _vocab_prev_trap="$(trap -p EXIT)"
    if [[ -z "$_vocab_prev_trap" ]]; then
      trap vocab_tmp_cleanup EXIT
    else
      _vocab_prev_cmd="$(eval "set -- $_vocab_prev_trap"; printf '%s' "$3")"
      # Decode in a subshell so sourcing this file preserves caller arguments.
      # shellcheck disable=SC2064
      trap "$_vocab_prev_cmd; vocab_tmp_cleanup" EXIT
      unset _vocab_prev_cmd
    fi
    unset _vocab_prev_trap
  fi
fi
unset _vocab_cd_here _c _vocab_in_job _vocab_tmp_new
return 0 2>/dev/null || true
