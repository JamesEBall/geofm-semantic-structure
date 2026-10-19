# shellcheck shell=bash
# SOCKS5 tunnel from an ENUCC compute node through the login node (compute nodes have no
# direct internet). Source from a Slurm job (safe under `set -euo pipefail`):
#
#   source "$VOCAB_ROOT/lib/tunnel.sh"
#   tunnel_start || exit 75   # ssh -N -D, exports proxy env, starts a watchdog, EXIT trap
#   tunnel_check || exit 75   # PC STAC reachable through the tunnel
#   ...                       # curl/requests/GDAL now go through $VOCAB_PROXY
#   tunnel_stop               # optional: also run by the EXIT trap (unless VOCAB_TUNNEL_NO_TRAP=1)
#
# Exports after tunnel_start: VOCAB_SOCKS_PORT, VOCAB_PROXY=socks5h://127.0.0.1:PORT,
# ALL_PROXY/all_proxy/HTTPS_PROXY/https_proxy/HTTP_PROXY/http_proxy (= VOCAB_PROXY),
# NO_PROXY/no_proxy, GDAL_HTTP_PROXY/GDAL_HTTPS_PROXY (= VOCAB_PROXY),
# VOCAB_TUNNEL_CIPHER_USED, VOCAB_TUNNEL_STATE_DIR (holds ssh.pid).
#
# Tunables: VOCAB_LOGIN_HOST (10.10.0.21), VOCAB_TUNNEL_KEY (~/.ssh/id_alcescluster),
# VOCAB_TUNNEL_JITTER (15 s, random startup/restart delay), VOCAB_TUNNEL_TIMEOUT (60 s),
# VOCAB_TUNNEL_CIPHER (aes128-gcm@openssh.com; empty = ssh default), VOCAB_TUNNEL_PORT
# (fixed port; default a free port), VOCAB_TUNNEL_INTERVAL (10 s watchdog period),
# VOCAB_TUNNEL_MAX_RESTARTS (20), VOCAB_TUNNEL_STATE_DIR, VOCAB_TUNNEL_NO_TRAP.
#
# ssh and the watchdog are detached from the caller's job table, so a bare `wait` in the
# caller does not block on them. Slurm still kills them at job end (same cgroup).

_vt_log() {
  printf '%s tunnel[%s:%s]: %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${HOSTNAME%%.*}" "${BASHPID:-$$}" "$*" >&2
}

_vt_pick_port() {
  local py
  for py in "${VOCAB_TUNNEL_PYTHON:-}" /usr/bin/python3.11 /usr/bin/python3 python3; do
    if [[ -n "$py" ]] && command -v "$py" >/dev/null 2>&1; then
      if "$py" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()' 2>/dev/null; then
        return 0
      fi
    fi
  done
  echo $(( 20000 + (RANDOM * 32768 + RANDOM) % 40000 ))
}

_vt_port_open() {
  timeout 3 bash -c 'exec 3<>"/dev/tcp/127.0.0.1/$1"' _ "$1" >/dev/null 2>&1
}

# Alive = exists and is not a zombie (orphans may be re-parented to a non-reaping process).
_vt_alive() {
  local st=""
  if [[ -z "${1:-}" ]] || ! kill -0 "$1" 2>/dev/null; then
    return 1
  fi
  st="$(awk '{print $3}' "/proc/$1/stat" 2>/dev/null || true)"
  [[ "$st" != Z && "$st" != X ]]
}

_vt_kill() {
  local p
  for p in "$@"; do
    if [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; then
      kill -TERM "$p" 2>/dev/null || true
    fi
  done
}

# Interruptible sleep: a trapped signal ends `wait` immediately.
_vt_sleep() {
  sleep "$1" &
  _VT_SLEEP_PID=$!
  wait "$_VT_SLEEP_PID" 2>/dev/null || true
}

# Start ssh -D on port $1 with cipher $2 ('' = ssh default); sets _VT_SSH_PID.
_vt_spawn() {
  local port=$1 cipher=$2
  local -a cmd=(ssh -N -n -D "127.0.0.1:${port}"
    -i "${VOCAB_TUNNEL_KEY:-$HOME/.ssh/id_alcescluster}"
    -o BatchMode=yes -o ExitOnForwardFailure=yes
    -o ServerAliveInterval=15 -o ServerAliveCountMax=4
    -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20
    -o ControlMaster=no -o ControlPath=none -o Compression=no
    -o IdentitiesOnly=yes -o LogLevel=ERROR)
  if [[ -n "$cipher" ]]; then
    cmd+=(-c "$cipher")
  fi
  cmd+=("${VOCAB_LOGIN_HOST:-10.10.0.21}")
  # The command substitution's subshell exits immediately, detaching ssh from our job table.
  _VT_SSH_PID="$("${cmd[@]}" </dev/null >/dev/null & echo $!)"
}

# 0 = port accepts connections, 1 = timeout, 2 = ssh exited.
_vt_wait_ready() {
  local port=$1 pid=$2 deadline=$(( SECONDS + $3 ))
  while (( SECONDS < deadline )); do
    if ! _vt_alive "$pid"; then
      return 2
    fi
    if _vt_port_open "$port"; then
      return 0
    fi
    sleep 0.3
  done
  return 1
}

# Bring the tunnel up on port $1, trying the preferred cipher first and then the ssh
# default (a rejected cipher makes ssh exit). Sets _VT_SSH_PID and _VT_CIPHER_USED.
_vt_connect() {
  local port=$1 cipher rc
  local -a ciphers=()
  if [[ -n "${_VT_CIPHER:-}" ]]; then
    ciphers+=("$_VT_CIPHER")
  fi
  ciphers+=("")
  for cipher in "${ciphers[@]}"; do
    _vt_spawn "$port" "$cipher"
    rc=0
    _vt_wait_ready "$port" "$_VT_SSH_PID" "${VOCAB_TUNNEL_TIMEOUT:-60}" || rc=$?
    if [[ "$rc" -eq 0 ]]; then
      _VT_CIPHER_USED="${cipher:-ssh-default}"
      if [[ -z "$cipher" && -n "${_VT_CIPHER:-}" ]]; then
        _VT_CIPHER=""
      fi
      return 0
    fi
    _vt_kill "$_VT_SSH_PID"
    if [[ "$rc" -eq 1 ]]; then
      return 1
    fi
  done
  return 1
}

_vt_watchdog() {
  local port=$1 parent=$2 pid="$_VT_SSH_PID" restarts=0
  local max="${VOCAB_TUNNEL_MAX_RESTARTS:-20}" interval="${VOCAB_TUNNEL_INTERVAL:-10}"
  local jitter="${VOCAB_TUNNEL_JITTER:-15}" pidfile="$VOCAB_TUNNEL_STATE_DIR/ssh.pid"
  [[ "$jitter" =~ ^[0-9]+$ ]] || jitter=15
  _VT_SLEEP_PID=""
  trap '_vt_kill "$pid" "$_VT_SSH_PID" "$_VT_SLEEP_PID"; exit 0' TERM INT HUP
  while true; do
    _vt_sleep "$interval"
    if ! kill -0 "$parent" 2>/dev/null || [[ -e "$VOCAB_TUNNEL_STATE_DIR/stop" ]]; then
      _vt_kill "$pid"
      exit 0
    fi
    if _vt_alive "$pid"; then
      if _vt_port_open "$port"; then
        continue
      fi
      _vt_sleep 2
      if _vt_alive "$pid" && _vt_port_open "$port"; then
        continue
      fi
    fi
    if (( restarts >= max )); then
      _vt_log "port $port: tunnel down and $max restarts used; giving up"
      _vt_kill "$pid"
      exit 1
    fi
    restarts=$(( restarts + 1 ))
    _vt_log "port $port: tunnel down (ssh pid $pid); restart $restarts/$max"
    _vt_kill "$pid"
    if (( jitter > 0 )); then
      _vt_sleep "$(( RANDOM % (jitter + 1) ))"
    fi
    if _vt_connect "$port"; then
      _vt_log "port $port: restarted (ssh pid $_VT_SSH_PID, cipher $_VT_CIPHER_USED)"
    else
      _vt_log "port $port: restart $restarts failed"
    fi
    pid="$_VT_SSH_PID"
    echo "$pid" >"$pidfile" 2>/dev/null
  done
}

tunnel_start() {
  if [[ -n "${_VT_SSH_PID:-}" && -n "${VOCAB_SOCKS_PORT:-}" ]] && _vt_alive "$_VT_SSH_PID"; then
    _vt_log "already running on port $VOCAB_SOCKS_PORT"
    return 0
  fi
  local jitter="${VOCAB_TUNNEL_JITTER:-15}" attempt port="" avail parent="${BASHPID:-$$}"
  [[ "$jitter" =~ ^[0-9]+$ ]] || jitter=15
  if (( jitter > 0 )); then
    sleep "$(( RANDOM % (jitter + 1) ))"
  fi
  _VT_CIPHER="${VOCAB_TUNNEL_CIPHER-aes128-gcm@openssh.com}"
  if [[ -n "$_VT_CIPHER" ]]; then
    avail="$(ssh -Q cipher 2>/dev/null || true)"
    if ! grep -qxF -- "$_VT_CIPHER" <<<"$avail"; then
      _VT_CIPHER=""
    fi
  fi
  for attempt in 1 2 3 4 5; do
    port="${VOCAB_TUNNEL_PORT:-}"
    if [[ -z "$port" ]]; then
      port="$(_vt_pick_port)"
    fi
    if _vt_connect "$port"; then
      break
    fi
    _vt_log "attempt $attempt on port $port failed"
    port=""
    sleep "$(( attempt * 2 + RANDOM % 5 ))"
  done
  if [[ -z "$port" ]]; then
    _vt_log "could not start the tunnel to ${VOCAB_LOGIN_HOST:-10.10.0.21}"
    return 1
  fi

  export VOCAB_SOCKS_PORT="$port"
  export VOCAB_PROXY="socks5h://127.0.0.1:${port}"
  export ALL_PROXY="$VOCAB_PROXY" all_proxy="$VOCAB_PROXY"
  export HTTPS_PROXY="$VOCAB_PROXY" https_proxy="$VOCAB_PROXY"
  export HTTP_PROXY="$VOCAB_PROXY" http_proxy="$VOCAB_PROXY"
  export NO_PROXY="localhost,127.0.0.1,10.0.0.0/8" no_proxy="localhost,127.0.0.1,10.0.0.0/8"
  export GDAL_HTTP_PROXY="$VOCAB_PROXY" GDAL_HTTPS_PROXY="$VOCAB_PROXY"
  export VOCAB_TUNNEL_CIPHER_USED="$_VT_CIPHER_USED"

  local root="${VOCAB_ROOT:-/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01}"
  export VOCAB_TUNNEL_STATE_DIR="${VOCAB_TUNNEL_STATE_DIR:-$root/run/tunnel/${SLURM_JOB_ID:-nojob}.${SLURM_ARRAY_TASK_ID:-0}.${parent}}"
  mkdir -p "$VOCAB_TUNNEL_STATE_DIR"
  rm -f "$VOCAB_TUNNEL_STATE_DIR/stop"
  echo "$_VT_SSH_PID" >"$VOCAB_TUNNEL_STATE_DIR/ssh.pid"
  _VT_WD_PID="$( (set +e +u +o pipefail; trap - EXIT; _vt_watchdog "$port" "$parent") </dev/null >/dev/null & echo $!)"
  _vt_log "up on 127.0.0.1:$port via ${VOCAB_LOGIN_HOST:-10.10.0.21} (ssh pid $_VT_SSH_PID, cipher $_VT_CIPHER_USED, watchdog $_VT_WD_PID)"

  if [[ "${VOCAB_TUNNEL_NO_TRAP:-0}" != 1 ]]; then
    local prev
    prev="$(trap -p EXIT)"
    if [[ -z "$prev" ]]; then
      trap tunnel_stop EXIT
    elif [[ "$prev" != *tunnel_stop* ]]; then
      eval "set -- $prev"   # -> trap -- '<previous command>' EXIT
      # shellcheck disable=SC2064  # expand the previous handler now, on purpose
      trap "tunnel_stop; $3" EXIT
    fi
  fi
  return 0
}

tunnel_check() {
  local port="${VOCAB_SOCKS_PORT:-}"
  if [[ -z "$port" ]]; then
    return 1
  fi
  curl -s -f -m 10 -o /dev/null --socks5-hostname "127.0.0.1:${port}" \
    "${VOCAB_TUNNEL_CHECK_URL:-https://planetarycomputer.microsoft.com/api/stac/v1}"
}

tunnel_stop() {
  local dir="${VOCAB_TUNNEL_STATE_DIR:-}" p="" _
  if [[ -n "$dir" && -d "$dir" ]]; then
    : >"$dir/stop" 2>/dev/null || true
  fi
  if [[ -n "${_VT_WD_PID:-}" ]]; then
    _vt_kill "$_VT_WD_PID"
    for _ in $(seq 1 50); do
      if ! _vt_alive "$_VT_WD_PID"; then
        break
      fi
      sleep 0.1
    done
  fi
  if [[ -n "$dir" && -f "$dir/ssh.pid" ]]; then
    p="$(cat "$dir/ssh.pid" 2>/dev/null || true)"
  fi
  _vt_kill "${_VT_SSH_PID:-}" "$p"
  if [[ -n "$dir" && -d "$dir" ]]; then
    rm -f "$dir/ssh.pid" "$dir/stop"
    rmdir "$dir" 2>/dev/null || true
  fi
  if [[ -n "${VOCAB_SOCKS_PORT:-}" ]]; then
    _vt_log "stopped (port $VOCAB_SOCKS_PORT)"
  fi
  unset _VT_WD_PID _VT_SSH_PID VOCAB_SOCKS_PORT VOCAB_PROXY VOCAB_TUNNEL_STATE_DIR
  unset ALL_PROXY all_proxy HTTPS_PROXY https_proxy HTTP_PROXY http_proxy
  unset GDAL_HTTP_PROXY GDAL_HTTPS_PROXY
  return 0
}
