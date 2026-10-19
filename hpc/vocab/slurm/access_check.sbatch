#!/bin/bash
#SBATCH --job-name=~🌍megaearth~access🛰️
#SBATCH --account=default
#SBATCH --partition=short
#SBATCH --nice=10000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
# Verifies every external data/service path from a COMPUTE node through the SOCKS tunnel
# and writes env/access-report.json. Also self-tests lib/tunnel.sh (detached ssh, watchdog
# restart on the same port after killing ssh, tunnel_stop cleanup). Submit from vocab-01
# after env_build:  sbatch --dependency=afterok:<env_build_job> slurm/access_check.sbatch
# Exit: 0 ok, 1 required access check failed, 75 tunnel could not be started.
set -euo pipefail

ROOT="${VOCAB_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
export VOCAB_ROOT="$ROOT"
ENV_DIR="$ROOT/env"
if [[ -z "${SLURM_JOB_ID:-}" || "$(hostname -s)" == login* ]]; then
  echo "access_check: submit with sbatch; never run on a login node" >&2
  exit 2
fi
cd "$ROOT"
# shellcheck source=/dev/null
source "$ENV_DIR/activate.sh"
# shellcheck source=/dev/null
source "$ROOT/lib/tunnel.sh"
echo "access_check: host=$(hostname -s) job=$SLURM_JOB_ID cpus=${SLURM_CPUS_PER_TASK:-?} $(ssh -V 2>&1)"
df -h /tmp /dev/shm 2>/dev/null | sed 's/^/  /' || true

export VOCAB_TUNNEL_JITTER="${VOCAB_TUNNEL_JITTER:-5}"
t0=$SECONDS
tunnel_start || { echo "access_check: tunnel_start failed" >&2; exit 75; }
start_s=$(( SECONDS - t0 ))
detached=true
[[ -z "$(jobs -p)" ]] || detached=false
tunnel_check || { echo "access_check: tunnel_check failed" >&2; exit 75; }
echo "access_check: tunnel up in ${start_s}s port=$VOCAB_SOCKS_PORT cipher=$VOCAB_TUNNEL_CIPHER_USED detached=$detached"

# Watchdog self-test: kill ssh, expect a restart on the same port.
pidfile="$VOCAB_TUNNEL_STATE_DIR/ssh.pid"
pid0="$(cat "$pidfile")"
kill "$pid0"
t1=$SECONDS
recovered=false
new_pid=null
for _ in $(seq 1 45); do
  sleep 2
  p="$(cat "$pidfile" 2>/dev/null || true)"
  if [[ -n "$p" && "$p" != "$pid0" ]] && tunnel_check; then
    recovered=true
    new_pid="$p"
    break
  fi
done
recovery_s=$(( SECONDS - t1 ))
echo "access_check: watchdog self-test recovered=$recovered in ${recovery_s}s (pid $pid0 -> $new_pid)"
printf '{"start_s": %s, "cipher": "%s", "ssh_detached_from_job_table": %s, "watchdog_selftest": {"killed_ssh_pid": %s, "recovered": %s, "new_ssh_pid": %s, "recovery_s": %s, "port": %s}}\n' \
  "$start_s" "$VOCAB_TUNNEL_CIPHER_USED" "$detached" "$pid0" "$recovered" "$new_pid" "$recovery_s" "$VOCAB_SOCKS_PORT" \
  >"$ENV_DIR/tunnel-selftest.json"
if [[ "$recovered" != true ]]; then
  tunnel_stop
  tunnel_start || exit 75
fi

rc=0
python -u "$ENV_DIR/check_access.py" --out "$ENV_DIR/access-report.json" \
  --tunnel-selftest "$ENV_DIR/tunnel-selftest.json" || rc=$?

# tunnel_stop self-test: ssh + watchdog gone, port closed.
ssh_pid="$(cat "$pidfile" 2>/dev/null || echo 0)"
wd_pid="${_VT_WD_PID:-0}"
port="$VOCAB_SOCKS_PORT"
tunnel_stop
sleep 1
ssh_gone=true; wd_gone=true; port_closed=true
_vt_alive "$ssh_pid" && ssh_gone=false
_vt_alive "$wd_pid" && wd_gone=false
_vt_port_open "$port" && port_closed=false
echo "access_check: tunnel_stop ssh_gone=$ssh_gone watchdog_gone=$wd_gone port_closed=$port_closed"
printf '{"tunnel_stop_selftest": {"ssh_gone": %s, "watchdog_gone": %s, "port_closed": %s, "proxy_vars_unset": %s}}\n' \
  "$ssh_gone" "$wd_gone" "$port_closed" "$([[ -z "${ALL_PROXY:-}${GDAL_HTTP_PROXY:-}" ]] && echo true || echo false)" \
  >"$ENV_DIR/tunnel-stop-selftest.json"
python "$ENV_DIR/check_access.py" --annotate "$ENV_DIR/access-report.json" \
  --tunnel-selftest "$ENV_DIR/tunnel-stop-selftest.json"
if [[ "$recovered" != true || "$ssh_gone" != true || "$wd_gone" != true || "$port_closed" != true ]]; then
  echo "access_check: tunnel self-test FAILED" >&2
  rc=1
fi
echo "access_check: exit $rc"
exit "$rc"
