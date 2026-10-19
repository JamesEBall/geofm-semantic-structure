#!/bin/bash
# Fast path for the planet snapshot: parallel HTTP range downloads on the login node, which has a
# direct internet route (the compute nodes only reach the internet through one SSH tunnel each).
# The author authorised the login node for pure downloads on 16 October 2026; nothing is decoded or
# processed here. Parts are written in exactly the layout the Slurm `osm_fetch` stage expects
# (`parts/part-NNNNN.bin` + `.json` with sha256), so that stage then verifies and concatenates them
# inside Slurm without re-downloading.
#
#   nohup bash slurm/login_fetch_planet.sh fetch [PARALLEL=12] > logs/planet-fetch.log 2>&1 &
#   bash slurm/login_fetch_planet.sh status          # parts present / total, bytes on disk
#   bash slurm/login_fetch_planet.sh verify          # re-hash every part against its .json (slow; 90 GB)
#
# Idempotent and resumable: completed parts (size matches) are skipped; a part whose size differs is
# re-fetched. A mirror that ignores the Range header and answers 200 with the whole 90 GB body is cut
# off by --max-filesize, and its part is discarded.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${MEGAEARTH_ROOT:-/mnt/scratch/users/40021461/geofm-semantic-expansion/vocab-01}"
MODE="${1:-fetch}"
PAR="${2:-12}"
SNAPSHOT="planet-251229.osm.pbf"
BYTES=90603886401
RANGE_MIB="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["osm"]["range_mib"])' "$HERE/config.json")"
SIZE=$((RANGE_MIB*1024*1024))
MIRRORS=(https://ftp.fau.de/osm-planet/pbf https://ftp5.gwdg.de/pub/misc/openstreetmap/planet.openstreetmap.org/pbf https://planet.openstreetmap.org/pbf)
DEST="$ROOT/shared/osm_fetch/parts"
N=$(( (BYTES + SIZE - 1) / SIZE ))
mkdir -p "$DEST"
export DEST SIZE BYTES SNAPSHOT
export MIRRORS_STR="${MIRRORS[*]}"

fetch_part() {
  local i="$1" start stop len name bin meta tmp url ok=0 sum code
  start=$((i*SIZE)); stop=$((start+SIZE-1)); [[ $stop -ge $BYTES ]] && stop=$((BYTES-1)); len=$((stop-start+1))
  name="$(printf 'part-%05d' "$i")"; bin="$DEST/$name.bin"; meta="$DEST/$name.json"; tmp="$bin.part"
  if [[ -f "$bin" && -f "$meta" && "$(stat -c %s "$bin")" -eq "$len" ]]; then return 0; fi
  for m in $MIRRORS_STR; do
    url="$m/$SNAPSHOT"
    rm -f "$tmp"
    # 206 is the only acceptable answer: a 200 means the Range header was ignored.
    code="$(curl -sS --retry 3 --retry-delay 2 --connect-timeout 20 --max-time 1800 \
         --max-filesize $((len+1)) -H 'Accept-Encoding: identity' -r "$start-$stop" \
         -o "$tmp" -w '%{http_code}' "$url" || true)"
    if [[ "$code" == "206" && -f "$tmp" && "$(stat -c %s "$tmp")" -eq "$len" ]]; then ok=1; break; fi
    echo "retry $name via next mirror (http $code)" >&2
    rm -f "$tmp"
  done
  [[ $ok -eq 1 ]] || { echo "FAILED $name" >&2; return 1; }
  sum="$(sha256sum "$tmp" | cut -d' ' -f1)"
  mv -f "$tmp" "$bin"
  printf '{"url":"%s","start":%d,"stop":%d,"bytes":%d,"sha256":"%s","fetched_on":"login-node"}\n' \
     "$url" "$start" "$stop" "$len" "$sum" > "$meta.tmp" && mv -f "$meta.tmp" "$meta"
  echo "done $name"
}
export -f fetch_part

verify_part() {
  local json="$1" bin sum want
  bin="${json%.json}.bin"
  want="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["sha256"])' "$json")"
  sum="$(sha256sum "$bin" | cut -d' ' -f1)"
  if [[ "$sum" != "$want" ]]; then echo "CORRUPT $(basename "$bin")" >&2; rm -f "$bin" "$json"; return 1; fi
}
export -f verify_part

case "$MODE" in
  fetch)
    echo "planet: $N parts of $RANGE_MIB MiB, $PAR parallel streams, into $DEST"
    rc=0
    seq 0 $((N-1)) | xargs -P "$PAR" -I{} bash -c 'fetch_part {}' || rc=$?
    have=$(ls "$DEST"/part-*.json 2>/dev/null | wc -l)
    echo "parts with metadata: $have / $N"
    if [[ "$have" -eq "$N" ]]; then echo "all parts present; the Slurm osm_fetch stage will verify, concatenate and MD5-check"; exit 0; fi
    echo "incomplete ($rc); rerun to resume" >&2; exit 1 ;;
  status)
    have=$(ls "$DEST"/part-*.json 2>/dev/null | wc -l)
    echo "parts: $have / $N"; du -sh "$DEST" 2>/dev/null ;;
  verify)
    rc=0
    ls "$DEST"/part-*.json | xargs -P "$PAR" -I{} bash -c 'verify_part {}' || rc=$?
    [[ $rc -eq 0 ]] && echo "all parts verified" || { echo "corrupt parts removed; rerun fetch" >&2; exit 1; } ;;
  *) echo "usage: $0 fetch|status|verify [PARALLEL]" >&2; exit 2 ;;
esac
