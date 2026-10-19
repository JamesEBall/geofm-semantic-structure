#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python_bin="${PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then "$python_bin" -m venv .venv; fi
extras=''
for arg in "$@"; do
  if [[ "$arg" == '--from-source' || "$arg" == '--prepare-only' ]]; then extras='[source]'; fi
done
.venv/bin/python -m pip install --quiet -e ".${extras}"
exec .venv/bin/python -m study.run "$@"
