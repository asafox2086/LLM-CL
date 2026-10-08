#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
if [[ "${LLMCL_FOREGROUND:-0}" == 1 ]]; then
    exec "$repo_root/.conda-env/bin/python" -u exp/launch_olora.py "$@"
fi
mkdir -p exp/logs
log="$repo_root/exp/logs/launch_$(date -u +%Y%m%dT%H%M%S%N).log"
nohup setsid "$repo_root/.conda-env/bin/python" -u exp/launch_olora.py "$@" > "$log" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$log.pid"
printf 'Supervisor PID: %s\nLog: %s\n' "$pid" "$log"
