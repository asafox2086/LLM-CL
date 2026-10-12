#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:?Usage: bash exp/start_detached.sh METHOD GPU [experiment arguments]}"
gpu="${2:?GPU ID required}"
shift 2
case "$method" in
    olora|migu_lora|sapt_lora|seq_lora) ;;
    *) echo "Unsupported method: $method" >&2; exit 2 ;;
esac
cd "$repo_root"
mkdir -p exp/logs
stamp="$(date -u +%Y%m%dT%H%M%S%N)"
log="exp/logs/${method}_${stamp}.log"
LLMCL_FOREGROUND=1 nohup setsid bash "exp/run_${method}.sh" --gpus "$gpu" "$@" > "$log" 2>&1 < /dev/null &
printf 'Supervisor PID: %s\nLog: %s/%s\n' "$!" "$repo_root" "$log"
