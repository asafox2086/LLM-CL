#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
method="${1:?Usage: bash exp/start_medical_detached.sh METHOD GPU [--resume RUN_DIRECTORY]}"
gpu="${2:?GPU ID required}"
shift 2
case "$method" in
    seq_lora|migu_lora|olora|sapt_lora) ;;
    *) echo "Unsupported method: $method" >&2; exit 2 ;;
esac
cd "$repo_root"
mkdir -p exp/logs
log="exp/logs/medical_${method}_$(date -u +%Y%m%dT%H%M%S%N).log"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
nohup setsid "$repo_root/.conda-env/bin/python" -u exp/medical_continuation.py \
  --config "exp/configs/${method}_t5large_medical2.json" --gpu "$gpu" "$@" > "$log" 2>&1 < /dev/null &
printf '%s\n' "$!" > "$log.pid"
printf 'Supervisor PID: %s\nLog: %s/%s\n' "$!" "$repo_root" "$log"
