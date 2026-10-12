#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$repo_root/exp/run_olora.sh" --config "$repo_root/exp/configs/seq_lora_t5large.json" "$@"
