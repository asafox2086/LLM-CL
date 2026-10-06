#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
exec "$repo_root/.conda-env/bin/python" -u exp/launch_olora.py "$@"
