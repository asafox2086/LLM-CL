#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
conda env create --prefix "$repo_root/.conda-env" --file exp/environment.yml
"$repo_root/.conda-env/bin/python" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118
"$repo_root/.conda-env/bin/python" -m pip install -r exp/requirements.lock --index-url "${LLMCL_PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
"$repo_root/.conda-env/bin/python" -m pip check
