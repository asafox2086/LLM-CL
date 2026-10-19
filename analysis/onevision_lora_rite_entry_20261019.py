"""Set the physical device before importing the independent worker."""
import argparse
import json
import os
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', choices=['0', '1'], required=True)
parser.add_argument('--phase', choices=['train', 'full'], required=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
try:
    from onevision_lora_rite_trial_20261019 import run_trial
    run_trial(args.phase)
except Exception:
    folder = Path(__file__).resolve().parents[1] / 'summary/onevision_diagnostics_20261016/lora_rite_continuous_20261019/v1'
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(), no_blind_retry=True)) + '\n')
    raise
