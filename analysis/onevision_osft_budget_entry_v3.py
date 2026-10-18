"""Set the physical device before importing the independent SVD control."""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', choices=['0', '1'], required=True)
parser.add_argument('--phase', choices=['train', 'full'], required=True)
parser.add_argument('--view', choices=['svd_qv_r8_free', 'svd_qv_r8_orthogonal'], required=True)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from onevision_osft_budget_trial_v3 import run_trial
run_trial(args.phase, args.view)
