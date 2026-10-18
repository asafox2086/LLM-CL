"""Choose the authorized physical GPU before importing the worker."""
import argparse
import os
p=argparse.ArgumentParser()
p.add_argument('--gpu',choices=['0','1'],required=True)
p.add_argument('--phase',choices=['train','full'],required=True)
p.add_argument('--view',choices=['principal_coeff_r16','random_coeff_r16'],required=True)
a=p.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from onevision_directional_coeff_trial import run_trial
run_trial(a.phase,a.view)
