"""Adopt the live pct10 metadata queue while deferring unadmitted FOREVER.

No scientific runner or model is changed. Existing phase wrappers and GPU
children continue untouched under the original frozen phase implementation.
"""
from pathlib import Path
import importlib.util,json,os

ROOT=Path(os.environ['LLMCL_ROOT']).resolve()
SOURCE=ROOT/'exp/two_backbone_pct10_20261025/dispatch.py'
spec=importlib.util.spec_from_file_location('pct10_frozen_dispatch',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
original_read=module.read
QUEUE=ROOT/'summary/two_backbone_pct10_20261025/queue_old.json'

def reviewed_read(path):
    value=original_read(path)
    if Path(path)==QUEUE:
        for job in value['jobs'].values():
            if job['method']=='FOREVER':
                assert job['status'] in ('waiting','waiting_forever_pct10_scheduler_review'),job
                job['status']='waiting_forever_pct10_scheduler_review'
                job['defer_reason']='10% full dynamic replay scheduler and preflight contract are not admitted; original 1% assumptions must not be bypassed'
        value['FOREVER_deferred_before_first_model']=True
    return value

if __name__=='__main__':
    module.read=reviewed_read
    module.main()
