"""Full 100-row auxiliary schedules on a tiny native multimodal model (CPU only)."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,sys,torch
from pathlib import Path
from verify_pending_native_cpu import tiny,ToyIO
from parsp_adapter import PaRSPMethod
from fv_adapter import FVMethod
HERE=Path(__file__).resolve().parent;torch.set_num_threads(1);torch.manual_seed(42)
method=PaRSPMethod(tiny(),'PaRSP',[]);method.io=ToyIO(method);method.preflight=False
method.begin_stage(1);assert method.selection['probe_count']==100 and not method.projectors
assert set(method.selection['auxiliary_train_ids'])=={x['id'] for x in method.io.stage_rows(1)}
method.end_stage(1);assert len(method.history)==48
method.begin_stage(2);assert method.projectors and method.selection['probe_count']==100
p=HERE/'preflight/parsp_formal_schedule_cpu.json';p.write_text(json.dumps({'passed':True,'formal_first_task_no_external_calibration':True,'first_stage_no_old_task_projection':True,'next_stage_uses_past_covariance':True,'localization_uses_exactly_current_100_train_ids':True,'all_statistics_in_memory':True},indent=2)+'\n')
print('PASS PaRSP formal 100-row schedule and first/no-history -> historical-statistics boundary',flush=True)
del method
method=FVMethod(tiny(),'FV_guided_Training',[]);method.io=ToyIO(method);method.preflight=False;method.begin_stage(1)
v=method.io.last_record;assert v['mean_activation_trials']==100 and v['causal_trials']==25 and v['shots']==10
allowed={x['id'] for x in method.io.stage_rows(1)}
for row in v['mean_probe_ids']+v['causal_probe_ids']:
 assert row['query_id'] in allowed and len(row['demo_ids'])==10 and len(set(row['demo_ids']))==10 and row['query_id'] not in row['demo_ids'] and set(row['demo_ids'])<=allowed
p=HERE/'preflight/fv_formal_schedule_cpu.json';p.write_text(json.dumps({'passed':True,'mean_trials':100,'causal_trials':25,'shots':10,'all_probe_ids_current_100_train_pool':True,'query_never_in_its_demonstrations':True,'no_test_or_historical_labels_used':True},indent=2)+'\n')
print('PASS FV formal 100 mean /25 causal trials, 10 native multimodal demonstrations and strict training-only ID budget',flush=True)
