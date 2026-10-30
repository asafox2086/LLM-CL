"""Read final fixed-exit pilot artifacts only. No model, CP, QA text or score.
The original controller/slot and both exact workers must be fully closed first.
"""
from pathlib import Path
import argparse,hashlib,json,math,os,sys,time
METHODS=('PASs_MoE','CL_MoE');ARMS=('A','B','C','D');EPS=sys.float_info.epsilon

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def bad_constant(x):raise ValueError('nonfinite JSON literal: '+x)
def read(p):return json.loads(Path(p).read_bytes(),parse_constant=bad_constant)
def finite(x):
 if type(x) not in (int,float) or not math.isfinite(x):raise ValueError('finite numeric field required')
 return float(x)
def mean(xs):
 xs=[finite(v) for v in xs]
 if not xs:raise ValueError('empty mean')
 return math.fsum(xs)/len(xs)
def closed_mean(actual,values):
 expected=mean(values);scale=64*EPS*(mean([abs(finite(v)) for v in values])+abs(expected))
 assert abs(finite(actual)-expected)<=scale,(actual,expected,scale)
def gone(who):
 p=Path('/proc')/str(who['pid'])/'stat'
 try:return p.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])
 except FileNotFoundError:return True

def terminal_inputs(binding):
 b=read(binding);D=Path(b['admission_path']).parent;O=D/'CONTROLLER_V1';root=Path(b['output_root'])
 assert b['schema']=='residual-exit-pilot-v1' and b['admitted'] is True and b['block_index']==11
 admission=read(b['admission_path']);assert admission['approved'] is True and admission['binding_sha256']==sha(binding)
 assert sha(D/'controller.py')==admission['controller_sha256']
 for name,digest in b['sources'].items():assert sha(Path(b['source_dir'])/name)==digest,name
 assert sha(b['protocol_path'])==b['protocol_sha256']
 if (O/'FAILURE.json').exists():return dict(status='FAILED',raw_rows_parsed=False),None
 if not (O/'COMPLETE.json').exists() or not (O/'SLOT.json').exists():return dict(status='WAITING',raw_rows_parsed=False),None
 complete=read(O/'COMPLETE.json');slot=read(O/'SLOT.json')
 if slot.get('released') is not True:return dict(status='WAITING',raw_rows_parsed=False),None
 assert complete['complete'] is True and complete['released'] is True
 assert complete['binding_sha256']==slot['binding_sha256']==sha(binding)
 assert slot['status']=='complete' and Path(slot['terminal_path'])==O/'COMPLETE.json' and slot['terminal_sha256']==sha(O/'COMPLETE.json')
 assert complete['gpu_uuid']==slot['gpu_uuid']==b['gpu_uuid'] and complete['GPU']==slot['GPU']==3
 if not gone(slot['identity']):return dict(status='WAITING',raw_rows_parsed=False),None
 phases=complete['phases'];assert len(phases)==2 and [p['method'] for p in phases]==list(METHODS)
 # Check every terminal before opening any method RESULT/FIT/READOUT.
 for phase in phases:
  method=phase['method'];run=root/method
  if not all(p.exists() for p in [run/'COMPLETE.json',run/'RESULT.json',O/(method+'_EXIT.json')]):return dict(status='WAITING',raw_rows_parsed=False),None
  exit_=read(O/(method+'_EXIT.json'));assert exit_==phase and exit_['returncode']==0 and exit_['worker_gone'] is True
  if not gone(exit_['identity']):return dict(status='WAITING',raw_rows_parsed=False),None
  c=read(run/'COMPLETE.json');assert c['identity']==exit_['identity'] and c['binding_sha256']==sha(binding)
  assert not (run/'FAILURE.json').exists() and finite(c['seconds'])<=7200
  assert c['RESULT_sha256']==sha(run/'RESULT.json')
 A=Path(b['asset_root']);md_path=A/'DATA_SELECTION_METADATA.json';assert sha(md_path)==b['data_metadata_sha256'];md=read(md_path)
 expected=md['readout'];assert len(expected)==20 and len({x['id_sha256'] for x in expected})==20
 domains={x['domain'] for x in expected};assert len(domains)==5 and all(sum(x['domain']==d for x in expected)==4 for d in domains)
 inputs={str(binding):sha(binding),str(D/'controller.py'):sha(D/'controller.py'),str(b['admission_path']):sha(b['admission_path']),str(b['protocol_path']):sha(b['protocol_path']),str(O/'COMPLETE.json'):sha(O/'COMPLETE.json'),str(O/'SLOT.json'):sha(O/'SLOT.json'),str(md_path):sha(md_path)}
 for n,h in b['sources'].items():inputs[str(Path(b['source_dir'])/n)]=h
 results={};fits={}
 for method in METHODS:
  run=root/method;r=read(run/'RESULT.json');fit=read(run/'FIT.json');loaded=read(run/'MODEL_LOADED.json')
  assert r['method']==method and r['schema']=='residual-exit-pilot-v1' and r['block_index']==11
  assert r['teacher_unchanged'] and r['student_only_existing_down_weight_changed'] and r['weights_restored_to_A']
  assert r['not_full_24block_validation'] and r['not_quality_verification'] and r['final_added_parameters']==0
  cnt=r['counts'];assert cnt['checkpoint_loads']==1 and cnt['teacher_main_forward']==100 and cnt['student_calibration_forward']==cnt['student_readout_forward']==80
  assert type(cnt['teacher_feature_forward']) is int and cnt['teacher_feature_forward']>=0
  if method=='PASs_MoE':assert cnt['teacher_feature_forward']==0
  assert all(cnt[k]==0 for k in ('generations','CE','F1','TEST_QA','backwards','optimizer_updates','new_checkpoints'))
  assert r['readout_sha256']==sha(run/'READOUT.jsonl') and r['fit_sha256']==sha(run/'FIT.json')
  assert r['temporary_resources_sha256']==sha(run/'TEMPORARY_RESOURCES.json')
  resources=read(run/'TEMPORARY_RESOURCES.json');assert resources['resources']==r['temporary_resources'] and resources['student_registered_layout']==r['student_registered_layout']
  layout=r['student_registered_layout'];assert layout==loaded['student_registered_layout']
  assert layout['registered_neural_parameters']==893675552 and layout['registration_deduplicated'] and layout['additional_deployment_parameters']==0 and layout['all_arms_same_registered_graph']
  assert type(layout['registered_neural_weight_bytes']) is int and layout['registered_neural_weight_bytes']>0
  assert r['temporary_resources']['is_deployment_inference_peak'] is False and r['temporary_resources']['is_speedup_measurement'] is False
  assert fit['QA']==80 and fit['fixed_bias'] is True and fit['additional_deployment_parameters']==0 and fit['candidate_weight_dtype']=='torch.float16'
  assert finite(fit['lambda_value'])>0 and fit['lambda_value']==.01*fit['teacher_gram_mean_diagonal'] and set(fit['arms'])=={'B','C','D'}
  for arm in ('B','C','D'):
   x=fit['arms'][arm];assert x['lambda_value']==fit['lambda_value']
   for stage in ('precast_fit','cast_fit'):
    assert x[stage]['includes_actual_FP16_GEMM'] is False
    for k in ('ideal_linear_residual_raw','ridge_penalty','regularized_objective_raw','FP64_reduction_scale'):finite(x[stage][k])
  results[method]=r;fits[method]=fit
  for name in ('STARTED.json','COMPLETE.json','RESULT.json','MODEL_LOADED.json','FIT.json','READOUT.jsonl','TEMPORARY_RESOURCES.json'):
   p=run/name;inputs[str(p)]=sha(p)
  inputs[str(O/(method+'_EXIT.json'))]=sha(O/(method+'_EXIT.json'))
  for i,selection in enumerate(expected):
   p=run/f'READOUT_{i:03d}_COMPLETE.json';q=read(p)
   assert q['qa_index']==i and q['qa_id_sha256']==selection['id_sha256'] and q['arms']==list(ARMS)
   inputs[str(p)]=sha(p)
 return dict(status='READY',raw_rows_parsed=False),(b,D,root,expected,inputs,results,fits)

def paired(rows,left,right,ids):
 # Losses: right-left <0 is a raw numeric win for right; not significance/equivalence.
 out={}
 for label,key in [('raw_block_MSE','block'),('full_vocab_KL','KL')]:
  a=[rows[(i,left)][key] for i in ids];b=[rows[(i,right)][key] for i in ids];delta=[v-u for u,v in zip(a,b)]
  out[label]=dict(N=len(ids),left_mean=mean(a),right_mean=mean(b),mean_right_minus_left=mean(delta),right_wins=sum(x<0 for x in delta),right_losses=sum(x>0 for x in delta),exact_numeric_ties=sum(x==0 for x in delta))
 return dict(left=left,right=right,lower_is_better=True,metrics=out)

def main():
 p=argparse.ArgumentParser();p.add_argument('--binding',type=Path,required=True);p.add_argument('--execute',action='store_true');args=p.parse_args()
 status,state=terminal_inputs(args.binding)
 if state is None:print(json.dumps(status));return
 b,D,root,expected,inputs,results,fits=state;destination=D/'PILOT_PAIRED_SUMMARY.json'
 if destination.exists():
  cached=read(destination);assert cached['input_sha256']==inputs and cached['consumer_sha256']==sha(__file__) and cached['binding_sha256']==sha(args.binding)
  print(json.dumps(dict(status='CACHED',sha256=sha(destination),raw_rows_parsed=False)));return
 if not args.execute:print(json.dumps(dict(status='READY',raw_rows_parsed=False,input_sha256=inputs)));return
 claim=D/'PILOT_READOUT_STARTED.json'
 assert not claim.exists(),'Earlier consumer started without valid summary: no automatic replay'
 with claim.open('x') as f:json.dump(dict(binding_sha256=sha(args.binding),consumer_sha256=sha(__file__),input_sha256=inputs,unix=time.time()),f,indent=2);f.flush();os.fsync(f.fileno())
 expected_map={x['id_sha256']:x for x in expected};ids=[x['id_sha256'] for x in expected];answer={}
 for method in METHODS:
  rows={};all_records=[]
  with (root/method/'READOUT.jsonl').open() as stream:
   for line in stream:
    rec=json.loads(line,parse_constant=bad_constant);identity=rec['qa_id_sha256'];arm=rec['arm']
    assert identity in expected_map and arm in ARMS and (identity,arm) not in rows
    assert rec['domain']==expected_map[identity]['domain'] and rec['task']==expected_map[identity]['task']
    m=rec['metrics'];block=finite(m['block_output']['mean_squared_vector_error']);kl=finite(m['teacher_to_student_KL']);assert block>=0
    assert type(m['target_positions']) is int and m['target_positions']>0 and 0<=finite(m['top1_agreement'])<=1
    rows[(identity,arm)]=dict(block=block,KL=kl,domain=rec['domain'],record=rec);all_records.append(rec)
  assert len(rows)==len(all_records)==80 and set(rows)=={(i,a) for i in ids for a in ARMS}
  for identity in ids:
   assert len({rows[(identity,a)]['record']['metrics']['target_positions'] for a in ARMS})==1
  # Recompute only aggregation of existing scalar metrics, no model/score replay.
  for arm in ARMS:
   values=[rows[(i,arm)]['record']['metrics'] for i in ids];r=results[method]['arms'][arm]
   closed_mean(r['raw_block_MSE'],[m['block_output']['mean_squared_vector_error'] for m in values]);closed_mean(r['KL'],[m['teacher_to_student_KL'] for m in values])
   closed_mean(r['top1_agreement'],[m['top1_agreement'] for m in values]);closed_mean(r['block_relative_squared_error'],[m['block_output']['relative_squared_error'] for m in values])
   closed_mean(r['block_MSE_delta_vs_A'],[m['block_MSE_delta_vs_A'] for m in values]);closed_mean(r['KL_delta_vs_A'],[m['KL_delta_vs_A'] for m in values])
   assert set(r['domains'])=={x['domain'] for x in expected},'All five domain means must be present'
   for domain,x in r['domains'].items():
    vs=[rows[(i,arm)]['record']['metrics'] for i in ids if expected_map[i]['domain']==domain];assert len(vs)==4
    closed_mean(x['KL'],[m['teacher_to_student_KL'] for m in vs]);closed_mean(x['block_relative_squared_error'],[m['block_output']['relative_squared_error'] for m in vs])
  comparisons={f'{l}_to_{r}':dict(all_QA=paired(rows,l,r,ids),domains={d:paired(rows,l,r,[i for i in ids if expected_map[i]['domain']==d]) for d in sorted({x['domain'] for x in expected})}) for l,r in [('B','C'),('C','D'),('A','D')]}
  a=results[method]['arms'];block_better=all(a['D']['raw_block_MSE']<a[z]['raw_block_MSE'] for z in ('A','C'));KL_better=all(a['D']['KL']<a[z]['KL'] for z in ('A','C'))
  c_prefix=all(a['C']['raw_block_MSE']<a[z]['raw_block_MSE'] and a['C']['KL']<a[z]['KL'] for z in ('A','B'))
  decision=dict(D_block_mean_better_than_A_and_C=block_better,D_KL_mean_better_than_A_and_C=KL_better,C_means_better_than_A_and_B=c_prefix,
   fixed_point_exit_alignment_supported_in_both_recorded_means=block_better and KL_better,
   local_to_final_bridge_not_supported=block_better and not KL_better,
   interpretation='descriptive fixed-point inequalities only; no effect-size cutoff, equivalence test, significance, expansion authorization or automatic new run')
  answer[method]=dict(original_equal_QA_arm_means=a,paired_comparisons=comparisons,decision_checks=decision,
   fit_diagnostics=fits[method],fit_warning='C and D fit losses target different outputs and MUST NOT rank the arms; each fit loss excludes actual FP16 GEMM',
   student_registered_layout=results[method]['student_registered_layout'],temporary_resources=results[method]['temporary_resources'],temporary_peak_is_not_deployment_peak=True)
 out=dict(schema='fixed-residual-exit-paired-readout-v1',consumer_sha256=sha(__file__),binding_sha256=sha(args.binding),input_sha256=inputs,methods=answer,
  paired_QA_per_method=20,raw_rows_per_method=80,new_models=0,new_scores=0,QA_text_read=0,checkpoint_payloads_read=0,
  scope='trained historical TRAIN subset, teacher-forced answer/EOS positions; one fixed block12, not full24block, free generation, F1, semantic capability or multi-seed significance',
  exact_numeric_ties_are_not_equivalence=True)
 # A changed final input cannot silently receive an old-input interpretation.
 assert all(sha(Path(p))==h for p,h in inputs.items())
 with destination.open('x') as f:json.dump(out,f,ensure_ascii=False,allow_nan=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({m:dict(decision_checks=v['decision_checks'],all_QA_pairs={k:x['all_QA'] for k,x in v['paired_comparisons'].items()}) for m,v in answer.items()}))
if __name__=='__main__':main()
