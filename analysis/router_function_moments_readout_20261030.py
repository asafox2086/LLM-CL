"""Read complete native-moment observations only. No model/QA/score replay."""
from pathlib import Path
import argparse,hashlib,json,importlib.util,math,os,time
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def gone(who):
 p=Path('/proc')/str(who['pid'])/'stat'
 return not p.exists() or p.read_text().rsplit(') ',1)[1].split()[19]!=str(who['startticks'])
def distribution(xs):
 xs=[float(x) for x in xs if x is not None]
 assert all(math.isfinite(x) for x in xs)
 return dict(n=len(xs),minimum=min(xs),median=float(np.median(xs)),maximum=max(xs)) if xs else dict(n=0,minimum=None,median=None,maximum=None)
def ratio(x,y):return x/y if y>0 else None

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 b=read(args.binding);d=Path(b['admission_path']).parent;e=Path(b['observation_source_dir']);root=Path(b['output_root'])
 assert b['admitted'] and read(d/'ADMISSION.json')['approved'] and read(d/'ADMISSION.json')['binding_sha256']==sha(args.binding)
 for name,digest in b['observation_sources'].items():assert sha(e/name)==digest,name
 assert sha(b['selection_path'])==b['selection_sha256']
 if not (d/'CONTROLLER_COMPLETE.json').exists():print(json.dumps(dict(ready=False,raw_moments_read=False)));return
 cc=read(d/'CONTROLLER_COMPLETE.json');assert cc['complete'] and cc['binding_sha256']==sha(args.binding) and cc['model_attempts']==2
 assert not (d/'CONTROLLER_FAILURE.json').exists()
 if (d/'READOUT.json').exists():
  cached=read(d/'READOUT.json');assert cached['binding_sha256']==sha(args.binding) and cached['consumer_sha256']==sha(__file__) and set(cached['methods'])=={'PASs_MoE','CL_MoE'}
  print(json.dumps(dict(cached=True,sha256=sha(d/'READOUT.json'))));return
 sources={}
 for method in ('PASs_MoE','CL_MoE'):
  run=root/method;complete=read(run/'COMPLETE.json');res=read(run/'RESULT.json');exit_=read(run/'EXIT.json')
  assert exit_['returncode']==0 and complete['identity']==exit_['identity'] and gone(exit_['identity'])
  assert complete['binding_sha256']==sha(args.binding) and complete['RESULT_sha256']==sha(run/'RESULT.json')
  assert not (run/'FAILURE.json').exists() and complete['seconds']<=7200
  assert res['raw_rows']==13440 and res['weights_and_buffers_unchanged'] and res['not_quality_verification']
  assert res['checkpoint_sha256']==b['methods'][method]['checkpoint_sha256'] and res['calibration_ids_sha256']==b['calibration_ids_sha256']
  cnt=res['counts'];assert cnt['checkpoint_loads']==1 and cnt['TRAIN_QA']==cnt['answer_decoder_forward']==80
  assert all(cnt[k]==0 for k in ['generations','CE','F1','TEST_QA','backwards','optimizer_updates','new_checkpoints'])
  assert sha(run/'QA_MOMENTS.jsonl')==res['QA_moments_sha256']
  sources[method]=dict(result_sha256=sha(run/'RESULT.json'),complete_sha256=sha(run/'COMPLETE.json'),moments_sha256=res['QA_moments_sha256'],seconds=complete['seconds'],counts=cnt)
 if not args.execute:print(json.dumps(dict(ready=True,raw_moments_read=False,sources=sources)));return
 assert sha(e/'statistics.py')==b['observation_sources']['statistics.py']
 spec=importlib.util.spec_from_file_location('frozen_router_moment_description',e/'statistics.py');stats=importlib.util.module_from_spec(spec);spec.loader.exec_module(stats)
 assert not (d/'READOUT_STARTED.json').exists(),'Earlier readout started without valid result; no automatic replay'
 with (d/'READOUT_STARTED.json').open('x') as f:
  json.dump(dict(binding_sha256=sha(args.binding),consumer_sha256=sha(__file__),sources=sources,unix=time.time()),f,indent=2);f.flush();os.fsync(f.fileno())
 selection=read(b['selection_path'])['calibration'];expected_ids={x['id_sha256']:x['domain'] for x in selection};assert len(expected_ids)==80
 answer={};matched_layers=None
 for method in ('PASs_MoE','CL_MoE'):
  grouped={};identities=set();max_error=0.;max_bound=0.;position_contract={};qa_receipts={}
  for index,sel in enumerate(selection):
   qr=read(root/method/f'QA_{index:03d}_COMPLETE.json');assert qr['qa_index']==index and qr['qa_id_sha256']==sel['id_sha256'] and qr['task']==sel['task'] and qr['domain']==sel['domain'] and qr['layers']==168
   qa_receipts[qr['qa_id_sha256']]=qr
  with (root/method/'QA_MOMENTS.jsonl').open() as stream:
   for line in stream:
    row=json.loads(line);qh=hashlib.sha256(row['qa_id'].encode()).hexdigest();layer=row['layer_id']
    assert expected_ids[qh]==row['domain'] and (qh,layer) not in identities;identities.add((qh,layer))
    assert row['expert_count']==(2 if method=='PASs_MoE' else 8)
    assert row['moment_target']=='FP64_sum_of_actual_expert_outputs_times_actual_effective_gates; not native FP32 delta'
    assert row['actual_native_layer_output_closure']['bitwise_equal'] and row['closure']['all_elements_within_tolerance']
    assert row['moment_arithmetic_closure']['violating_elements']==0
    assert row['gate_dispatch_operator'] in ('aten.softmax.int','aten._softmax.default')
    assert row['selected_tokens']==len(row['prediction_positions'])==len(row['position_kinds']) and row['position_kinds'][-1]=='eos'
    assert all(k=='answer' for k in row['position_kinds'][:-1])
    pos=row['prediction_positions'];assert pos==list(range(pos[0],pos[0]+len(pos))) and row['selected_tokens']==qa_receipts[qh]['target_tokens']
    contract=(tuple(pos),tuple(row['position_kinds']))
    if qh in position_contract:assert position_contract[qh]==contract
    else:position_contract[qh]=contract
    grouped.setdefault(layer,[]).append(row)
    max_error=max(max_error,row['closure']['max_absolute_error']);max_bound=max(max_bound,row['closure']['max_absolute_bound'])
  assert len(identities)==13440 and len(grouped)==168
  for layer,rows in grouped.items():assert len(rows)==80 and {hashlib.sha256(x['qa_id'].encode()).hexdigest() for x in rows}==set(expected_ids)
  if matched_layers is None:matched_layers=set(grouped)
  else:assert matched_layers==set(grouped)
  descriptions={layer:stats.describe_layer(rows) for layer,rows in sorted(grouped.items())}
  def summarize(domain=None):
   vals=[x['all'] if domain is None else x['domains'][domain] for x in descriptions.values()]
   R=sum(x['centered_Rjoint_raw'] for x in vals);I=sum(x['centered_Rindep_raw'] for x in vals);A=sum(x['mean_routed_update_energy'] for x in vals);B=sum(x['mean_base_energy'] for x in vals)
   return dict(layer_count=len(vals),gate_deviation_trace=distribution([x['gate_deviation_trace_raw'] for x in vals]),
    contrast_over_total_expert_energy=distribution([ratio(x['contrast_energy'],float(np.trace(x['mean_H']))) for x in vals]),
    joint_over_independent=distribution([ratio(x['centered_Rjoint_raw'],x['centered_Rindep_raw']) for x in vals]),
    joint_over_routed_update_energy=distribution([ratio(x['centered_Rjoint_raw'],x['mean_routed_update_energy']) for x in vals]),
    joint_over_base_energy=distribution([ratio(x['centered_Rjoint_raw'],x['mean_base_energy']) for x in vals]),
    energy_sums_across_projection_outputs=dict(Rjoint=R,Rindep=I,update=A,base=B,joint_over_independent=ratio(R,I),joint_over_update=ratio(R,A),joint_over_base=ratio(R,B)),
    numerical_diagnostics=dict(negative_uncentered_beyond_recorded_roundoff=sum(x['Rjoint_negative_beyond_roundoff'] for x in vals),max_centered_uncentered_difference=max(abs(x['centered_vs_uncentered_Rjoint_difference']) for x in vals),negative_centered_beyond_FP64_scale=int(sum(x['centered_Rjoint_raw'] < -64*np.finfo(np.float64).eps*sum(abs(t) for t in x['centered_Rjoint_terms']) for x in vals)),max_centered_uncentered_Rindep_difference=max(abs(x['centered_vs_uncentered_Rindep_difference']) for x in vals)),
    interpretation='descriptive local projection energies; not network error, CE, quality, parameter importance or generic compression guarantee')
  answer[method]=dict(sources=sources[method],all_layers=summarize(),domains={domain:summarize(domain) for domain in sorted(set(expected_ids.values()))},layers=descriptions,max_native_ideal_absolute_error=max_error,max_elementwise_rounding_bound=max_bound)
 result=dict(schema='router-function-moments-readout-v1',binding_sha256=sha(args.binding),consumer_sha256=sha(__file__),methods=answer,TEST_QA=0,new_model_loads=0,new_scores=0,new_generations=0,new_QA=0,TRAIN_only=True,not_quality_verification=True,no_success_threshold=True)
 with (d/'READOUT.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({m:{'all_layers':v['all_layers'],'domains':v['domains']} for m,v in answer.items()}))
if __name__=='__main__':main()
