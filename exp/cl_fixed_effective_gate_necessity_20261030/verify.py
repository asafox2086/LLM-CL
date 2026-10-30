"""DRAFT unique final-score closure, no re-generation or re-scoring of baseline/predictions.
Uses original worker's native token_f1 fields and frozen IDs/references/counts.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,math,os,sys

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,j):
 with p.open('x') as f:json.dump(j,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def who(pid):
 try:
  p=Path('/proc')/str(pid);v=(p/'stat').read_text().rsplit(') ',1)[1].split();return dict(pid=pid,pgid=int(v[2]),uid=p.stat().st_uid,startticks=v[19])
 except FileNotFoundError:return None
def aggregate(records):
 domains={}
 for r in records:
  d=domains.setdefault(r['domain'],dict(count=0,weighted_F1=0.));d['count']+=r['count'];d['weighted_F1']+=r['count']*r['token_f1']
 assert len(domains)==5 and sum(x['count'] for x in domains.values())==21454
 values={d:x['weighted_F1']/x['count'] for d,x in domains.items()}
 return dict(final19=records,domain_QA_weighted_F1=values,five_domain_macro=math.fsum(values.values())/5,AP19=math.fsum(x['token_f1'] for x in records)/19,TEST_QA=21454)
def paired_bucket():return dict(static_higher=0,dynamic_higher=0,F1_tie=0,answer_string_same=0,positive_deltas=[],negative_deltas=[])
def add_pair(bucket,static,dynamic):
 delta=static['scores']['token_f1']-dynamic['scores']['token_f1']
 if delta>0:bucket['static_higher']+=1;bucket['positive_deltas'].append(delta)
 elif delta<0:bucket['dynamic_higher']+=1;bucket['negative_deltas'].append(delta)
 else:bucket['F1_tie']+=1
 assert isinstance(static['prediction'],str) and isinstance(dynamic['prediction'],str)
 bucket['answer_string_same']+=int(static['prediction']==dynamic['prediction'])
def finish_pairs(bucket):
 positive=math.fsum(bucket['positive_deltas']);negative=math.fsum(bucket['negative_deltas']);N=bucket['static_higher']+bucket['dynamic_higher']+bucket['F1_tie']
 return dict(QA=N,static_higher=bucket['static_higher'],dynamic_higher=bucket['dynamic_higher'],F1_tie=bucket['F1_tie'],answer_string_same=bucket['answer_string_same'],answer_string_different=N-bucket['answer_string_same'],positive_delta_sum=positive,negative_delta_sum=negative,net_delta_sum=positive+negative,mean_per_QA_delta=(positive+negative)/N,delta_units='sum of stored per-QA F1 point differences; not five-domain macro')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 spec=importlib.util.spec_from_file_location('cl_fixed_worker_scope',Path(__file__).parent/'run.py');runmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(runmod)
 b=read(a.binding);old,orig,root,port,original_run,cfg,rows,manifest,c,baseline=runmod.scope(b);out=Path(b['output_root']);O=out/'verification_v1'
 exitpath=Path(b['worker_exit_receipt_path'])
 if not (out/'COMPLETE.json').exists() or not exitpath.exists():print(json.dumps(dict(status='WAITING',prediction_rows_opened=False)));return
 assert not (out/'FAILURE.json').exists();start=read(out/'STARTED.json');complete=read(out/'COMPLETE.json');ex=read(exitpath)
 assert ex['returncode']==0 and ex['identity']==start['identity']==complete['identity'] and ex['identity_gone'] and who(ex['identity']['pid'])!=ex['identity']
 assert complete['binding_sha256']==start['binding_sha256']==sha(a.binding) and complete['RESULT_sha256']==sha(out/'RESULT.json')
 result=read(out/'RESULT.json');assert result['binding_sha256']==sha(a.binding) and result['original_parameters_and_buffers_unchanged'] and result['anchors_task_weights_unchanged'] and result['generation_config_unchanged'] and result['no_bank_removed_or_folded']
 gate=result['fixed_effective_gate_report'];assert gate['fixed_main_coverage_complete'] and gate['counts']['fixed_main_calls']>0 and set(gate['fixed_per_layer'].values())=={gate['counts']['fixed_main_calls']} and len(gate['fixed_per_layer'])==168
 controls=read(out/'CONTROLS_COMPLETE.json');assert [x['calibration_index'] for x in controls['controls']]==[0,16,32,48,64]
 assert all(x['all_168_projection_full_output_byte_digests_equal'] and x['whole_native_tokens_bitwise_equal'] and x['new_scores']==0 for x in controls['controls'])
 if not a.execute:print(json.dumps(dict(status='READY',no_models_or_scoring=True,output_exists=O.exists())));return
 assert not O.exists(),'Unique verifier: no replay/alternative output';O.mkdir();write(O/'STARTED.json',dict(binding_sha256=sha(a.binding),worker_RESULT_sha256=sha(out/'RESULT.json'),source_sha256=sha(Path(__file__)),baseline_rescored=False))
 data=runmod.tests(root,manifest);records=[];proofs=[];prediction_calls=0
 original_hashes={x['task']:x for x in read(b['baseline_prediction_hashes_path'])['records']};baseline_by_task={x['task']:x for x in baseline}
 paired_overall=paired_bucket();paired_domains={}
 assert len(result['scores'])==19
 for task,expected in zip(manifest['tasks'],result['scores']):
  task_id=task['task'];assert expected['task']==task_id and expected['domain']==task['domain'];p=Path(expected['predictions_path']);s=Path(expected['score_path'])
  assert p==out/'evaluations/stage_19'/f'T{task_id:02d}_test_predictions.jsonl' and s==p.with_name(p.name.replace('.jsonl','.scores.json'))
  assert sha(p)==expected['predictions_sha256'] and sha(s)==expected['score_sha256'];score=read(s);native=data[(task_id,'test')];n=0;F1=[]
  dh=original_hashes[task_id];dp=Path(dh['path']);raw=dp.read_bytes();assert hashlib.sha256(raw).hexdigest()==dh['sha256'] and len(raw)==dh['bytes']
  original=[json.loads(line) for line in raw.splitlines()];assert len(original)==len(native)==dh['TEST_count']
  dynamic_F1=[];dynamic_by_id={}
  for oldrow,testrow in zip(original,native):
   assert oldrow['id']==testrow['id'] and oldrow['references']==testrow['answers'] and oldrow['generation_cap']==task['max_new_tokens']
   assert oldrow['id'] not in dynamic_by_id and math.isfinite(oldrow['scores']['token_f1']) and 0<=oldrow['scores']['token_f1']<=100
   dynamic_by_id[oldrow['id']]=oldrow;dynamic_F1.append(oldrow['scores']['token_f1'])
  assert sum(dynamic_F1)/len(dynamic_F1)==baseline_by_task[task_id]['token_f1'],'Original dynamic raw scores must exactly close original native ordered score'
  paired_domain=paired_domains.setdefault(task['domain'],paired_bucket())
  with p.open() as f:
   for line in f:
    v=json.loads(line);row=native[n];assert v['id']==row['id'] and v['references']==row['answers'] and v['generation_cap']==task['max_new_tokens'] and 1<=v['output_tokens']<=v['generation_cap']
    assert 0<=v['scores']['token_f1']<=100 and math.isfinite(v['scores']['token_f1']);F1.append(v['scores']['token_f1']);prediction_calls+=v['output_tokens'];n+=1
    original_row=dynamic_by_id[v['id']];assert original_row['references']==v['references'];add_pair(paired_overall,v,original_row);add_pair(paired_domain,v,original_row)
  assert n==task['files']['test']['count']==score['count']==expected['count']
  native_mean=sum(F1)/n;fsum_mean=math.fsum(F1)/n # native rt.evaluate uses this exact ordered sum
  assert native_mean==score['token_f1']==expected['token_f1']
  records.append(dict(task=task_id,domain=task['domain'],count=n,token_f1=score['token_f1']));proofs.append(dict(task=task_id,predictions_sha256=sha(p),score_sha256=sha(s),native_ordered_sum_bitwise_equal=True,fsum_mean_minus_native_score_report_only=fsum_mean-score['token_f1'],original_dynamic_predictions_sha256=dh['sha256'],original_dynamic_native_ordered_sum_bitwise_equal=True))
 assert prediction_calls==gate['counts']['fixed_main_calls'] and result['counts']['TEST_generations']==21454
 fixed=aggregate(records);dynamic=aggregate(baseline)
 paired=dict(overall=finish_pairs(paired_overall),domains={d:finish_pairs(v) for d,v in paired_domains.items()},F1_tie_is_not_answer_string_equality=True,reference='static minus original dynamic stored native token_f1; no re-scoring')
 assert paired['overall']['QA']==21454 and sum(v['QA'] for v in paired['domains'].values())==21454
 report=dict(schema='cl-fixed-effective-gate-verified-v1',binding_sha256=sha(a.binding),worker_RESULT_sha256=sha(out/'RESULT.json'),complete_sha256=sha(out/'COMPLETE.json'),exit_receipt_sha256=sha(exitpath),
  fixed=fixed,dynamic_original_baseline_reused=dynamic,delta_five_domain_macro=fixed['five_domain_macro']-dynamic['five_domain_macro'],score_closure=proofs,
  no_baseline_or_prediction_rescoring=True,source_metric_is_original_native_token_F1=True,coefficient_reference='frozen_oldTF55e_mean_C64_cast_FP32_no_renorm',no_parameter_or_memory_reduction=True,
  FWT_BWT=None,no_posthoc_preservation_margin=True,no_general_static_necessity_claim=True,paired_changes=paired,baseline_prediction_hashes_sha256=b['baseline_prediction_hashes_sha256'],paired_five_domain_mean_delta=math.fsum(v['mean_per_QA_delta'] for v in paired['domains'].values())/5)
 write(O/'VERIFIED.json',report);write(O/'COMPLETE.json',dict(VERIFIED_sha256=sha(O/'VERIFIED.json'),source_sha256=sha(Path(__file__))));print(json.dumps(dict(status='VERIFIED',VERIFIED_sha256=sha(O/'VERIFIED.json'),delta_five_domain_macro=report['delta_five_domain_macro'])))
if __name__=='__main__':main()
