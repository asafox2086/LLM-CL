"""One paired read of existing verified PAS dynamic/static stored token_f1.
No model, generation, scoring, coefficient fitting or CL partial inputs.
The static recipe includes fixed mixing, folding and dtype changes: not pure router causality.
"""
from pathlib import Path
import argparse,hashlib,json,math,os
TRUST={'dynamic_proof':'a17c1000dfff6816644d3e86170fe255d6c3225eebfa911d4c0834fcabb0dc9f','static_verified':'3f77fd3ad08933402d6f5abc1e8d13fba66946cc9bba9a81023a8a01d649c410','static_result':'f0608237e99a90e1b8ab9956b11f40dac08faae3522794c282bc08c6e206dda6','binding':'469735a796879588c1df722b12a5573010da3a1210d0453e8883c33f7c2268dc'}
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write(chr(10));f.flush();os.fsync(f.fileno())
def bucket():return dict(N=0,static_higher=0,dynamic_higher=0,F1_tie=0,answer_string_same=0,positive=[],negative=[])
def add(v,s,d):
 delta=s['scores']['token_f1']-d['scores']['token_f1'];v['N']+=1
 if delta>0:v['static_higher']+=1;v['positive'].append(delta)
 elif delta<0:v['dynamic_higher']+=1;v['negative'].append(delta)
 else:v['F1_tie']+=1
 assert isinstance(s['prediction'],str) and isinstance(d['prediction'],str)
 v['answer_string_same']+=int(s['prediction']==d['prediction'])
def finish(v):
 pos=math.fsum(v.pop('positive'));neg=math.fsum(v.pop('negative'));assert v['N']==v['static_higher']+v['dynamic_higher']+v['F1_tie']
 return dict(**v,answer_string_different=v['N']-v['answer_string_same'],positive_delta_sum=pos,negative_delta_sum=neg,net_delta_sum=pos+neg,mean_QA_delta=(pos+neg)/v['N'])
def close(a,b):return math.isfinite(a) and math.isfinite(b) and abs(a-b)<=1e-9*max(1,abs(a),abs(b))
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True)
 for n in TRUST:a.add_argument('--'+n.replace('_','-'),type=Path,required=True)
 a.add_argument('--execute',action='store_true');a=a.parse_args();root=a.root
 inputs={};objs={}
 for n,h in TRUST.items():
  p=getattr(a,n);raw=p.read_bytes();assert sha(raw)==h;inputs[n]=dict(path=str(p),sha256=h);objs[n]=json.loads(raw)
 inputs['source_sha256']=sha(Path(__file__).read_bytes())
 d=objs['dynamic_proof'];s=objs['static_result'];v=objs['static_verified'];b=objs['binding']
 assert root==Path(b['source_root']);dynamic=root/d['candidate']['run'];static=root/'exp/pas_static_final_confirmation_20261029/runs/evaluation_v1'
 assert dynamic==Path(b['original_run']) and d['candidate']['method']=='PASs_MoE' and d['candidate']['backbone']=='onevision'
 assert d['checkpoint_sha256']==b['checkpoint_sha256']=='583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a'
 assert d['config_sha256']==b['config_sha256']=='79edcca96dd315260816932438b7c0ec57835f5761d2ec0b88d8ecdb5c837fb6'
 assert v['RESULT_sha256']==TRUST['static_result'] and v['binding_sha256']==TRUST['binding'] and v['checkpoint_sha256']==s['checkpoint_sha256']
 assert d['final_QA']==s['TEST_QA']==v['TEST_QA']==21454 and len(d['final_native19'])==len(s['tasks'])==19
 assert s['score_field']=='token_f1' and s['metric']==d['primary_metric']=='existing raw Answer F1'
 assert sha((static/'COMPLETE.json').read_bytes())==v['COMPLETE_sha256'] and sha((static/'RESULT.json').read_bytes())==TRUST['static_result']
 assert sha((dynamic/'config.json').read_bytes())==b['config_sha256']
 raw=(root/b['manifest_path']).read_bytes();assert sha(raw)==b['manifest_sha256'];manifest=json.loads(raw)
 assert len(manifest['tasks'])==19 and sum(t['files']['test']['count'] for t in manifest['tasks'])==21454
 for rel,key in [('analysis/answer_f1.py','analysis/answer_f1.py'),('summary/primary_metric_20261017/metric_lock.json','summary/primary_metric_20261017/metric_lock.json')]:assert sha((root/rel).read_bytes())==b['runtime_files'][key]==d['source_sha256'][key]
 O=root/'summary/pas_static_final_confirmation_20261029/paired_token_f1_v1'
 if (O/'COMPLETE.json').exists():
  result=read(O/'RESULT.json');assert result['inputs']==inputs and read(O/'COMPLETE.json')['RESULT_sha256']==sha((O/'RESULT.json').read_bytes());print(json.dumps(dict(status='CACHED',RESULT_sha256=sha((O/'RESULT.json').read_bytes()))));return
 assert not O.exists(),'One readout namespace: preserve any started/failed attempt, do not replay'
 if not a.execute:print(json.dumps(dict(status='READY',prediction_rows_opened=False,new_models=0,new_scores=0)));return
 O.mkdir();write(O/'STARTED.json',dict(inputs=inputs,no_models_or_scoring=True))
 all_=bucket();groups={};tasks=[];seen=set();proofs=[];domain_totals={}
 for index,(t,st) in enumerate(zip(manifest['tasks'],s['tasks'])):
  k=t['task'];assert k==index+1==st['task'] and t['domain']==st['domain']
  rel=f'evaluations/stage_19/T{k:02d}_test_predictions.jsonl';db=(dynamic/rel).read_bytes();sb=(static/rel).read_bytes()
  assert sha(db)==d['prediction_sha256'][rel]==d['source_sha256'][str((dynamic/rel).relative_to(root))] and sha(sb)==s['outputs'][rel]
  dr=[json.loads(x) for x in db.splitlines()];sr=[json.loads(x) for x in sb.splitlines()];n=t['files']['test']['count'];assert len(dr)==len(sr)==st['count']==n
  q=bucket();g=groups.setdefault(t['domain'],bucket());ds=[];ss=[]
  for dy,sy in zip(dr,sr):
   assert dy['id']==sy['id'] and dy['id'] not in seen and dy['references']==sy['references'];seen.add(dy['id'])
   assert dy['generation_cap']==sy['generation_cap']==t['max_new_tokens']
   for row in (dy,sy):assert math.isfinite(row['scores']['token_f1']) and 0<=row['scores']['token_f1']<=100
   ds.append(dy['scores']['token_f1']);ss.append(sy['scores']['token_f1'])
   for target in (q,g,all_):add(target,sy,dy)
  assert sum(ds)/n==d['final_native19'][index] and sum(ss)/n==st['AnswerF1'],'Native ordered stored-score closure'
  dg=domain_totals.setdefault(t['domain'],dict(N=0,dynamic_sum=0.,static_sum=0.));dg['N']+=n;dg['dynamic_sum']+=sum(ds);dg['static_sum']+=sum(ss)
  tasks.append(dict(task=k,domain=t['domain'],dynamic_F1=sum(ds)/n,static_F1=sum(ss)/n,paired=finish(q)))
  proofs.append(dict(task=k,dynamic_raw_sha256=sha(db),static_raw_sha256=sha(sb),same_ID_reference_order_count=True,native_ordered_score_exact=True))
 assert len(seen)==all_['N']==21454
 domains={}
 for name,g in groups.items():
  totals=domain_totals[name];sf=totals['static_sum']/totals['N'];df=totals['dynamic_sum']/totals['N'];assert close(sf,s['domains'][name]['AnswerF1']) and close(df,d['domains'][name]['F1'])
  pair=finish(g);assert close(pair['mean_QA_delta'],sf-df);domains[name]=dict(QA=totals['N'],static_F1=sf,dynamic_F1=df,delta_F1=sf-df,paired=pair)
 macro=math.fsum(x['delta_F1'] for x in domains.values())/5;assert close(macro,s['domain_macro_final']-d['domain_macro_final'])
 result=dict(inputs=inputs,tasks=tasks,domains=domains,overall_QA_pooled=finish(all_),five_domain_macro_delta=macro,static_macro_verified=s['domain_macro_final'],dynamic_macro_verified=d['domain_macro_final'],proofs=proofs,new_models=0,new_generations=0,new_scores=0,F1_tie_is_not_answer_string_equality=True,delta_sign='static minus dynamic stored native F1',sum_units='Per-QA F1-point sums, not five-domain macro',interpretation_boundary='PAS fixed mixing plus folding/dtype changes; not pure gate causality, not router training failure, not CL or 10pct evidence.')
 write(O/'RESULT.json',result);h=sha((O/'RESULT.json').read_bytes());write(O/'COMPLETE.json',dict(RESULT_sha256=h,source_sha256=inputs['source_sha256']));print(json.dumps(dict(status='COMPLETE',RESULT_sha256=h)))
if __name__=='__main__':main()
