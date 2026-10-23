"""One case score pass, then proof-only gate and aggregate. No work on import."""
import ast
from collections import Counter
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from pathlib import Path
from core import ROOT,PORT,REPORT,WORK,CASES,BUDGET,TRAINABLE,EXTERNAL_ANCHOR,study,sha,write

DOMAINS=('natural','medical','geography','finance','traffic')
SOURCES=('core.py','runtime.py','common.py','run.py','freeze.py','dispatch.py')

def data():
 s=study();path,m,original=s.original_train()
 split=json.loads((s.REPORT/'SPLIT.json').read_text());assert (split['FIT_QA'],split['DEV_QA'])==(1213,100)
 fit={};dev={};mapped={};plan=[];groups={}
 for gid,g in split['groups'].items():
  for i in g['ids']:groups[i]=gid
 for t in m['tasks']:
  k=t['task'];fs=set(split['fit_ids'][str(k)]);ds=set(split['dev_ids'][str(k)]);assert not fs&ds
  fit[k]=[r for r in original[k] if r['id'] in fs];dev[k]=[r for r in original[k] if r['id'] in ds]
  assert [r['id'] for r in fit[k]]==split['fit_ids'][str(k)]
  assert [r['id'] for r in dev[k]]==split['dev_ids'][str(k)]
  mapped[k,'train']=fit[k];mapped[k,'dev']=dev[k]
  order=sorted(fit[k],key=lambda r:hashlib.sha256(f"42:{k}:{r['id']}".encode()).digest())
  t['training_order_sha256']=hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
  t['files']={'train':{'count':len(fit[k])},'dev':{'count':len(dev[k])}}
  plan.append(dict(stage=k,QA=len(fit[k]),updates=math.ceil(len(fit[k])/16)))
 assert sum(p['updates'] for p in plan)==87
 return path,m,split,fit,dev,mapped,plan,groups

def check_ready(base=True):
 v=json.loads((REPORT/'READY.json').read_text());assert v['cases']==list(CASES) and v['case_budgets']==BUDGET
 assert v['aggregate_budget_seconds']==36000 and v['trainable_coordinates']==TRAINABLE
 for key in ('sources','evidence','dependencies'):
  for n,h in v[key].items():assert sha(ROOT/n)==h,n
 if base:
  for n,h in v['base_files'].items():assert sha(ROOT/n)==h,n
 assert study().training_ast()[1]==v['training_ast_sha256']
 assert str(Path(sys.executable).resolve())==v['interpreter']
 for n,h in v['environment'].items():assert sha(n)==h,n
 return v

def scorer():
 tree=ast.parse((PORT/'runtime_utils.py').read_text());ns=dict(re=re,Counter=Counter)
 defs=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('score','tokens')]
 assert len(defs)==2
 exec(compile(ast.Module(body=defs,type_ignores=[]),'<original-score>','exec'),ns)
 return ns['score']

def verify_case(case):
 assert case in CASES
 report=REPORT/'cases'/case;marker=report/'CASE_VERIFY_STARTED.json';proof=report/'CASE_VERIFIED.json'
 assert not marker.exists() and not proof.exists(),'No second case verification or score pass'
 ready=check_ready(False);_,m,split,fit,dev,_,plan,_=data();root=WORK/case
 write(marker,dict(case=case,complete_sha256=sha(report/'COMPLETE.json'),score_pass_limit=1))
 c=json.loads((report/'COMPLETE.json').read_text());started=json.loads((report/'STARTED.json').read_text())
 assert c['complete'] and c['case']==case and c['TEST_QA']==0 and c['worker_pid']==started['pid']
 assert c['ready_sha256']==sha(REPORT/'READY.json') and 0<c['model_RUN_seconds']<=BUDGET[case]
 assert started['ready_sha256']==c['ready_sha256'] and started['budget_seconds']==BUDGET[case]
 exited=json.loads((report/'ACTUAL_EXIT.json').read_text())
 assert exited['case']==case and exited['identity']=={k:started[k] for k in ('pid','startticks','uid','pgid')}
 assert exited['code']==0 and c['model_RUN_seconds']<=exited['observed_elapsed_bound_seconds']<=BUDGET[case]
 required={root/n for n in ('INITIAL.json','TRAIN_LAYOUT.json','config.json','CALLBACKS.json','TRAIN_COMPLETE.json',
  'INPUT_RECORDS.jsonl','REQUESTS.json','logs/events.jsonl','resources.jsonl','memory_summary.json')}
 for p in plan:
  k=p['stage'];required.update(root/n for n in (f'checkpoints/S{k:02d}.json',f'validity/stage_{k:02d}.json',f'gradients/S{k:02d}_first_raw.pt'))
  for j in range(1,p['updates']+1):required.update((root/f'optimizer/S{k:02d}_{j:03d}.json',root/f'gradients/S{k:02d}_{j:03d}.json'))
 required.add(root/'checkpoints/S19.pt')
 required.update(root/f"evaluations/stage_19/T{t['task']:02d}_dev_predictions.jsonl" for t in m['tasks'] if dev[t['task']])
 if case=='E':
  snapshots=list((root/'method_state/I_snapshots').glob('*.pt'));assert len(snapshots)==19;required.update(snapshots)
 assert {p.relative_to(ROOT).as_posix() for p in required}<=set(c['file_sha256'])
 actual={p.relative_to(ROOT).as_posix() for p in root.rglob('*') if p.is_file() and p.name!='COMPILED.pt' and p.suffix not in ('.tmp','.partial')}
 assert actual==set(c['file_sha256'])
 for n,h in c['file_sha256'].items():assert sha(ROOT/n)==h,n
 cfg=json.loads((root/'config.json').read_text());assert cfg==c['config']
 assert (cfg['FIT_QA'],cfg['DEV_QA'],cfg['TEST_QA'],cfg['updates'])==(1213,100,0,87)
 assert cfg['source_sha256']==sha(Path(__file__).with_name('run.py'))
 assert cfg['manifest_sha256']==ready['manifest_sha256'] and cfg['split_sha256']==ready['split_sha256']
 initial=json.loads((root/'INITIAL.json').read_text());assert initial['common_qv_digest']==c['common_qv_digest']
 assert initial['base_files']==ready['base_files'] and initial['case']==case
 assert c['frozen_before_TRAIN']==c['frozen_after_TRAIN']
 assert c['final_reload'] and c['compiled_before_DEV']==c['compiled_after_DEV']
 assert sha(root/c['compiled_checkpoint']['path'])==c['compiled_checkpoint']['sha256']
 lay=c['compiled_layout'];assert lay['trainable_coordinates']==0
 expected=ready['base_weight_coefficients']+(86507520 if case=='E' else 0)
 assert lay['weight_coefficients']==expected and c['compiled_checkpoint']['weight_coefficients']==expected
 assert not any('.A.' in n or '.B.' in n for n in lay['weights']),'Common q/v must be folded'
 inputs=[json.loads(s) for s in (root/'INPUT_RECORDS.jsonl').read_text().splitlines()]
 lookup={}
 for x in inputs:
  if x['id'] in lookup:assert lookup[x['id']]==x['prepared_input_SHA256']
  lookup[x['id']]=x['prepared_input_SHA256']
 expected_ids={r['id'] for rows in [*fit.values(),*dev.values()] for r in rows}
 assert set(lookup)==expected_ids and len(lookup)==1313
 expected_train=[r['id'] for k in range(1,20) for r in sorted(fit[k],key=lambda r:hashlib.sha256(f"42:{k}:{r['id']}".encode()).digest())]
 assert [r['id'] for r in inputs if r['phase']=='TRAIN']==expected_train
 assert [r['id'] for r in inputs if r['phase']=='NATIVE_GATE']==[fit[k][0]['id'] for k in range(1,20)]
 assert [r['id'] for r in inputs if r['phase']=='DEV']==[r['id'] for k in range(1,20) for r in dev[k]]
 assert len(inputs)==1332
 for task in m['tasks']:
  for row in dev[task['task']]:assert row['id'] in lookup
 assert len(c['native_gates'])==19
 assert [g['stage'] for g in c['native_gates']]==list(range(1,20))
 assert c['request_counts']['answer_NLL']==1213+19 and c['request_counts']['generation']==100+19
 events=[json.loads(s) for s in (root/'logs/events.jsonl').read_text().splitlines()];assert len(events)==87
 train_layout=json.loads((root/'TRAIN_LAYOUT.json').read_text());coords=TRAINABLE[case]
 assert train_layout['trainable_coordinates']==coords
 assert all(v['dtype']=='torch.float32' for v in train_layout['weights'].values() if v['requires_grad'])
 callbacks=json.loads((root/'CALLBACKS.json').read_text());assert len(callbacks)==19
 for p in plan:
  k=p['stage'];es=[e for e in events if e['stage']==k]
  assert [e['step'] for e in es]==list(range(1,p['updates']+1)) and es[-1]['qa_seen']==p['QA']
  valid=json.loads((root/'validity'/f'stage_{k:02d}.json').read_text())
  assert valid['passed'] and valid['actual_parameters_changed'] and valid['positive_lr_updates']==p['updates']
  cb=callbacks[k-1];assert cb['stage']==k and cb['post_backward']==cb['recording_true']==cb['recording_false']==p['QA']
  assert cb['after_update']==p['updates']
  if case=='E':assert cb['usage_callbacks_before_end']>0 and cb['importance_modules']==96
  cp=json.loads((root/'checkpoints'/f'S{k:02d}.json').read_text())
  assert cp['stage']==k and cp['full_graph_spec_and_method_state'] and cp['bytes']>0 and len(cp['sha256'])==64
  for e in es:
   j=e['step'];assert all(math.isfinite(e[n]) for n in ('loss','gradient_l2_preclip','gradient_l2_postclip'))
   assert all(math.isfinite(v) and v>0 for v in e['learning_rate_used'])
   g=json.loads((root/'gradients'/f'S{k:02d}_{j:03d}.json').read_text())
   assert set(g['raw'])==set(c['trainable_names']) and all(z['finite'] for z in g['raw'].values())
   o=json.loads((root/'optimizer'/f'S{k:02d}_{j:03d}.json').read_text())
   assert o['stage']==k and o['step']==j and o['parameter_IDs_match']
   ordered=sorted(fit[k],key=lambda r:hashlib.sha256(f"42:{k}:{r['id']}".encode()).digest())
   assert o['QA_IDs']==[r['id'] for r in ordered[(j-1)*16:j*16]]
   assert o['parameter_names']==c['trainable_names'] and len(set(o['parameter_IDs']))==len(o['parameter_IDs'])
   assert o['coordinates']==coords and o['gradient_bytes']==4*coords and o['two_moment_bytes']==8*coords
   assert o['moment_devices']==['cpu'] and o['gradient_dtypes']==['torch.float32']
 assert len(list((root/'optimizer').glob('*.json')))==87 and len(list((root/'gradients').iterdir()))==106
 rows=[json.loads(s) for s in (root/'resources.jsonl').read_text().splitlines()]
 assert rows and all(r['pid']==started['pid'] and r['visible_gpu']=='1' for r in rows)
 assert any(r['phase']=='run_finished' for r in rows)
 score=scorer();scores={};domains={d:[] for d in DOMAINS};prediction_hashes={}
 for t in m['tasks']:
  k=t['task']
  if not dev[k]:continue
  f=root/'evaluations/stage_19'/f'T{k:02d}_dev_predictions.jsonl'
  predictions=[json.loads(s) for s in f.read_text().splitlines()]
  assert [r['id'] for r in predictions]==[r['id'] for r in dev[k]]
  for r,g in zip(predictions,dev[k]):
   assert r['references']==g['answers'] and r['generation_cap']==t['max_new_tokens']
   assert r['prepared_input_SHA256']==lookup[g['id']]
   value=score(r['prediction'],g['answers'])['token_f1'];assert math.isfinite(value) and 0<=value<=100
   scores[g['id']]=value;domains[t['domain']].append(value)
  prediction_hashes[f.relative_to(ROOT).as_posix()]=sha(f)
 assert len(scores)==100 and all(len(v)==20 for v in domains.values())
 means={d:sum(v)/len(v) for d,v in domains.items()}
 result=dict(case=case,complete=True,quality=dict(domains=means,domain_macro=sum(means.values())/5),scores=scores,
  prediction_SHA256=prediction_hashes,input_SHA256=lookup,common_qv_digest=c['common_qv_digest'],
  base_files=ready['base_files'],compiled_layout=lay,model_RUN_seconds=c['model_RUN_seconds'],
  ready_SHA256=sha(REPORT/'READY.json'),complete_SHA256=sha(report/'COMPLETE.json'),TEST_QA=0,
  original_score_source_SHA256=sha(PORT/'runtime_utils.py'),controller_exit_SHA256=sha(report/'ACTUAL_EXIT.json'),
  observed_elapsed_bound_seconds=exited['observed_elapsed_bound_seconds'],only_one_score_pass=True)
 write(proof,result);return result

def bound_proof(case,ready):
 p=REPORT/f'cases/{case}/CASE_VERIFIED.json';v=json.loads(p.read_text())
 assert v['case']==case and v['complete'] and v['ready_SHA256']==sha(REPORT/'READY.json')
 assert v['base_files']==ready['base_files'] and v['complete_SHA256']==sha(REPORT/f'cases/{case}/COMPLETE.json')
 assert v['controller_exit_SHA256']==sha(REPORT/f'cases/{case}/ACTUAL_EXIT.json')
 return v

def gate():
 p=REPORT/'GATE_DECISION.json';assert not p.exists()
 ready=check_ready(False);e=bound_proof('E',ready);q=bound_proof('Q',ready)
 for key in ('input_SHA256','common_qv_digest','base_files','ready_SHA256'):assert e[key]==q[key],key
 value=dict(run_R=e['quality']['domain_macro']>q['quality']['domain_macro'],
  E_macro=e['quality']['domain_macro'],Q_macro=q['quality']['domain_macro'],
  source_proof_SHA256={k:sha(REPORT/f'cases/{k}/CASE_VERIFIED.json') for k in ('E','Q')},
  point_rule_only_not_stable_effect=True,no_new_score_pass=True)
 write(p,value);return value

def paired(a,b,split,label):
 spec=importlib.util.spec_from_file_location('native_reuse_group_math',ROOT/'analysis/osft_increment_paired_analysis_20261022.py')
 mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 groups=mod.known_groups(split);rng=random.Random(20261024);values={d:[] for d in (*DOMAINS,'macro')}
 for _ in range(10000):
  dd={}
  for d in DOMAINS:
   selected=[rng.randrange(len(groups[d])) for _ in groups[d]];ids=[i for j in selected for i in groups[d][j][1]]
   dd[d]=sum(a[i]-b[i] for i in ids)/len(ids);values[d].append(dd[d])
  values['macro'].append(sum(dd.values())/5)
 ids_by={d:[i for _,ii in groups[d] for i in ii] for d in DOMAINS}
 return dict(label=label,all_QA_differences=[dict(id_SHA256=hashlib.sha256(i.encode()).hexdigest(),domain=d,
  R=a[i],reference=b[i],difference_pp=a[i]-b[i]) for d,ii in ids_by.items() for i in ii],
  intervals_pp={d:[mod.percentile(v,.025),mod.percentile(v,.975)] for d,v in values.items()},
  seed=20261024,replicates=10000,exploratory_known_groups_not_holdout_seed_uncertainty_or_equivalence=True)

def aggregate():
 assert not (REPORT/'VERIFIED.json').exists() and not (REPORT/'AGGREGATE_STARTED.json').exists()
 write(REPORT/'AGGREGATE_STARTED.json',dict(no_second_bootstrap=True))
 ready=check_ready(False);g=json.loads((REPORT/'GATE_DECISION.json').read_text());keys=('E','Q','R') if g['run_R'] else ('E','Q')
 cases={k:bound_proof(k,ready) for k in keys}
 assert g['source_proof_SHA256']=={k:sha(REPORT/f'cases/{k}/CASE_VERIFIED.json') for k in ('E','Q')}
 assert g['run_R']==(cases['E']['quality']['domain_macro']>cases['Q']['quality']['domain_macro'])
 assert g['E_macro']==cases['E']['quality']['domain_macro'] and g['Q_macro']==cases['Q']['quality']['domain_macro']
 assert sum(c['model_RUN_seconds'] for c in cases.values())<=36000
 eligible=False;pairs={}
 if g['run_R']:
  split_path=study().REPORT/'SPLIT.json';assert sha(split_path)==ready['split_sha256'];split=json.loads(split_path.read_text())
  r=cases['R'];e=cases['E'];q=cases['Q']
  for key in ('input_SHA256','common_qv_digest','base_files','ready_SHA256'):assert r[key]==e[key]==q[key],key
  pairs={k:paired(r['scores'],cases[k]['scores'],split,'R_minus_'+k) for k in ('Q','E')}
  p=r['compiled_layout']['weight_coefficients']
  eligible=(p==q['compiled_layout']['weight_coefficients'] and p<e['compiled_layout']['weight_coefficients']
   and p<=.75*893675552 and r['quality']['domain_macro']>max(EXTERNAL_ANCHOR,e['quality']['domain_macro'],q['quality']['domain_macro']))
 result=dict(complete=True,cases=cases,proof_SHA256={k:sha(REPORT/f'cases/{k}/CASE_VERIFIED.json') for k in keys},
  R_unmeasured=not g['run_R'],new_pairs=pairs,possible_followup_necessity_review_only=eligible,
  automatic_full_confirmation=False,old_anchor=EXTERNAL_ANCHOR,old_score_and_pairs_not_repeated=True,
  not_novel_framework_or_stable_equivalent_gain=True,TEST_QA=0)
 write(REPORT/'VERIFIED.json',result);return result

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=['case','gate','aggregate']);ap.add_argument('--case',choices=CASES)
 args=ap.parse_args()
 if args.action=='case':assert args.case;verify_case(args.case)
 elif args.action=='gate':gate()
 else:aggregate()
