"""Consume six new phases plus a sealed reused prefix once; no model imports."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import random
from controller import validate_reused_prefix

ARMS=('plain','true','shuffled')
ORDER=[(a,p) for a in ARMS for p in ('train','eval')]
MANIFEST='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SPLIT='d8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'

def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with p.open('x',encoding='utf8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def same_live(who):
 try:
  p=Path('/proc')/str(who['pid']);fields=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return fields[0]!='Z' and fields[19]==str(who['startticks']) and p.stat().st_uid==who['uid'] and int(fields[2])==who['pgid']
 except (OSError,IndexError,ValueError):return False

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--run',type=Path,required=True)
 a=ap.parse_args();root,run=a.root,a.run
 assert root.is_absolute() and run.is_absolute()
 assert not (run/'SUPERVISOR_FAILURE.json').exists()
 super=read(run/'SUPERVISOR_COMPLETE.json')
 assert not same_live(super['identity']), 'Controller must exit before consumption'
 assert [(p['arm'],p['phase']) for p in super['phases']]==ORDER
 assert super['new_generations']==555 and super['optimizer_updates']==54 and super['FIT_backwards']==762
 assert super['new_scores']==super['TEST_QA']==0 and super['active_worker'] is None
 source=Path(super['source_dir'])
 for n,h in super['source_sha256'].items():assert sha(source/n)==h
 protocol=super['protocol'];assert sha(Path(protocol['path']))==protocol['sha256']
 start=read(run/'SUPERVISOR_STARTED.json')
 assert start['identity']==super['identity'] and start['source_sha256']==super['source_sha256']
 assert start['protocol']==protocol and start['binding_sha256']==super['binding_sha256']
 binding_paths=[Path(name) for name,h in read(run/'plain/SOURCE_BINDING.json').items() if h==super['binding_sha256']]
 assert len(binding_paths)==1 and sha(binding_paths[0])==super['binding_sha256']
 binding=read(binding_paths[0]);reused=validate_reused_prefix(binding)
 prefix_run=Path(reused['run'])
 assert read(run/'REUSED_PREFIX.json')==reused
 assert sha(run/'REUSED_PREFIX.json')==start['reused_prefix_sha256']==super['reused_prefix_sha256']
 assert super['GPU']==start['GPU']==reused['GPU']==1
 assert start['reused_prefix_cost']==super['reused_prefix_cost']==reused['historical_cost']
 assert start['cumulative_model_budget_seconds']==super['cumulative_model_budget_seconds']==9000
 checked={};receipts={}
 for index,(arm,phase) in enumerate(ORDER):
  folder=run/arm
  started=read(folder/(phase+'_STARTED.json'));complete=read(folder/(phase+'_COMPLETE.json'))
  exit=read(run/('PHASE_'+arm+'_'+phase+'_EXIT.json'))
  assert exit['returncode']==0 and exit['identity_absent'] and not exit['timeout']
  assert started['identity']==complete['identity']==exit['identity']
  assert not same_live(exit['identity'])
  sealed=super['phases'][index]
  assert all(sealed[k]==exit[k] for k in exit)
  assert sealed['started_sha256']==sha(folder/(phase+'_STARTED.json'))
  assert sealed['complete_sha256']==sha(folder/(phase+'_COMPLETE.json'))
  assert complete['arm']==arm and complete['phase']==phase and complete['GPU']==super['GPU']
  assert complete['seconds']<=started['limit_seconds']
  assert complete['new_scores']==complete['TEST_QA']==0
  assert complete['data_binding']['binding_sha256']==super['binding_sha256']
  for name,h in complete['source_sha256'].items():assert sha(Path(name))==h,name
  for relative,h in complete['files'].items():
   p=(folder/relative).resolve();assert p.is_relative_to(folder.resolve())
   if str(p) not in checked:checked[str(p)]=sha(p)
   assert checked[str(p)]==h,relative
  receipts[arm,phase]=complete
 manifest_path=root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
 split_path=root/'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
 assert sha(manifest_path)==MANIFEST and sha(split_path)==SPLIT
 manifest,split=read(manifest_path),read(split_path)
 rows={};tasks={t['task']:t for t in manifest['tasks']};type_maps={}
 for t in manifest['tasks']:
  ref=t['files']['train'];p=root/ref['path'];assert sha(p)==ref['sha256']
  data=[json.loads(x) for x in p.read_bytes().splitlines()];by={x['id']:x for x in data}
  assert len(by)==len(data)==ref['count']
  for i in split['dev_ids'][str(t['task'])]:rows[t['task'],i]=by[i]
  if t['task']==16:
   ids=split['fit_ids']['16'];types=('single_choice','multiple_choice','numerical')
   true=[types.index(by[i]['metadata']['question_type']) for i in ids]
   shuffled=list(true);random.Random(42).shuffle(shuffled)
   assert len(ids)==69 and sum(x!=y for x,y in zip(true,shuffled))==40
   for arm,labels in [('true',true),('shuffled',shuffled)]:
    type_maps[arm]={hashlib.sha256(i.encode()).hexdigest():v for i,v in zip(ids,labels)}
 prefix=read(prefix_run/'PREFIX_STATE.json')
 heads=[read(run/arm/'HEAD_INITIAL.json') for arm in ARMS]
 assert len({h['digest'] for h in heads})==1 and all(h['parameters']==2691 and h['RNG_restored'] for h in heads)
 for arm in ARMS:
  supervision=read(run/arm/'SUPERVISION.json')
  if arm=='plain':assert supervision==[]
  else:
   assert len(supervision)==69 and {x['id_sha256']:x['label'] for x in supervision}==type_maps[arm]
   assert all(math.isfinite(x['answer_CE']) and math.isfinite(x['auxiliary_CE']) for x in supervision)
  reload=read(run/arm/'PREFIX_RELOAD.json');assert reload['digest']==prefix['model_digest']
  assert reload['checkpoint']==prefix['checkpoint'] and reload['exact_full_graph']
  trained=receipts[arm,'train']['checkpoint'];evaluated=receipts[arm,'eval']
  assert evaluated['reloaded']['digest']==trained['digest']
  assert evaluated['reloaded']['layout']==trained['layout'] and evaluated['complete_state_and_buffers_unchanged']
 raw=(root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py').read_bytes()
 assert hashlib.sha256(raw).hexdigest()=='4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240'
 nodes=[n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name in ('tokens','score')]
 assert len(nodes)==2
 ns=dict(re=re,Counter=Counter);exec(compile(ast.Module(body=nodes,type_ignores=[]),'<original-score>','exec'),ns)
 pending={}
 for arm in ('prefix',)+ARMS:
  for phase in (('before','acquired') if arm=='prefix' else ('before','acquired','final')):
   records=read((prefix_run if arm=='prefix' else run/arm)/(phase+'_predictions.json'))
   eligible={key for key in rows if (key[0]<=15 if arm=='prefix' else (True if phase=='final' else key[0]>=16))}
   assert len(records)==len(eligible) and {(x['task'],x['id']) for x in records}==eligible
   for x in records:
    original=rows[x['task'],x['id']];task=tasks[x['task']]
    assert x['references']==original['answers'] and x['domain']==task['domain']
    assert x['generation_cap']==task['max_new_tokens'] and isinstance(x['prediction'],str)
   pending[arm,phase]=records
 assert sum(map(len,pending.values()))==660
 before16=[{x['id']:(x['prediction'],x['output_tokens']) for x in pending[arm,'before'] if x['task']==16} for arm in ARMS]
 assert before16[0]==before16[1]==before16[2], 'Common-prefix readout mismatch'
 # Claim only after structural validation; failed/partial scoring is never retried automatically.
 write(run/'SCORE_CLAIM.json',dict(expected_unique_scores=660,verifier_sha256=sha(Path(__file__)),reused_prefix_sha256=sha(run/'REUSED_PREFIX.json')))
 scores={};summary={}
 for (arm,phase),records in pending.items():
  out=[]
  for x in records:
   value=ns['score'](x['prediction'],x['references'])['token_f1'];assert math.isfinite(value)
   out.append(dict(id_sha256=hashlib.sha256(x['id'].encode()).hexdigest(),task=x['task'],domain=x['domain'],F1=value,
    question_type=rows[x['task'],x['id']].get('metadata',{}).get('question_type'),strict_single=bool(re.fullmatch('[A-D]',x['prediction']))))
  key=arm+':'+phase;scores[key]=out
  groups={d:[x for x in out if x['domain']==d] for d in sorted({x['domain'] for x in out})}
  summary[key]=dict(QA=len(out),domains={d:sum(x['F1'] for x in rr)/len(rr) for d,rr in groups.items()},
   finance_by_type={typ:dict(QA=len(rr),F1=sum(x['F1'] for x in rr)/len(rr),strict_single=sum(x['strict_single'] for x in rr))
    for typ in ('single_choice','multiple_choice','numerical') if (rr:=[x for x in out if x['domain']=='finance' and x['question_type']==typ])})
  if phase=='final':
   assert len(groups)==5 and all(len(rr)==20 for rr in groups.values())
   summary[key]['five_domain_macro_F1']=sum(summary[key]['domains'].values())/5
 write(run/'SCORES.json',scores)
 write(run/'OUTCOME.json',dict(summary=summary,unique_scores=660,formal_result=False,
  new_generations=555,new_optimizer_updates=54,new_FIT_backwards=762,reused_prefix=reused,
  reused_unscored_prefix_answer_objects=120,new_answer_objects=540,
  final_costs={arm:dict(checkpoint=receipts[arm,'train']['checkpoint'],memory=read(run/arm/'memory.json')) for arm in ARMS},
  limitations=['single seed exploratory DEV','only four multi-choice DEV items','extra training metadata supervision','not proof of routing or type-conflict cause']))
 write(run/'VERIFIED.json',dict(scores_sha256=sha(run/'SCORES.json'),outcome_sha256=sha(run/'OUTCOME.json'),unique_scores=660,
  reused_prefix_sha256=sha(run/'REUSED_PREFIX.json'),supervisor_complete_sha256=sha(run/'SUPERVISOR_COMPLETE.json'),
  verifier_sha256=sha(Path(__file__)),new_model_phases=6))

if __name__=='__main__':main()
