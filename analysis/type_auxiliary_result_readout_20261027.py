"""Explain verified type-auxiliary results without generation or rescoring.

Reuses the common 60-item prefix once and each arm's 40-item suffix to report
100-item acquisition. This is exploratory DEV, never a formal main-table row.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

ARMS=('plain','true','shuffled')
DOMAINS=('natural','medical','geography','finance','traffic')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def key(x):return x['task'],x['id_sha256']
def mean(rows):return sum(x['F1'] for x in rows)/len(rows) if rows else None

def paired(left,right):
 a={key(x):x for x in left};b={key(x):x for x in right}
 assert len(a)==len(left) and len(b)==len(right) and set(a)==set(b)
 changes=[dict(task=k[0],id_sha256=k[1],before=a[k]['F1'],after=b[k]['F1'],delta=b[k]['F1']-a[k]['F1']) for k in sorted(a)]
 positive=[x for x in changes if x['delta']>0];negative=[x for x in changes if x['delta']<0]
 return dict(QA=len(changes),mean_delta=(mean(right)-mean(left)) if changes else None,
  higher=len(positive),lower=len(negative),tie=len(changes)-len(positive)-len(negative),
  gross_gain=sum(x['delta'] for x in positive),gross_loss=-sum(x['delta'] for x in negative),
  changed=[x for x in changes if x['delta']!=0])

def summarize(rows):
 groups={d:[x for x in rows if x['domain']==d] for d in DOMAINS}
 assert len(rows)==100 and all(len(v)==20 for v in groups.values())
 return dict(five_domain_macro_F1=sum(mean(v) for v in groups.values())/5,
             domain_F1={d:mean(v) for d,v in groups.items()})

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert a.run.is_absolute() and a.output.is_absolute() and not a.output.exists()
 v=read(a.run/'VERIFIED.json');assert v['unique_scores']==660
 assert sha(a.run/'SCORES.json')==v['scores_sha256'] and sha(a.run/'OUTCOME.json')==v['outcome_sha256']
 claim=read(a.run/'SCORE_CLAIM.json');assert claim['expected_unique_scores']==660
 assert claim['verifier_sha256']=='1caa7e611ac1b5e970dafebd622cf13046501830ce10d3a9662d7f7b9b57c750'
 scores=read(a.run/'SCORES.json');outcome=read(a.run/'OUTCOME.json')
 expected={'prefix:before','prefix:acquired'}|{arm+':'+p for arm in ARMS for p in ('before','acquired','final')}
 assert set(scores)==expected and sum(map(len,scores.values()))==660
 assert all(math.isfinite(x['F1']) and 0<=x['F1']<=100 for rows in scores.values() for x in rows)
 combined={}
 for arm in ARMS:
  for phase in ('before','acquired'):
   assert len(scores['prefix:'+phase])==60 and len(scores[arm+':'+phase])==40
   combined[arm,phase]=scores['prefix:'+phase]+scores[arm+':'+phase]
  combined[arm,'final']=scores[arm+':final']
 summary={arm:{p:summarize(combined[arm,p]) for p in ('before','acquired','final')} for arm in ARMS}
 comparisons={}
 for control,treatment in (('plain','true'),('shuffled','true'),('plain','shuffled')):
  name=control+'_to_'+treatment
  comparisons[name]={}
  for phase in ('acquired','final'):
   left,right=combined[control,phase],combined[treatment,phase]
   comparisons[name][phase]=dict(all_DEV=paired(left,right),
    by_domain={d:paired([x for x in left if x['domain']==d],[x for x in right if x['domain']==d]) for d in DOMAINS},
    finance_by_type={t:paired([x for x in left if x['domain']=='finance' and x['question_type']==t],[x for x in right if x['domain']=='finance' and x['question_type']==t]) for t in ('single_choice','multiple_choice','numerical')})
 finance={}
 for arm in ARMS:
  before=[x for x in combined[arm,'before'] if x['task']==16]
  acquired=[x for x in combined[arm,'acquired'] if x['task']==16]
  assert Counter(x['question_type'] for x in acquired)=={'single_choice':13,'multiple_choice':4,'numerical':3}
  finance[arm]=dict(new_acquisition_change=paired(before,acquired),
   strict_single_by_type={t:dict(QA=len(rr),strict_single=sum(x['strict_single'] for x in rr),original_F1=mean(rr))
    for t in ('single_choice','multiple_choice','numerical') if (rr:=[x for x in acquired if x['question_type']==t])})
 costs={}
 for arm in ARMS:
  c=outcome['final_costs'][arm];windows=c['memory'];assert len(windows)==5
  weights=c['checkpoint']['layout']['weights']
  costs[arm]=dict(coefficients=weights['coefficients'],weight_MiB=weights['bytes']/1024**2,
   peak_allocated_MiB=max(w['peak_allocated'] for w in windows)/1024**2,
   peak_reserved_MiB=max(w['peak_reserved'] for w in windows)/1024**2,
   checkpoint_bytes=c['checkpoint']['bytes'],checkpoint_sha256=c['checkpoint']['sha256'])
 result=dict(verified_sha256=sha(a.run/'VERIFIED.json'),scores_sha256=v['scores_sha256'],outcome_sha256=v['outcome_sha256'],
  consumer_sha256=sha(Path(__file__)),summary=summary,comparisons=comparisons,finance=finance,costs=costs,
  new_scores=0,new_models=0,formal_main_table_result=False,
  limits=['Only four multi-choice DEV items; report concentration, not significance.',
         'Common prefix reused rather than rescored or counted as independent arms.',
         'Before/acquired aggregates join task-specific times, not one model snapshot.',
         'Deployment costs exclude training peaks, training time and auxiliary optimizer state.',
         'True only beating shuffled does not establish improvement over plain.',
         'Type labels are extra training supervision; no causal proof of type conflict.',
         'Fixed five-window GPU peaks are not full-TEST maxima.'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(output=str(a.output),sha256=sha(a.output),new_scores=0)))

if __name__=='__main__':main()
