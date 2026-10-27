"""First Qwen S15/S16 type comparison of sealed 1%/10% outputs; no rescoring."""
from pathlib import Path
import hashlib,json,math,time
from collections import Counter
from finance_strong_baseline_acquisition_20261027 import summarize,compare,question_type,lines,f1

ROOT=Path('/data1/liyapeng/program/LLMCL')
OUTPUT=ROOT/'summary/train_answer_prior_20261025/QWEN_FINANCE_TYPES_20261027.json'
PROOFS={
 'pct01':(ROOT,'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/completed/2a0b6d43f89c3cec6cb9.json','ff979a21b171372834940e3b68bdcaa84c7db4cdf9eaccb8939fa4b3c84bd59a'),
 'pct10':(ROOT/'exp/two_backbone_pct10_20261025/private_old_archive','summary/two_backbone_pct10_20261025/completion_observer_native/completed/dca76f5e9c3b31a8fc93.json','23e5d338cdc1579b6f8ca4f72d511f32e23896e248bfaf91e0ae775beb257081')}
MANIFESTS={
 'pct01':('data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json','36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777',1313),
 'pct10':('data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json','02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7',13105)}
TYPES=('single_choice','multiple_choice','numerical','UNKNOWN')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def relative(root,name):
 p=Path(name);assert not p.is_absolute() and '..' not in p.parts
 return root/p

def main():
 assert not OUTPUT.exists()
 helper=Path(__file__).with_name('finance_strong_baseline_acquisition_20261027.py')
 assert sha(helper)=='fd306888772f88fa53127c53047770598a842b3f205f42c2fc4c433c5350844d'
 ledger={}
 def read(p,expected):
  raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected
  ledger[str(p)]=dict(sha256=expected,bytes=len(raw));return raw
 data={};originals=None;bindings={}
 for cond,(where,name,proofsha) in PROOFS.items():
  proof=json.loads(read(where/name,proofsha));candidate=proof['candidate']
  assert candidate['backbone']=='qwen' and candidate['method']=='SeqFT_LoRA' and candidate.get('condition','pct01')==cond
  assert proof['rescored_cells']==74 and proof['final_QA']==21454 and proof['positive_finite_training_stages']==19
  run=candidate['run'];configrel=run+'/config.json'
  assert proof['source_sha256'][configrel]==proof['config_sha256']
  config=json.loads(read(relative(where,configrel),proof['config_sha256']))
  assert config['backbone']=='qwen' and config['method']=='SeqFT_LoRA' and config['condition']==cond
  assert config['seed']==42 and config['epochs']==1 and config['test_fraction']==1 and not config['preflight_only']
  manifestrel,manifestsha,total=MANIFESTS[cond]
  assert config['manifest_sha256']==proof['source_sha256'][manifestrel]==manifestsha
  manifest=json.loads(read(ROOT/manifestrel,manifestsha))
  assert len(manifest['tasks'])==19 and sum(t['files']['train']['count'] for t in manifest['tasks'])==total
  task=manifest['tasks'][15];assert task['task']==16 and task['domain']=='finance'
  test=task['files']['test'];assert test['count']==2233 and proof['source_sha256'][test['path']]==test['sha256']
  rows=lines(read(relative(ROOT,test['path']),test['sha256']))
  ids=[x['id'] for x in rows];assert len(ids)==len(set(ids))==2233
  if originals is None:originals=rows
  assert rows==originals
  data[cond]={}
  for stage in (15,16):
   suffix=f'evaluations/stage_{stage:02d}/T16_test_predictions.jsonl';rel=run+'/'+suffix
   expected=proof['prediction_sha256'][suffix];assert proof['source_sha256'][rel]==expected
   predictions=lines(read(relative(where,rel),expected));assert [x['id'] for x in predictions]==ids
   for x,y in zip(predictions,rows):
    assert x['references']==y['answers'] and x.get('metadata',{})==y.get('metadata',{})
    assert isinstance(x['prediction'],str) and x['generation_cap']==task['max_new_tokens'];f1(x)
   data[cond][stage]=predictions
  bindings[cond]=dict(proof_sha256=proofsha,run=run,config_sha256=proof['config_sha256'],manifest_sha256=manifestsha)
 groups={t:[i for i,x in enumerate(originals) if t=='ALL' or question_type(x)==t] for t in ('ALL',)+TYPES}
 assert sum(len(groups[t]) for t in TYPES)==2233
 summary={};acquisition={};cross={}
 for cond in data:
  summary[cond]={str(s):{g:summarize([data[cond][s][i] for i in indices]) for g,indices in groups.items()} for s in (15,16)}
  acquisition[cond]={g:compare([data[cond][15][i] for i in indices],[data[cond][16][i] for i in indices]) for g,indices in groups.items()}
 for s in (15,16):
  cross[str(s)]={g:compare([data['pct01'][s][i] for i in indices],[data['pct10'][s][i] for i in indices]) for g,indices in groups.items()}
 result=dict(complete=True,unix=time.time(),consumer_sha256=sha(Path(__file__)),helper_sha256=sha(helper),bindings=bindings,source_files=ledger,
  question_type_denominators={g:len(v) for g,v in groups.items()},summary=summary,acquisition=acquisition,cross_ratio=cross,
  existing_F1_values_read=8932,new_scores=0,new_models=0,new_generations=0,checkpoint_reads=0,raw_answers_in_output=False,
  limitations=['Retrospective sealed TEST description; no method selection or change to frozen type-aux study.',
   'Same task before/acquired are different weights; ratios change data, updates, curriculum and historical hardware jointly.',
   'Original token F1 is not set accuracy, numerical equivalence or semantic correctness.',
   'Strict single means raw fullmatch A-D, without stripping; other outputs are not necessarily correct.',
   'Qwen results restrict cross-backbone generalization; they do not identify routing, capacity or type-conflict causality.'])
 with OUTPUT.open('x',encoding='utf8') as h:json.dump(result,h,indent=2,allow_nan=False);h.write('\n')
 print(json.dumps(dict(output=str(OUTPUT),sha256=sha(OUTPUT),summary=summary,new_scores=0)))
if __name__=='__main__':main()
