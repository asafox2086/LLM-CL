"""Frozen finance TRAIN input cues only; no TEST, model, scorer or label use."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

POOLS = {
 'pct01': ('data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json', '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777',89),
 'pct10': ('data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json','02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7',887)}
TYPES=('single_choice','multiple_choice','numerical','UNKNOWN')
PATTERNS={
 'explicit_single': r'单选|单项选择|\bsingle[ -]choice\b|\bchoose\s+one\b',
 'explicit_multi': r'多选|多项选择|不定项|\bselect\s+all\b',
 'ambiguous_multiple_choice':r'\bmultiple[ -]choice\b'}

def sha(raw): return hashlib.sha256(raw).hexdigest()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
def read_bound(path, expected, ledger):
 raw=path.read_bytes(); assert sha(raw)==expected,str(path)
 ledger[str(path)]=dict(sha256=sha(raw),bytes=len(raw));return raw
def flags(text):return {k:bool(re.search(p,text,re.I)) for k,p in PATTERNS.items()}
def stats(items):
 return dict(QA=len(items),flags={k:sum(x[k] for x in items) for k in PATTERNS},
  only_explicit_single=sum(x['explicit_single'] and not x['explicit_multi'] for x in items),
  only_explicit_multi=sum(x['explicit_multi'] and not x['explicit_single'] for x in items),
  both_explicit_cues=sum(x['explicit_single'] and x['explicit_multi'] for x in items),
  neither_explicit_cue=sum(not x['explicit_single'] and not x['explicit_multi'] for x in items))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
 assert a.root.is_absolute() and a.output.is_absolute() and not a.output.exists()
 root=a.root;ledger={};result={}
 source=root/'exp/two_backbone_native_cl_20261017_v2/runtime_utils.py'
 raw=read_bound(source,'4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240',ledger)
 node=next(n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name=='input_batch')
 segment=ast.get_source_segment(raw.decode(),node)
 assert "text=task['instruction']" in segment and "text+='\\nQuestion: '+question" in segment
 assert 'metadata' not in segment
 for pool,(relative,expected,count) in POOLS.items():
  manifest=json.loads(read_bound(root/relative,expected,ledger));task=manifest['tasks'][15]
  assert task['task']==16 and task['domain']=='finance' and manifest['condition']==pool
  f=task['files']['train'];assert f['count']==count and Path(f['path']).name=='T16_train.jsonl'
  rel=Path(f['path']);assert not rel.is_absolute() and '..' not in rel.parts
  rows=[json.loads(line) for line in read_bound(root/rel,f['sha256'],ledger).splitlines()]
  assert len(rows)==len({row['id'] for row in rows})==count
  groups={t:[] for t in TYPES};details=[]
  for row in rows:
   q=row['question'];c=row.get('context','');assert isinstance(q,str) and isinstance(c,str)
   assert len(c)<=6000  # No hidden retrieval change in this text-only reconstruction.
   typ=row.get('metadata',{}).get('question_type');typ=typ if typ in TYPES[:3] else 'UNKNOWN'
   q=q.replace('<image>','')
   text=task['instruction']+('\nContext:\n'+c if c else '')+'\nQuestion: '+q
   entry=dict(id_sha256=sha(row['id'].encode()),question=flags(q),context=flags(c),constructed_text=flags(text))
   groups[typ].append(entry);details.append(entry)
  result[pool]=dict(QA=count,instruction_sha256=sha(task['instruction'].encode()),instruction_flags=flags(task['instruction']),
   type_counts={t:len(x) for t,x in groups.items()},by_type={t:{field:stats([x[field] for x in group]) for field in ('question','context','constructed_text')} for t,group in groups.items()},
   aggregate={field:stats([x[field] for x in details]) for field in ('question','context','constructed_text')},
   per_ID_flags_sha256=sha(canonical(details)))
 out=dict(complete=True,pools=result,patterns=PATTERNS,source_files=ledger,consumer_sha256=sha(Path(__file__).read_bytes()),
  metadata_forwarded_by_input_batch=False,answer_fields_used=False,TEST_files_read=0,new_models=0,new_scores=0,
  limits=['Cue co-occurrence is not a semantic contradiction; generic instruction may enumerate multiple modes.',
   'Multiple choice is ambiguous and not an explicit multi-answer requirement.',
   'No numeric cue dictionary: numerical rows are stratified observations only.',
   'Missing literal cues do not imply semantic ambiguity or explain model failure.',
   'Constructed text before chat template/tokenization, not historical token replay; no classifier fitted.'])
 a.output.parent.mkdir(parents=True,exist_ok=True)
 with a.output.open('x',encoding='utf8') as stream:json.dump(out,stream,ensure_ascii=False,indent=2);stream.write('\n')
 print(json.dumps(dict(output=str(a.output),sha256=sha(a.output.read_bytes()),pools=result)))
if __name__=='__main__':main()
