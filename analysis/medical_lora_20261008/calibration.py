"""Explicit diagnostic rescoring; preserve original experiment metrics."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'exp'))
from common import normalize_answer
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
NUMBER={'zero':'0','one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9','ten':'10'}
ALIAS={'right hemisphere':'right','right cerebral hemisphere':'right','left hemisphere':'left','left cerebral hemisphere':'left',
 'posterioranterior':'pa','posteroanterior':'pa','posterior anterior':'pa','anteriorposterior':'ap','anteroposterior':'ap','anterior posterior':'ap'}
def canon(text):
 t=normalize_answer(text);return ALIAS.get(t,NUMBER.get(t,t))
rows=[];rescued=[]
for method in METHODS:
 for stage in [0,9]:
  path=RUN/method/'predictions'/f'stage_{stage:02d}'/'medical_vqa_rad.jsonl';raw=[json.loads(l) for l in path.read_text().splitlines()]
  for group in ['all','OPEN','CLOSED']:
   chosen=[r for r in raw if group=='all' or r['answer_type']==group]
   original=sum(r['scores']['exact_match'] for r in chosen)/len(chosen)
   corrected=100*sum(any(canon(r['prediction'])==canon(ref) for ref in r['references']) for r in chosen)/len(chosen)
   rows.append({'method':method,'stage':stage,'task':'medical_vqa_rad','group':group,'n':len(chosen),'original_EM':original,'explicit_alias_EM':corrected,'consensus_diagnostic':None})
   if group=='all':
    for r in chosen:
     if r['scores']['exact_match']==0 and any(canon(r['prediction'])==canon(ref) for ref in r['references']):rescued.append({'method':method,'stage':stage,'instance_id':r['instance_id'],'prediction':r['prediction'],'references':r['references']})
  raw=[json.loads(l) for l in (RUN/method/'predictions'/f'stage_{stage:02d}'/'natural_vqav2.jsonl').read_text().splitlines()]
  vals=[]
  for r in raw:
   pred=normalize_answer(r['prediction']);refs=[normalize_answer(x) for x in r['references']];assert len(refs)==10
   vals.append(sum(min(1,sum(x==pred for j,x in enumerate(refs) if j!=i)/3) for i in range(10))/10)
  rows.append({'method':method,'stage':stage,'task':'natural_vqav2','group':'all','n':200,'original_EM':sum(r['scores']['exact_match'] for r in raw)/200,
   'explicit_alias_EM':None,'consensus_diagnostic':100*sum(vals)/200})
with (OUT/'score_calibration.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'alias_rescued_cases.json').write_text(json.dumps(rescued,indent=2)+'\n')
(OUT/'calibration_protocol.json').write_text(json.dumps({'numbers':NUMBER,'aliases':ALIAS,
 'medical_alias_scope':'whole-answer explicit number/laterality/projection equivalence only; no model/LLM judge, not a validated clinical score',
 'natural_consensus_scope':'leave-one-annotator-out agreement divided by3 and capped at1; repository normalization preserved, not official VQA normalization or benchmark score'},indent=2)+'\n')
print('Diagnostic calibration complete.')
