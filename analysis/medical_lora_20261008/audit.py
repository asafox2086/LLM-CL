"""Offline diagnostic audit of fixed records, logs and adapter checkpoints."""
import csv,hashlib,io,json,re,sys
from collections import Counter
from pathlib import Path
from statistics import mean,median
import torch
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from common import Scorer,normalize_answer
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
DATA=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
TASKS=json.loads((RUN/'seq_lora/config.json').read_text())['tasks']
torch.set_num_threads(4)
def write(name,data): (OUT/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def csvout(name,rows):
 with (OUT/name).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def tokens(text):return re.findall(r"[a-z]+(?:'[a-z]+)?",text.lower())
def records(p):return [json.loads(line) for line in p.read_text().splitlines()]
scorer=Scorer();dataset=[]
for task in TASKS:
 for split,rows in DATA[task].items():
  refs=[r.get('training_answer',r['references'][0]) for r in rows]
  lens=[len(tokens(t)) for t in refs]
  counts=Counter(normalize_answer(t) for t in refs)
  dataset.append({'task':task,'split':split,'n':len(rows),'images':len({r['image'] for r in rows if r.get('image')}),'unique_targets':len(counts),
   'top_target':counts.most_common(1)[0][0],'top_target_share':counts.most_common(1)[0][1]/len(rows),
   'mean_prompt_tokens':mean(len(r['prompt_ids']) for r in rows),'mean_target_tokens':mean(len(r['input_ids'])-len(r['prompt_ids']) for r in rows),
   'mean_target_words':mean(lens),'median_target_words':median(lens),'prompt_truncated':sum(r['prompt_truncated'] for r in rows),'target_truncated':sum(r['target_truncated'] for r in rows),
   'references_both_yes_no':sum({'yes','no'}<={t.strip().lower() for t in r['references']} for r in rows),
   'references_disagree':sum(len(set(s.lower().strip() for s in r['references']))>1 for r in rows)})
csvout('dataset_audit.csv',dataset)
metricrows=[];classrows=[];changes=[];sources={}
for method in METHODS:
 for stage in range(10):
  for task in TASKS:
   p=RUN/method/'predictions'/f'stage_{stage:02d}'/(task+'.jsonl');rs=records(p);assert len(rs)==200
   sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
   texts=[r['prediction'] for r in rs];ps=Counter(normalize_answer(t) for t in texts)
   metricrows.append({'method':method,'stage':stage,'task':task,'rougeL':mean(r['scores']['rougeL'] for r in rs),
    'EM':mean(r['scores']['exact_match'] for r in rs),'token_f1':mean(r['scores']['token_f1'] for r in rs),
    'prediction_words':mean(len(tokens(t)) for t in texts),'reference_words':mean(len(tokens(r['references'][0])) for r in rs),
    'unique_predictions':len(ps),'top_prediction':ps.most_common(1)[0][0],'top_prediction_share':ps.most_common(1)[0][1]/200,
    'limit_hits':sum(r['hit_generation_limit'] for r in rs),'empty_predictions':sum(not t.strip() for t in texts),
    'negation_predictions':sum(bool(re.search(r'\b(no|not|without|negative|absent|denies|denied)\b',t,re.I)) for t in texts),
    'negation_references':sum(bool(re.search(r'\b(no|not|without|negative|absent|denies|denied)\b',r['references'][0],re.I)) for r in rs)})
   if task=='medical_vqa_rad':
    for group in ['all','CLOSED','OPEN','CHEST','HEAD','ABD']:
     selected=[r for r in rs if group=='all' or r.get('answer_type')==group or r.get('organ')==group]
     yn=[r for r in selected if r['references'][0].strip().lower() in ['yes','no']]
     classrows.append({'method':method,'stage':stage,'group':group,'n':len(selected),
      'EM':mean(r['scores']['exact_match'] for r in selected),'rougeL':mean(r['scores']['rougeL'] for r in selected),
      'yes_no_count':len(yn),'yes_references':sum(r['references'][0].strip().lower()=='yes' for r in yn),
      'predict_yes':sum(r['prediction'].strip().lower()=='yes' for r in yn),'predict_no':sum(r['prediction'].strip().lower()=='no' for r in yn),
      'yes_to_no':sum(r['references'][0].strip().lower()=='yes' and r['prediction'].strip().lower()=='no' for r in yn),
      'no_to_yes':sum(r['references'][0].strip().lower()=='no' and r['prediction'].strip().lower()=='yes' for r in yn)})
 for stage in [6,7,8,9]:
  for task in TASKS[:stage-1]:
   before=records(RUN/method/'predictions'/f'stage_{stage-1:02d}'/(task+'.jsonl'));after=records(RUN/method/'predictions'/f'stage_{stage:02d}'/(task+'.jsonl'))
   assert [r['instance_id'] for r in before]==[r['instance_id'] for r in after]
   for a,b in zip(before,after):
    if a['scores']['exact_match']==100 and b['scores']['exact_match']==0:
     changes.append({'method':method,'stage':stage,'task':task,'instance_id':a['instance_id'],'prompt':a['prompt'],
      'references':a['references'],'before':a['prediction'],'after':b['prediction'],'before_scores':a['scores'],'after_scores':b['scores']})
csvout('answer_statistics.csv',metricrows);csvout('medical_vqa_groups.csv',classrows);write('old_answer_regressions.json',changes)
logs=[]
for method in METHODS:
 for stage in range(1,10):
  base=RUN/method/'checkpoints'/f'stage_{stage:02d}'/'train';rs=records(base/'training.jsonl');assert len(rs)==63
  resources=json.loads((base/'resources.json').read_text())
  logs.append({'method':method,'stage':stage,'task':TASKS[stage-1],'updates':len(rs),'CE_first8':mean(r['answer_cross_entropy'] for r in rs[:8]),'CE_last8':mean(r['answer_cross_entropy'] for r in rs[-8:]),
   'CE_mean':mean(r['answer_cross_entropy'] for r in rs),'penalty_mean':mean(r['regularization_loss'] for r in rs),'grad_norm_mean':mean(r['gradient_norm'] for r in rs),
   'grad_clipped_fraction':mean(r['gradient_norm']>1 for r in rs),'amp_scale_first':rs[0]['amp_scale'],'amp_scale_last':rs[-1]['amp_scale'],
   'seconds':resources['seconds'],'trainable':resources['current_trainable_parameters']})
csvout('training_diagnostics.csv',logs)
# Gauge-invariant matrix diagnostics; do not compare A/B factors as if they were delta W.
def state(method,stage):
 p=RUN/method/'checkpoints'/f'stage_{stage:02d}'/'completed.pt'
 x=torch.load(p,map_location='cpu',weights_only=False)
 return torch.load(io.BytesIO(x['method']),map_location='cpu',weights_only=True)
def factor(layer,index):return layer[f'{index}.up.weight'].double()*4,layer[f'{index}.down.weight'].double()
def dot(x,y):return float(((x[0].T@y[0])*(x[1]@y[1].T)).sum())
def norm(x):return max(0,dot(x,x))**.5

def difference(x,y):return torch.cat([x[0],-y[0]],1),torch.cat([x[1],y[1]],0)
weightrows=[];aggregates=[]
for method in METHODS:
 prev=None;prev_delta={}
 for stage in range(1,10):
  current=state(method,stage);total_new2=total_old2=total_dot=0
  nextdelta={}
  for name,layer in current['layers'].items():
   if method in ['seq_lora','migu_lora']:
    x=factor(layer,0);old=factor(prev['layers'][name],0) if prev else (torch.zeros_like(x[0]),x[1])
    delta=difference(x,old);kind='change_of_shared_adapter'
   else:
    delta=factor(layer,stage-1);kind='new_adapter_unrouted'
    oldcount=stage-1
    old=(torch.cat([factor(layer,i)[0] for i in range(oldcount)],1),torch.cat([factor(layer,i)[1] for i in range(oldcount)],0)) if oldcount else (torch.zeros_like(delta[0]),delta[1])
   nn=norm(delta);oo=norm(old);dd=dot(delta,old);total_new2+=nn**2;total_old2+=oo**2;total_dot+=dd
   qb,rb=torch.linalg.qr(delta[0],mode='reduced');qa,ra=torch.linalg.qr(delta[1].T,mode='reduced')
   sv=torch.linalg.svdvals(rb@ra.T);energy=(sv**2);rank1=float(energy[0]/energy.sum()) if float(energy.sum()) else None
   cosprev=None
   if name in prev_delta:
    nprev=norm(prev_delta[name]);cosprev=dot(delta,prev_delta[name])/(nn*nprev) if nn*nprev else None
   weightrows.append({'method':method,'stage':stage,'task':TASKS[stage-1],'module':name,'diagnostic':kind,'delta_frobenius':nn,
    'old_delta_frobenius':oo,'cosine_with_old_adapter_sum':dd/(nn*oo) if nn*oo else None,
    'cosine_with_previous_task_update':cosprev,'rank1_energy_fraction':rank1,'rank4_energy_fraction':float(energy[:4].sum()/energy.sum()) if float(energy.sum()) else None})
   nextdelta[name]=delta
  aggregates.append({'method':method,'stage':stage,'task':TASKS[stage-1],'diagnostic':kind,'all_delta_frobenius':total_new2**.5,
   'old_adapter_sum_frobenius':total_old2**.5,'cosine_with_old_adapter_sum':total_dot/(total_new2*total_old2)**.5 if total_new2*total_old2 else None})
  prev=current;prev_delta=nextdelta
csvout('adapter_layer_diagnostics.csv',weightrows);csvout('adapter_stage_diagnostics.csv',aggregates)
write('provenance.json',{'source_run':str(RUN),'raw_prediction_files_sha256':sources,'prediction_records':72000,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 'scope':'offline observational diagnostics; lexical negation is not clinical correctness; SAPT adapter norms exclude routing and are not effective update norms'})
print('Offline audit completed.',flush=True)
