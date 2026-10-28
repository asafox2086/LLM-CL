from pathlib import Path
from collections import Counter
import json,hashlib
r=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028/exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1')
paths=[r/f'evaluations/stage_{s:02d}/T07_test_predictions.jsonl' for s in [6,7]]
raw=[p.read_bytes() for p in paths];rows=[[json.loads(line) for line in b.splitlines()] for b in raw]
a,b=rows;assert len(a)==len(b)==89
assert all((x['id'],x['dataset'],x['references'])==(y['id'],y['dataset'],y['references']) for x,y in zip(a,b))
c=Counter();diff=Counter();trans=Counter()
for x,y in zip(a,b):
 delta=y['scores']['token_f1']-x['scores']['token_f1'];c['improved' if delta>0 else 'worsened' if delta<0 else 'unchanged']+=1
 if x['scores']['token_f1']==100 and y['scores']['token_f1']<100:c['lost_perfect']+=1
 if x['scores']['token_f1']<100 and y['scores']['token_f1']==100:c['new_perfect']+=1
 if delta:trans[(x['prediction'],y['prediction'])]+=1
 diff['positive_sum' if delta>0 else 'negative_sum']+=delta
for p,h in zip(paths,raw):assert p.read_bytes()==h
result=dict(task=7,QA=89,counts=dict(c),F1_delta_sums=dict(diff),before_F1=sum(x['scores']['token_f1'] for x in a)/89,after_F1=sum(x['scores']['token_f1'] for x in b)/89,before_prediction_counts=dict(Counter(x['prediction'] for x in a)),after_prediction_counts=dict(Counter(x['prediction'] for x in b)),score_changing_transitions=[dict(before=k[0],after=k[1],count=v) for k,v in trans.most_common()],before_tokens=dict(Counter(x['output_tokens'] for x in a)),after_tokens=dict(Counter(x['output_tokens'] for x in b)),source_sha256={str(p):hashlib.sha256(h).hexdigest() for p,h in zip(paths,raw)},new_scores=0,existing_QA_rows_read=178)
print(json.dumps(result,ensure_ascii=False))
