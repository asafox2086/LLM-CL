from pathlib import Path
import json,hashlib,time
root=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028');r=root/'exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1'
ledger={}
def read(p):
 b=p.read_bytes();h=hashlib.sha256(b).hexdigest();ledger[str(p.relative_to(root))]={'sha256':h,'json':json.loads(b)};return json.loads(b)
cfg=read(r/'config.json');man=read(root/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
assert cfg['manifest_sha256']=='36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
rows=[]
for t in range(1,14):
 stage=read(r/f'stages/stage_{t:02d}.json');cp=read(r/f'checkpoints/stage_{t:02d}.json')
 assert stage['complete'] and stage['weights_saved'] and cp['stage']==t and cp['method']=='TaskGradReuse' and cp['manifest_sha256']==cfg['manifest_sha256']
 assert (r/cp['path']).stat().st_size==cp['bytes']
 task=man['tasks'][t-1]; scores={}
 for label,s in [('initial',0),('before',t-1),('acquired',t)]:
  d=read(r/f'evaluations/stage_{s:02d}/T{t:02d}_test_predictions.scores.json');assert d['count']==task['files']['test']['count'];scores[label]=d['token_f1']
 rows.append(dict(task=t,task_id=task['task_id'],domain=task['domain'],native_label=task['native_label'],test_qa=task['files']['test']['count'],train_qa=task['files']['train']['count'],**scores,gain=scores['acquired']-scores['before']))
for rel,d in ledger.items():assert hashlib.sha256((root/rel).read_bytes()).hexdigest()==d['sha256']
print(json.dumps(dict(status='INTERIM_COMMITTED_STAGES_NOT_FINAL_VERIFIED',through_stage=13,unix=time.time(),rows=rows,sources=ledger,new_scores=0,prediction_rows_read=0,checkpoint_payload_reads=0)))
