"""Read-only SAPT routing audit, using exact frozen fused prompt pooling."""
import csv,hashlib,io,json,sys
from pathlib import Path
import torch
from safetensors import safe_open
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'exp'),str(ROOT/'code')]
from sapt_lora import SharedAttention
from cv_data import feature_path
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
DATA=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(4)
with safe_open(ROOT/'model/Qwen2-VL-2B-Instruct/model-00001-of-00002.safetensors',framework='pt',device='cpu') as f:
 embedding=f.get_tensor('model.embed_tokens.weight').cuda()
image_token=151655
assert image_token==json.loads((ROOT/'model/Qwen2-VL-2B-Instruct/config.json').read_text())['image_token_id']
pools={};routing=[];items=[]
with torch.inference_mode():
 for task,sp in DATA.items():
  for split in ['train','test']:
   if split=='train' and not task.startswith('medical'):continue
   rows=sp[split];pooled=[]
   for r in rows:
    ids=torch.tensor(r['prompt_ids'],device='cuda');x=embedding[ids].float()
    if r.get('image'):
     features=torch.load(feature_path(r),map_location='cpu',weights_only=True)['features'].cuda().float()
     x[ids==image_token]=features
    pooled.append(x.amax(0))
   pools[task,split]=torch.stack(pooled)
 for stage in range(1,10):
  state=torch.load(RUN/'sapt_lora/checkpoints'/f'stage_{stage:02d}'/'completed.pt',map_location='cpu',weights_only=False)
  state=torch.load(io.BytesIO(state['method']),map_location='cpu',weights_only=True)
  router=SharedAttention(1536,100,'cuda')
  for i in range(stage):router.keys.append(torch.nn.Parameter(state['router'][f'keys.{i}'].cuda()))
  router.load_state_dict(state['router']);router.eval()
  for (task,split),pool in pools.items():
   weights=router(pool).softmax(-1).cpu();j=list(DATA).index(task)
   rows=DATA[task][split];means=weights.mean(0).tolist()
   correcttask=j if j<stage else None
   routing.append({'stage':stage,'task':task,'split':split,'n':len(rows),'current_adapter_weight':means[-1],
    'task_adapter_weight':means[correcttask] if correcttask is not None else None,'top_adapter':int(weights.mean(0).argmax())+1,
    'entropy_mean':float(-(weights*weights.clamp_min(1e-30).log()).sum(-1).mean()),
    **{'adapter_'+str(i+1):means[i] if i<stage else None for i in range(9)}})
   if stage in [6,7,9] and task.startswith('medical') and split=='test':
    items.extend({'stage':stage,'task':task,'instance_id':r['instance_id'],'weights':w.tolist()} for r,w in zip(rows,weights))
with (OUT/'routing_statistics.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(routing[0]),lineterminator='\n');w.writeheader();w.writerows(routing)
(OUT/'routing_items.json').write_text(json.dumps(items,indent=2)+'\n')
(OUT/'routing_protocol.json').write_text(json.dumps({'pool':'exact frozen fused embedding elementwise max; identical to exp/cv_runtime.py prompt_pools',
 'checkpoint_stages':list(range(1,10)),'no_training':True,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Routing audit complete.',flush=True)
