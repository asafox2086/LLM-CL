"""Read-only teacher-forced gradient conflict and finite update audit."""
import csv,hashlib,io,json,sys,time
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from cv_data import MODEL,processor
from cv_runtime import answer_loss,batch_tensors,install_cached_vision
from run_olora import create_method
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
config=json.loads((RUN/'seq_lora/config.json').read_text())
data=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(4);torch.manual_seed(42)
model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
install_cached_vision(model);method=create_method(model,config);tok=processor().tokenizer
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
ce=[];vectors={};parameter_vectors={};ids={}
for stage in [5,6,7,8,9]:
 state=torch.load(RUN/'seq_lora/checkpoints'/f'stage_{stage:02d}'/'completed.pt',map_location='cpu',weights_only=False)
 method.load(io.BytesIO(state['method']));params=method.parameters()
 parameter_vectors[stage]=torch.cat([p.detach().cpu().float().flatten() for p in params])
 model.train()
 for module in model.modules():
  if isinstance(module,torch.nn.Dropout):module.eval()
 for task in config['tasks']:
  # Fixed hash order, 8 held-out examples each; no training or selection.
  rows=sorted(data[task]['test'],key=lambda r:hashlib.sha256(('medical-gradient42\n'+r['instance_id']).encode()).hexdigest())[:8]
  ids[task]=[r['instance_id'] for r in rows];model.zero_grad(set_to_none=True);losses=[]
  for row in rows:
   batch=batch_tensors([row],tok,model.device,True)
   with torch.autocast('cuda',dtype=torch.float16):loss=answer_loss(model,batch)
   losses.append(float(loss.detach()));(loss*(1024/len(rows))).backward()
  g=torch.cat([p.grad.detach().cpu().float().flatten()/1024 for p in params]);assert torch.isfinite(g).all()
  vectors[stage,task]=g
  ce.append({'stage':stage,'task':task,'n':8,'teacher_forced_CE':sum(losses)/8,'gradient_norm':float(g.norm())})
  print(json.dumps(ce[-1]),flush=True)
conflicts=[];projections=[]
for stage in [5,6,7,8,9]:
 for a in config['tasks']:
  for b in config['tasks']:
   ga=vectors[stage,a];gb=vectors[stage,b];den=float(ga.norm()*gb.norm())
   conflicts.append({'stage':stage,'task_a':a,'task_b':b,'gradient_cosine':float(ga@gb)/den if den else None})
 if stage<9:
  delta=parameter_vectors[stage+1]-parameter_vectors[stage]
  for task in config['tasks']:
   g=vectors[stage,task];before=next(r['teacher_forced_CE'] for r in ce if r['stage']==stage and r['task']==task)
   after=next(r['teacher_forced_CE'] for r in ce if r['stage']==stage+1 and r['task']==task)
   projections.append({'from_stage':stage,'to_stage':stage+1,'task':task,'first_order_CE_change':float(g@delta),'observed_CE_change':after-before,
    'update_gradient_cosine':float(g@delta)/float(g.norm()*delta.norm()) if float(g.norm()*delta.norm()) else None})
for filename,rows in [('gradient_CE.csv',ce),('gradient_conflicts.csv',conflicts),('update_gradient_projections.csv',projections)]:
 with (OUT/filename).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
(OUT/'gradient_protocol.json').write_text(json.dumps({'method':'SeqLoRA shared adapter','stages':[5,6,7,8,9],'ids':ids,
 'loss':'exact exp/cv_runtime.py answer_loss; mean per example then mean 8 examples','dropout':'disabled','gradient_scale':1024,'optimizer_updates':0,
 'scope':'small held-out diagnostic; local gradients and finite updates, not causal attribution to all methods','code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('Gradient audit complete.',flush=True)
