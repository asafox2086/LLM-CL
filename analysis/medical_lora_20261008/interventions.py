"""Read-only gate override and CE-vs-regularizer gradient diagnostics."""
import csv,hashlib,io,json,sys
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from cv_data import MODEL,processor
from cv_runtime import answer_loss,batch_tensors,install_cached_vision,install_sapt
from run_olora import create_method
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run'])
DATA=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(4)
model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
install_cached_vision(model);tok=processor().tokenizer
config=json.loads((RUN/'sapt_lora/config.json').read_text());method=create_method(model,config);install_sapt(method)
routing_ce=[];selection={}
for stage in [6,7,9]:
 state=torch.load(RUN/'sapt_lora/checkpoints'/f'stage_{stage:02d}'/'completed.pt',map_location='cpu',weights_only=False);method.load(io.BytesIO(state['method']));model.eval()
 for task in ['medical_mts_dialog_note','medical_iu_xray_impression','medical_vqa_rad']:
  j=config['tasks'].index(task)
  if j>=stage:continue
  rows=sorted(DATA[task]['test'],key=lambda r:hashlib.sha256(('medical-gate42\n'+r['instance_id']).encode()).hexdigest())[:16]
  selection[task]=[r['instance_id'] for r in rows]
  for mode in ['learned_routing','oracle_current_task_adapter']:
   values=[]
   for row in rows:
    batch=batch_tensors([row],tok,model.device,True);method.prepare_batch([row],tok,batch,False)
    if mode=='oracle_current_task_adapter':
     for layer in method.layers.values():
      weights=torch.zeros((1,stage),device=model.device);weights[:,j]=1;layer.attention=weights
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):values.append(float(answer_loss(model,batch)))
   routing_ce.append({'stage':stage,'task':task,'mode':mode,'n':16,'teacher_forced_CE':sum(values)/16})
with (OUT/'routing_override_CE.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(routing_ce[0]),lineterminator='\n');w.writeheader();w.writerows(routing_ce)
# Remove SAPT wrappers before attaching O-LoRA; base model weights stay unchanged.
for name,layer in method.layers.items():
 parent,child=name.rsplit('.',1);setattr(model.get_submodule(parent),child,layer.base)
delattr(model,'sapt_router');del method
config=json.loads((RUN/'olora/config.json').read_text());method=create_method(model,config)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False
results=[]
for stage in [6,7,9]:
 state=torch.load(RUN/'olora/checkpoints'/f'stage_{stage:02d}'/'completed.pt',map_location='cpu',weights_only=False);method.load(io.BytesIO(state['method']))
 model.train()
 for m in model.modules():
  if isinstance(m,torch.nn.Dropout):m.eval()
 params=method.parameters();task=config['tasks'][stage-1]
 rows=sorted(DATA[task]['test'],key=lambda r:hashlib.sha256(('medical-penalty42\n'+r['instance_id']).encode()).hexdigest())[:8]
 model.zero_grad(set_to_none=True);losses=[]
 for row in rows:
  batch=batch_tensors([row],tok,model.device,True)
  with torch.autocast('cuda',dtype=torch.float16):loss=answer_loss(model,batch)
  losses.append(float(loss.detach()));(loss*128/8).backward()
 gce=torch.cat([(p.grad.detach().cpu().float()/128 if p.grad is not None else torch.zeros_like(p.detach().cpu()).float()).flatten() for p in params])
 model.zero_grad(set_to_none=True);reg=method.penalty(config['orthogonal_weight'],config['l2_weight']);reg.backward()
 greg=torch.cat([(p.grad.detach().cpu().float() if p.grad is not None else torch.zeros_like(p.detach().cpu()).float()).flatten() for p in params]);assert torch.isfinite(gce).all() and torch.isfinite(greg).all()
 results.append({'stage':stage,'task':task,'CE':sum(losses)/8,'weighted_penalty':float(reg.detach()),'CE_grad_norm':float(gce.norm()),
 'penalty_grad_norm':float(greg.norm()),'grad_norm_ratio':float(greg.norm()/gce.norm()),'gradient_cosine':float(gce@greg)/float(gce.norm()*greg.norm())})
with (OUT/'regularization_gradients.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
(OUT/'intervention_protocol.json').write_text(json.dumps({'routing_ids':selection,'routing_override':'oracle task ID one-hot; checkpoint parameters fixed; inference CE only, not retraining or deployable routing',
 'regularization':'O-LoRA last adapter trainable, historical adapters frozen, dropout off, weighted orthogonal penalty vs mean8 test CE gradients',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'optimizer_updates':0},indent=2)+'\n')
print('Gate and regularization interventions complete.',flush=True)
