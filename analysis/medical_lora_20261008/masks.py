"""Measure what the existing MIGU masks remove from held-out answer gradients."""
import csv,hashlib,io,json,sys
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'exp'))
from cv_data import MODEL,processor
from cv_runtime import install_cached_vision,answer_loss,batch_tensors
from run_olora import create_method
RUN=Path(json.loads((ROOT/'summary_cv/latest.json').read_text())['run']);config=json.loads((RUN/'migu_lora/config.json').read_text())
data=torch.load(ROOT/'exp/CV_result/shared/data.pt',map_location='cpu',weights_only=False)
torch.set_num_threads(4)
model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
install_cached_vision(model);method=create_method(model,config);tok=processor().tokenizer
state=torch.load(RUN/'migu_lora/checkpoints/stage_05/completed.pt',map_location='cpu',weights_only=False);method.load(io.BytesIO(state['method']))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads();model.config.use_cache=False;model.train()
for module in model.modules():
 if isinstance(module,torch.nn.Dropout):module.eval()
results=[];ids={}
for task in ['task002_quoref_answer_generation','medical_mts_dialog_note','medical_iu_xray_impression','medical_vqa_rad']:
 rows=sorted(data[task]['test'],key=lambda r:hashlib.sha256(('medical-mask42\n'+r['instance_id']).encode()).hexdigest())[:8]
 ids[task]=[r['instance_id'] for r in rows];method.before_update(0);model.zero_grad(set_to_none=True)
 for row in rows:
  batch=batch_tensors([row],tok,model.device,True);method.prepare_batch([row],tok,batch,True)
  with torch.autocast('cuda',dtype=torch.float16):loss=answer_loss(model,batch)
  method.after_forward();(loss*128/8).backward()
 before=sum(float((p.grad.detach().float()/128).square().sum()) for p in method.parameters())
 layerrows=[]
 for name,layer in method.layers.items():
  for key,linear in layer.adapters[0].items():
   mask=layer.magnitudes[key]>=torch.quantile(layer.magnitudes[key],method.threshold)
   layerrows.append((key,float(mask.float().mean())))
 method.before_step()
 after=sum(float((p.grad.detach().float()/128).square().sum()) for p in method.parameters())
 results.append({'stage':5,'task':task,'n':8,'gradient_energy_retained':after/before,
  'down_rows_retained':sum(v for k,v in layerrows if k=='down')/len([1 for k,v in layerrows if k=='down']),
  'up_rows_retained':sum(v for k,v in layerrows if k=='up')/len([1 for k,v in layerrows if k=='up'])})
with (OUT/'migu_gradient_masks.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(results[0]),lineterminator='\n');w.writeheader();w.writerows(results)
(OUT/'mask_protocol.json').write_text(json.dumps({'stage':5,'method':'MIGU-LoRA','ids':ids,'mask_ratio':.7,'dropout':'off','optimizer_updates':0,
 'scope':'8 held-out questions per task, real implementation magnitudes over prompt+answer tokens; descriptive removed gradient energy, not a medical-entity importance measure',
 'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
print('MIGU gradient masks measured.',flush=True)
