import argparse
import io
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import Qwen2VLForConditionalGeneration
from common import ROOT, write_json
from cv_data import CACHE, MODEL, processor
from cv_runtime import install_cached_vision, install_sapt, batch_tensors, answer_loss, fused_embeddings, generate
from cv_train import train_task
from recovery import method_bytes
from run_olora import create_method

p=argparse.ArgumentParser();p.add_argument('--method',required=True);args=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(42)
data=torch.load(CACHE/'data.pt',weights_only=True)
config=json.loads((ROOT/f'exp/configs/{args.method}_qwen2vl.json').read_text())
proc=processor();tok=proc.tokenizer
model=Qwen2VLForConditionalGeneration.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float16,device_map={'':0},attn_implementation='sdpa')
install_cached_vision(model)
method=create_method(model,config);install_sapt(method)
# Exact equivalence to stock multimodal forward, including multimodal position IDs.
row=data['medical_vqa_rad']['train'][0]
batch=batch_tensors([row],tok,model.device,True)
model.eval()
with torch.no_grad():
 pos,_=model.get_rope_index(batch['input_ids'],batch['image_grid_thw'],attention_mask=batch['attention_mask'])
 ref=model(**batch,position_ids=pos,use_cache=False).loss
 optimized=answer_loss(model,batch)
 assert torch.allclose(ref,optimized,atol=2e-3,rtol=1e-4),(ref,optimized)
# Last-stage footprint, true vision and longest language context in a training update.
for _ in range(8):method.begin_task()
longest=max(data[config['tasks'][0]]['train'],key=lambda r:len(r['input_ids']))
selected=[longest,row]
dest=ROOT/'exp/CV_result/validation'/args.method/'two_updates'
testconfig={**config,'batch_size':1,'target_tokens':16}
resources=train_task(method,tok,{'train':selected,'dev':[row]},testconfig,dest,'preflight',False)
assert method.task_count==9
assert any(torch.count_nonzero(layer.adapters[-1]['up'].weight).item() for layer in method.layers.values())
model.eval()
tokens,texts=generate(model,method,tok,[longest,row,row,longest],testconfig)
assert len(texts)==4
state=torch.load(dest/'recovery.pt',map_location='cpu',weights_only=False)
assert state['update']==2 and state['offset']==0 and state['epoch']==1
assert state['optimizer']['state'] and state['scaler']
if args.method == 'sapt_lora':
 from cv_reflection import finish_task
 reflection=finish_task(method,tok,data['medical_vqa_rad']['train'][:2],dest/'reflection_test',
     {**testconfig,'sapt_pseudo_samples':2})
 assert len(method.memory[-1]['pools'])==2
 finish_task(method,tok,data['medical_vqa_rad']['train'][:2],dest/'reflection_test',
     {**testconfig,'sapt_pseudo_samples':2})
 method.begin_task()
 probe=batch_tensors([row],tok,model.device,True)
 method.prepare_batch([row],tok,probe,True)
 kl=method.penalty(0,0)
 assert torch.isfinite(kl)
 kl.backward()
 write_json(dest/'reflection_passed.json',{'image_conditioned_reflection':True,'next_stage_kl':kl.item(),'resources':reflection})
write_json(dest/'passed.json',{'method':args.method,'stock_loss':ref.item(),'optimized_loss':optimized.item(),
 'nine_stage_train':resources,'mixed_batch4_generation':texts,'peak_bytes':torch.cuda.max_memory_allocated(),
 'checks':['true image MRoPE loss equivalence','finite AMP gradients','nine-adapter capacity','mixed left-padding generation','optimizer/scheduler/scaler/RNG checkpoint']})
print('PREFLIGHT PASSED',args.method,flush=True)
