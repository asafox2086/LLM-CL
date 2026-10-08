"""Interrupted optimizer-step recovery must exactly match an uninterrupted run."""
import io
import json
import random
import sys
import tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from transformers import Qwen2VLConfig,Qwen2VLForConditionalGeneration
from common import ROOT,write_json
from cv_data import processor
from cv_runtime import install_sapt
import cv_train
from run_olora import create_method
from recovery import method_bytes

torch.set_num_threads(2)
tok=processor().tokenizer
out=ROOT/'exp/CV_result/validation/recovery';out.mkdir(parents=True,exist_ok=True)
rows=[{'task_id':'tiny','instance_id':str(i),'prompt_ids':[10,11,12],
 'input_ids':[10,11,12,20+i,21,22],'labels':[-100,-100,-100,20+i,21,22]} for i in range(3)]
def same(a,b):
 if isinstance(a,torch.Tensor):assert torch.equal(a,b)
 elif isinstance(a,dict):
  assert a.keys()==b.keys()
  for k in a:same(a[k],b[k])
 elif isinstance(a,(tuple,list)):
  assert len(a)==len(b)
  for x,y in zip(a,b):same(x,y)
 else:assert a==b,(a,b)

for name in ['seq_lora','migu_lora','olora','sapt_lora']:
 cfg=json.loads((ROOT/f'exp/configs/{name}_qwen2vl.json').read_text())
 cfg.update(batch_size=1,checkpoint_selection='last_epoch',precision='fp32_cpu')
 def make():
  random.seed(9);np.random.seed(9);torch.manual_seed(9)
  mc=Qwen2VLConfig(vocab_size=100,hidden_size=96,intermediate_size=128,num_hidden_layers=1,
    num_attention_heads=4,num_key_value_heads=2,image_token_id=90,video_token_id=91,vision_start_token_id=92,rope_scaling={'type':'mrope','mrope_section':[4,4,4]},
    vision_config={'depth':1,'embed_dim':32,'hidden_size':96,'num_heads':4,'mlp_ratio':2})
  model=Qwen2VLForConditionalGeneration(mc)
  method=create_method(model,cfg);install_sapt(method)
  for _ in range(2):method.begin_task()
  if name=='sapt_lora':method.memory=[{'task_id':'old','pools':torch.randn(4,96),'attention':[.4,.6]}]
  return method
 with tempfile.TemporaryDirectory(dir=out) as temp:
  root=Path(temp)
  full=make();cv_train.train_task(full,tok,{'train':rows},cfg,root/'full','tiny')
  state_full=torch.load(io.BytesIO(method_bytes(full)),weights_only=True)
  del full
  interrupted=make();original=cv_train.save_training
  def stop(*args,**kwargs):
   original(*args,**kwargs)
   if kwargs['update']==2:raise InterruptedError('simulated termination after committed update')
  cv_train.save_training=stop
  try:cv_train.train_task(interrupted,tok,{'train':rows},cfg,root/'resume','tiny')
  except InterruptedError:pass
  else:raise AssertionError('interruption missing')
  finally:cv_train.save_training=original
  del interrupted
  resumed=make();cv_train.train_task(resumed,tok,{'train':rows},cfg,root/'resume','tiny',resume=True)
  same(state_full,torch.load(io.BytesIO(method_bytes(resumed)),weights_only=True))
  a=torch.load(root/'full/recovery.pt',weights_only=False);b=torch.load(root/'resume/recovery.pt',weights_only=False)
  for key in ['optimizer','scheduler','scaler','update','epoch','offset']:same(a[key],b[key])
  write_json(out/(name+'.json'),{'passed':True,'interrupted_after_update':2,'total_updates':3,'exact_adapter_router_optimizer_scheduler_match':True})
  print('EXACT RECOVERY PASSED',name,flush=True)
