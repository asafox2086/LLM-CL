"""Native output reuse versus FFN-local PAS. No work is executed on import."""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[3]
PORT=ROOT/'exp/two_backbone_native_cl_20261017_v2'
REPORT=ROOT/'summary/final_weight_reduction_20261023/native_response_reuse_pilot_v1'
WORK=Path(__file__).parent/'private'
BASE=ROOT/'model/SmolVLM-500M-Instruct_a7da5b9'
MODEL_ID='HuggingFaceTB/SmolVLM-500M-Instruct'
REVISION='a7da5b986cb59b408707209984f360a5f4ad7e47'
CASES=('E','Q','R')
BUDGET={'E':21600,'Q':7200,'R':7200}
TRAINABLE={'Q':819200,'R':79462400,'E':87326720}
PREFIX='model.text_model.layers.'
EXTERNAL_ANCHOR=43.79198740853657

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_suffix(path.suffix+'.tmp')
 with tmp.open('w',encoding='utf-8') as f:
  json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(path)
 if os.name!='nt':
  fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
  try:os.fsync(fd)
  finally:os.close(fd)

def study():
 p=ROOT/'exp/osft_increment_screen_20261019/v1/study.py'
 s=importlib.util.spec_from_file_location('native_reuse_frozen_study',p)
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def digest(tensors):
 sys.path.insert(0,str(PORT))
 from training_validity import parameter_digest
 return parameter_digest(list(tensors))

def layout(model):
 """Coefficients are not optimizer coordinates; deduplicate exact shared views."""
 weights={};buffers={};seen=set();storage_intervals={}
 for kind,items,out in [('weight',model.named_parameters(),weights),('buffer',model.named_buffers(),buffers)]:
  for n,p in items:
   key=(str(p.device),p.untyped_storage().data_ptr(),p.storage_offset(),tuple(p.shape),tuple(p.stride()),str(p.dtype))
   if key in seen:continue
   seen.add(key)
   # Reject non-contiguous views rather than silently miscount internal overlap.
   assert p.is_contiguous(),n
   storage=(str(p.device),p.untyped_storage().data_ptr());start=p.storage_offset()*p.element_size();end=start+p.numel()*p.element_size()
   prior=storage_intervals.setdefault(storage,[]);overlap=0
   for a,b in prior:overlap+=max(0,min(end,b)-max(start,a))
   unique=end-start-overlap;assert unique>=0,n
   intervals=sorted(prior+[(start,end)]);merged=[]
   for a,b in intervals:
    if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(b,merged[-1][1]))
    else:merged.append((a,b))
   storage_intervals[storage]=merged
   assert unique%p.element_size()==0,n
   out[n]=dict(shape=list(p.shape),numel=p.numel(),bytes=p.numel()*p.element_size(),
    storage_unique_bytes=unique,storage_unique_elements=unique//p.element_size(),dtype=str(p.dtype),requires_grad=p.requires_grad)
 return dict(weights=weights,buffers=buffers,weight_coefficients=sum(x['storage_unique_elements'] for x in weights.values()),
  weight_bytes=sum(x['storage_unique_bytes'] for x in weights.values()),buffer_bytes=sum(x['storage_unique_bytes'] for x in buffers.values()),
  trainable_coordinates=sum(p.numel() for p in model.parameters() if p.requires_grad))

def names(model,targets):
 import torch
 return [(n,p) for n,p in model.named_modules() if n.startswith(PREFIX)
  and n.rsplit('.',1)[-1] in targets and isinstance(p,torch.nn.Linear)]

def put(model,name,module):
 parent,child=name.rsplit('.',1);setattr(model.get_submodule(parent),child,module)

def seeded(base,name,callback):
 import torch
 devices=[base.weight.device.index] if base.weight.is_cuda else []
 with torch.random.fork_rng(devices=devices):
  torch.manual_seed(int(hashlib.sha256(('42:native-reuse:'+name).encode()).hexdigest()[:8],16))
  return callback()

def install_experts(model):
 import torch
 sys.path.insert(0,str(PORT))
 from pass_adapter import components
 source,revision=components();native=source['CoINMOELoraLinear']
 class Hosted(native):
  def forward(self,x,**kwargs):
   if self.training and torch.is_grad_enabled():
    with torch.autograd.graph.save_on_cpu(pin_memory=True),torch.autocast(x.device.type,enabled=False):
     return super().forward(x,**kwargs)
   with torch.autocast(x.device.type,enabled=False):return super().forward(x,**kwargs)
 layers=[]
 for name,base in names(model,{'gate_proj','up_proj','down_proj'}):
  def make():
   m=Hosted('default',base.in_features,base.out_features,r=128,lora_alpha=256,lora_dropout=.05,
    expert_num=2,gate_rank=16,gate_tau=1.5,gate_router='a',gate_tie='A_T')
   m.weight,m.bias=base.weight,base.bias
   return m.to(base.weight.device)
  m=seeded(base,name,make);put(model,name,m);layers.append(m)
 return layers,source,revision

def make_method(model,case,state_root):
 import math
 import torch
 from types import SimpleNamespace
 sys.path.insert(0,str(PORT))
 from methods import SeqLinear
 assert case in CASES
 for p in model.parameters():p.requires_grad=False
 common=[]
 for name,base in names(model,{'q_proj','v_proj'}):
  m=seeded(base,name,lambda:SeqLinear(base,rank=8,alpha=32));put(model,name,m);common.append(name)
 expert_layers=[];learner=None;revision=None
 if case=='R':
  for _,base in names(model,{'down_proj'}):
   base.weight=torch.nn.Parameter(base.weight.detach().float().clone())
 if case=='E':
  assert not Path(state_root).exists(),'Fresh attempt cannot load pre-existing author CL state'
  expert_layers,source,revision=install_experts(model)
  args=SimpleNamespace(moelora_orth_lambda=1e-4,moelora_B_reg_scale=.1,
   moelora_importance_top_p=.4,output_dir=str(state_root))
  learner=source['moeloraLearner'](args,0,model)
  assert not learner._I_agg and not learner._A_ref and not learner._B_ref
 class Method:
  name='NativeResponseReuse'
  learning_rate=1e-4
  host_optimizer=True
  offload_activations=True
  def __init__(self):
   self.model=model;self.case=case;self.learner=learner;self.expert_layers=expert_layers
   self.common_names=common;self.original_revision=revision;self.records=[];self.live=None
  def set_instruction(self,stage):pass
  def begin_stage(self,stage):
   assert self.live is None
   self.live=dict(stage=stage,recording_true=0,recording_false=0,post_backward=0,after_update=0)
   if learner is not None:
    learner.before_train(stage,model)
    for layer in expert_layers:layer.gate_tau=1.5
  def recording(self,enabled):
   if learner is not None:learner.enable_record(enabled)
   if self.live is not None:self.live['recording_true' if enabled else 'recording_false']+=1
  def post_backward(self,scale,progress):
   if learner is not None:
    ps=[p for p in model.parameters() if p.grad is not None]
    for p in ps:p.grad.div_(scale)
    learner.post_backward(model,cl_task_progress=progress)
    for p in ps:p.grad.mul_(scale)
   self.live['post_backward']+=1
  def after_update(self,step,total):
   self.live['after_update']+=1
   for layer in expert_layers:layer.gate_tau=.7+.4*(1+math.cos(math.pi*min(1,step/total)))
  def end_stage(self,stage):
   assert self.live['stage']==stage
   if learner is not None:
    self.live.update(usage_callbacks_before_end=learner._cur_I_steps,
     regularization_calls_before_end=learner._loss_calls_in_task,
     current_importance_modules_before_end=len(learner._cur_I_ema))
    Path(state_root).mkdir(parents=True,exist_ok=True)
    save=learner._save_cl_state;learner._save_cl_state=lambda task:None
    learner._cl_state_root=str(state_root)
    try:learner.after_train(stage,model)
    finally:learner._save_cl_state=save
    self.live.update(importance_modules=len(learner._I_agg),usage_callbacks_after_end=learner._cur_I_steps,
     reference_bytes=sum(v.numel()*v.element_size() for field in ('_I_agg','_A_ref','_B_ref') for v in getattr(learner,field).values()))
   self.records.append(self.live);self.live=None
  def method_state(self):
   return dict(case=case,records=self.records,learner=None if learner is None else
    {k:getattr(learner,k) for k in ['_I_agg','_A_ref','_B_ref','_cur_I_ema','_cur_I_steps','_loss_calls_in_task']},
    gate_tau=[x.gate_tau for x in expert_layers])
 return Method()

def compile_final(method,dtype=None):
 """Destructive final compilation; never assumes floating-point fold is lossless."""
 import torch
 from torch import nn
 from methods import SeqLinear
 dtype=torch.bfloat16 if dtype is None else dtype
 model=method.model;model.eval();method.recording(False)
 with torch.no_grad():
  for name,m in list(model.named_modules()):
   if isinstance(m,SeqLinear):
    base=nn.Linear(m.weight.shape[1],m.weight.shape[0],bias=m.bias is not None,
     device=m.weight.device,dtype=dtype)
    base.weight.copy_((m.weight.float()+m.scale*(m.B.weight.float()@m.A.weight.float())).to(dtype))
    if m.bias is not None:base.bias.copy_(m.bias.to(dtype))
    put(model,name,base)
  if method.case=='R':
   for _,m in names(model,{'down_proj'}):m.to(dtype=dtype)
  for p in model.parameters():p.requires_grad=False
 return model

def restore_compiled(path,device='cpu',dtype=None):
 import torch
 from transformers import Idefics3Config,Idefics3ForConditionalGeneration
 payload=torch.load(path,map_location='cpu',weights_only=False)
 assert payload['kind']=='native_reuse_compiled_v1'
 dtype=torch.bfloat16 if dtype is None else dtype
 cfg=Idefics3Config.from_dict(payload['config'])
 model=Idefics3ForConditionalGeneration._from_config(cfg,torch_dtype=dtype,attn_implementation='sdpa')
 if payload['case']=='E':
  layers,_,_=install_experts(model)
  assert len(layers)==len(payload['gate_tau'])
  for m,tau in zip(layers,payload['gate_tau']):m.gate_tau=tau
 model.load_state_dict(payload['weights'],strict=True)
 assert digest(model.state_dict().values())==payload['state_digest']
 for p in model.parameters():p.requires_grad=False
 return model.to(device).eval()
