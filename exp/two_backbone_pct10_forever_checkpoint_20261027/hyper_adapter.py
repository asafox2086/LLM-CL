"""Hyper-LLaVA author top-1 experts, Poincare geometry and BC modality weights."""
import ast,hashlib,json,__future__,math
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
def author_components():
 folder=ROOT/"code/Hyper_LLaVA";known=json.loads((folder/"files_sha256.json").read_text());env={"torch":torch,"nn":nn,"F":F,"np":np,"transpose":lambda w,fan:w.T if fan else w};hashes={}
 for rel in ["Hyper/peft/tuners/clitmoelora.py","llava/model/llava_arch.py","compute_routing_weights.py"]:
  p=folder/"source"/rel;sha=next(x["sha256"] for x in known if x["path"]==rel);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;hashes[rel]=sha;tree=ast.parse(p.read_text())
  if rel.endswith("clitmoelora.py"):
   defs=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name in ["HyperMOELinearA","HyperMOELinearB","HyperMOEExpert"]]
   cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="HyperMOELoraLinear");defs +=[x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=="forward"]
  else:defs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in ["map_to_poincare","poincare_dist","calculate_log_bc_diagonal","compute_current_stage_weights"]]
  exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(p),"exec",flags=__future__.annotations.compiler_flag),env)
 return env,json.loads((folder/"source_manifest.json").read_text())["revision"],hashes
class HyperLinear(nn.Module):
 def __init__(self,base,kernel,expert_count):
  super().__init__();self.weight,self.bias=base.weight,base.bias;self.in_features,self.out_features=base.in_features,base.out_features;self.active_adapter="default";self.expert_num=expert_count;self.cur_task=0;self.layer=0;self.expert_weight=[1.]+[0.]*(expert_count-1);self.fan_in_fan_out=False;self.disable_adapters=False;self.merged=False;self.r={"default":8*expert_count};self.scaling={"default":2.};self.lora_A=nn.ModuleDict({"default":kernel["HyperMOELinearA"](base.in_features,8*expert_count,expert_count,0,True,0,self.expert_weight)});self.lora_B=nn.ModuleDict({"default":kernel["HyperMOELinearB"](8*expert_count,base.out_features,expert_count,0,True,0,self.expert_weight)});self.lora_dropout=nn.ModuleDict({"default":nn.Dropout(.05)});self.kernel=kernel
  for a,b in zip(self.lora_A["default"].loraA,self.lora_B["default"].loraB):nn.init.normal_(a.mlp.weight,mean=0.,std=.01);nn.init.zeros_(b.mlp.weight)
  self.to(base.weight.device)
 def forward(self,x,**kwargs):return self.kernel["forward"](self,x,**kwargs)
class HyperMethod:
 def __init__(self,model,name,instructions,embedding=None):
  self.expert_count=len(instructions);self.model=model;self.name=name;self.embedding=None;self.stage=0;self.learning_rate=1e-4;self.layers=[];self.statistics=[];self.features={};self.stage_features={};self.processor=None;self.offload_activations=False
  self.kernel,self.original_revision,self.source_hashes=author_components()
  for p in model.parameters():p.requires_grad=False
  targets={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
  for key,base in list(model.named_modules()):
   if key.startswith("model.layers.") and key.rsplit(".",1)[-1] in targets and isinstance(base,nn.Linear):
    parent,child=key.rsplit(".",1);layer=HyperLinear(base,self.kernel,self.expert_count);setattr(model.get_submodule(parent),child,layer);self.layers.append(layer)
  from transformers import AutoTokenizer,AutoModel
  path=ROOT/"model/all-MiniLM-L6-v2";self.text_tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True);self.text_encoder=AutoModel.from_pretrained(path,local_files_only=True).eval()
  for p in self.text_encoder.parameters():p.requires_grad=False
 def bind_context(self,processor,manifest,data,run,preflight=False):
  self.train_ids={t['task']:{x['id'] for x in data[(t['task'],'train')]} for t in manifest['tasks']}
 def set_instruction(self,task):pass
 def begin_stage(self,stage):
  self.stage=stage;self.stage_features={}
  for layer in self.layers:
   layer.cur_task=stage-1
   for i,(a,b) in enumerate(zip(layer.lora_A["default"].loraA,layer.lora_B["default"].loraB)):
    for p in list(a.parameters())+list(b.parameters()):p.requires_grad=i==stage-1
 def prepare_row(self,row,batch):
  if row["id"] not in self.features:
   with torch.no_grad():
    data=self.text_tokenizer(row["question"],return_tensors="pt",truncation=True,max_length=512);z=self.text_encoder(**data).last_hidden_state;mask=data["attention_mask"].unsqueeze(-1).float();txt=((z*mask).sum(1)/mask.sum(1)).float().cpu();img=None
    if "pixel_values" in batch:
     values=batch["pixel_values"]
     if values.shape[-1]==self.model.config.hidden_size:img=values.detach().float().mean(0,keepdim=True).cpu()
     else:
      with torch.autocast("cuda",dtype=torch.float16):img=self.model.visual(values.to(self.model.visual.get_dtype()),batch["image_grid_thw"]).detach().float().mean(0,keepdim=True).cpu()
   self.features[row["id"]]=(img,txt)
  img,txt=self.features[row["id"]]
  if self.model.training:self.stage_features[row["id"]]=(img,txt);return
  if not self.statistics:
   weights=[1.]+[0.]*(self.expert_count-1)
  else:
   txt_vars=[s["txt_var"].sum().item() for s in self.statistics];txt_scale=1./(np.mean(txt_vars)+1e-6);zt=self.kernel["map_to_poincare"](txt);ts=[];im_scores=[]
   iv=[s["img_var"].sum().item() for s in self.statistics if s["img_mean"] is not None];img_scale=1./(np.mean(iv)+1e-6) if iv else 1.
   for s in self.statistics:
    ts.append(-self.kernel["poincare_dist"](zt,self.kernel["map_to_poincare"](s["txt_mean"],s["txt_var"].sum().item(),txt_scale)).item())
    im_scores.append(-self.kernel["poincare_dist"](self.kernel["map_to_poincare"](img),self.kernel["map_to_poincare"](s["img_mean"],s["img_var"].sum().item(),img_scale)).item() if img is not None and s["img_mean"] is not None else -float("inf"))
   t=torch.tensor(ts);scale=t.std() if len(t)>1 else torch.tensor(1.);scale=scale if torch.isfinite(scale) and scale>=1e-6 else torch.tensor(1.);pt=F.softmax(t/scale,dim=0)
   valid=[i for i,s in enumerate(self.statistics) if s["img_mean"] is not None];wi=[0.]*len(self.statistics);pi=torch.zeros_like(pt)
   if img is not None and valid:
    stats=[{k:(v.numpy() if isinstance(v,torch.Tensor) else v) for k,v in self.statistics[i].items()} for i in valid];w=self.kernel["compute_current_stage_weights"](stats)
    scores=torch.tensor([im_scores[i] for i in valid]);scale=scores.std() if len(scores)>1 else torch.tensor(1.);scale=scale if torch.isfinite(scale) and scale>=1e-6 else torch.tensor(1.);prob=F.softmax(scores/scale,dim=0)
    for i,value,weight in zip(valid,prob,w):pi[i]=value;wi[i]=weight
   wi=torch.tensor(wi);scores=pi*wi+pt*(1-wi);chosen=int(scores.argmax());weights=[0.]*self.expert_count;weights[chosen]=1.
  for layer in self.layers:layer.expert_weight=weights
 def end_stage(self,stage):
  assert set(self.stage_features)==self.train_ids[stage],"Hyper routing statistics must use exactly the current training split"
  pairs=list(self.stage_features.values());txt=torch.cat([v[1] for v in pairs]);imgs=[v[0] for v in pairs if v[0] is not None];im=torch.cat(imgs) if imgs else None
  self.statistics.append({"txt_mean":txt.mean(0,keepdim=True),"txt_var":txt.var(0,unbiased=False),"img_mean":im.mean(0,keepdim=True) if im is not None else None,"img_var":im.var(0,unbiased=False) if im is not None else None})
 def method_state(self,stage):return {"method":self.name,"stage":stage,"upstream_revision":self.original_revision,"rank_per_expert":8,"experts":self.expert_count,"author_top1_forward":True,"poincare_variance_radius":True,"bc_adaptive_modality_fusion":True,"feature_encoder_adaptation":"frozen Qwen2-VL visual merger features and frozen MiniLM text; text-only samples use text route without fabricated images","statistics_current_train_only":True}
