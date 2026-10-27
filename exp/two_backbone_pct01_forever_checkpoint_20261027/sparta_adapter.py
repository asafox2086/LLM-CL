"""SpaRTA author dual-rank DATA kernels and progressive context router."""
import ast,copy,hashlib,json,logging,__future__,math
from pathlib import Path
from types import SimpleNamespace
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
def author_components():
 folder=ROOT/"code/SpaRTA";known=json.loads((folder/"files_sha256.json").read_text());ns={"torch":torch,"nn":nn,"copy":copy,"logger":logging.getLogger(__name__)};hashes={}
 for rel in ["src/llamafactory/model/inject_data.py","patch/modeling_qwen2.py"]:
  p=folder/"source"/rel;sha=next(x["sha256"] for x in known if x["path"]==rel);assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;hashes[rel]=sha;tree=ast.parse(p.read_text())
  if rel.endswith("inject_data.py"):defs=[x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="DATAInjectedLinear"]
  else:
   defs=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="tensor_prompt"]
   cls=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=="Qwen2Model")
   defs += [x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name in ["gram_schmidt","get_scale","ortho_penalty"]]
  exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(p),"exec",flags=__future__.annotations.compiler_flag),ns)
 return ns,json.loads((folder/"source_manifest.json").read_text())["revision"],hashes
class Router(nn.Module):
 def __init__(self,width,kernels,task_count):
  super().__init__();self.k=kernels;self.task_id=0;self.config=SimpleNamespace(adaprompt=8,n_tasks=task_count,ortho_mu=10,project=0,nomlp=False)
  for suffix in ["","2"]:
   for letter,dim in [("k",width),("a",width),("w",width)]:setattr(self,f"data_{letter}{suffix}_0",kernels["tensor_prompt"](8*task_count,dim))
 def gram_schmidt(self,v):return self.k["gram_schmidt"](self,v)
 def ortho_penalty(self,v):return self.k["ortho_penalty"](self,v)
 def allocate(self,stage):
  self.task_id=max(stage-1,0)
  for suffix in ["","2"]:
   for letter in ["k","a","w"]:
    name=f"data_{letter}{suffix}_0";setattr(self,name,self.gram_schmidt(getattr(self,name)))
 def forward(self,input):
  a,loss1=self.k["get_scale"](self,input.float(),"scale1",0);b,loss2=self.k["get_scale"](self,input.float(),"scale2",0)
  penalty=0 if loss1 is None else loss1+loss2
  return a.sigmoid(),b.sigmoid(),penalty
class SpaRTAMethod:
 def __init__(self,model,name,instructions,embedding=None):
  self.model=model;self.name=name;self.embedding=None;self.learning_rate=1e-4;self.stage=0;self.global_step=0;self.penalties=[];self.track=False;self.offload_activations=False;self.extra_loss_per_micro=True
  self.kernel,self.original_revision,self.source_hashes=author_components()
  for p in model.parameters():p.requires_grad=False
  self.layers=[];self.router_groups=[];width=model.config.hidden_size
  self.routers=nn.ModuleList([Router(width,self.kernel,len(instructions)).to(next(model.parameters()).device) for _ in range(math.ceil(len(model.model.layers)/4))]);model.sparta_routers=self.routers
  for i,decoder in enumerate(model.model.layers):
   group=i//4
   for block in [decoder.self_attn,decoder.mlp]:
    for child,base in list(block.named_children()):
     if not isinstance(base,nn.Linear):continue
     adapter=self.kernel["DATAInjectedLinear"](base.in_features,base.out_features,base.bias is not None,r=2,r2=8).to(device=base.weight.device,dtype=torch.float32);adapter.linear_data.weight=base.weight
     if base.bias is not None:adapter.linear_data.bias=base.bias
     setattr(block,child,adapter);self.layers.append(adapter);self.router_groups.append(group)
   if i%4==0:
    decoder.register_forward_pre_hook(self.make_hook(group))
  self.initial={name:param.detach().cpu().clone() for name,param in model.named_parameters() if param.requires_grad}
 def make_hook(self,group):
  def hook(module,args):
   low,high,penalty=self.routers[group](args[0])
   for layer,g in zip(self.layers,self.router_groups):
    if g==group:layer.scale1=low;layer.scale2=high
   if self.track and self.model.training and torch.is_grad_enabled():self.penalties.append(penalty)
  return hook
 def recording(self,enabled):self.track=enabled
 def set_instruction(self,task):pass
 def begin_stage(self,stage):
  self.stage=stage
  for router in self.routers:router.allocate(stage)
  self.penalties=[]
 def prepare_update(self,processor):self.penalties=[]
 def extra_loss(self):
  loss=sum(self.penalties);self.penalties=[];return loss
 def after_update(self,step,updates):
  self.global_step+=1
  if self.global_step%200==0:
   with torch.no_grad():
    for name,p in self.model.named_parameters():
     if p.requires_grad and name in self.initial and name.endswith((".weight",".bias")):
      mask=torch.rand_like(p)<.00001;p.copy_(torch.where(mask,self.initial[name].to(p.device),p))
 def method_state(self,stage):return {"method":self.name,"stage":stage,"upstream_revision":self.original_revision,"low_rank":2,"high_rank":8,"progressive_router":True,"components_per_task":8,"router_group_layers":4,"ortho_mu":10,"stochastic_restore_every_updates":200,"restore_probability":.00001,"note":"Author stochastic restore fires every 200 optimizer updates across stages; actual count is logged"}
