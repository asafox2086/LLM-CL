"""OSFT: original SVD/reconstruction/projection kernels on Qwen2-VL decoder."""
import ast,hashlib,json,math,typing as t,__future__
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ROOT=Path(__file__).resolve().parents[2]
def author_functions():
 rel="src/mini_trainer/osft_utils.py";folder=ROOT/"code/OSFT";p=folder/"source"/rel
 expected=next(x["sha256"] for x in json.loads((folder/"files_sha256.json").read_text()) if x["path"]==rel)
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 names={"create_svd_dict","reconstruct_weight_matrix","project_parameter_to_orthogonal_space","project_gradient_to_orthogonal_space"}
 defs=[n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in names];assert len(defs)==4
 env={"torch":torch,"nn":nn,"dist":dist,"math":math,"t":t,"OSFT_CACHE_V":False,"SVDDecompositionDict":dict}
 exec(compile(ast.fix_missing_locations(ast.Module(body=defs,type_ignores=[])),str(p),"exec",flags=__future__.annotations.compiler_flag),env)
 return env,json.loads((folder/"source_manifest.json").read_text())["revision"],expected
class OSFTLinear(nn.Module):
 def __init__(self,base,kernels):
  super().__init__();self.in_features=base.in_features;self.out_features=base.out_features;self.kernel=kernels;self.bias=base.bias
  self.output_dtype=base.weight.dtype;self.refactor(base.weight.detach())
 def svd(self):return {k:getattr(self,k) for k in ["U_high","S_high","V_high","U_low","S_low","V_low"]}|{"rank_high":self.U_high.shape[1]}
 def refactor(self,weight):
  k=int(min(weight.shape)*.75);s=self.kernel["create_svd_dict"](weight,k,upcast_dtype=torch.float32,output_dtype=torch.float32)
  for name in ["U_high","S_high","V_high"]:
   if name in self._buffers:self._buffers[name]=s[name]
   else:self.register_buffer(name,s[name])
  for name in ["U_low","S_low","V_low"]:setattr(self,name,s[name])
 def forward(self,x):
  w=self.kernel["reconstruct_weight_matrix"](self.svd(),torch.float32,self.output_dtype)
  return F.linear(x,w,self.bias)
class OSFTMethod:
 def __init__(self,model,name,instructions,embedding=None):
  self.model=model;self.name=name;self.layers=[];self.embedding=None;self.stage=0;self.learning_rate=1e-5;self.host_optimizer=True;self.offload_activations=True
  self.kernel,self.original_revision,self.source_hash=author_functions()
  for p in model.parameters():p.requires_grad=False
  targets={"q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"}
  candidates=[(k,v) for k,v in model.named_modules() if k.startswith("model.layers.") and k.rsplit(".",1)[-1] in targets and isinstance(v,nn.Linear)]
  for i,(key,base) in enumerate(candidates):
   parent,child=key.rsplit(".",1);layer=OSFTLinear(base,self.kernel);setattr(model.get_submodule(parent),child,layer);self.layers.append(layer)
   print(json.dumps({"event":"osft_svd_init","layer":key,"done":i+1,"total":len(candidates),"unfreeze_rank_ratio":.25}),flush=True)
 def set_instruction(self,task):pass
 def begin_stage(self,stage):
  if self.stage>0 and stage!=self.stage:
   for i,layer in enumerate(self.layers):
    with torch.no_grad():w=self.kernel["reconstruct_weight_matrix"](layer.svd(),torch.float32,layer.output_dtype).detach()
    layer.refactor(w);print(json.dumps({"event":"osft_stage_svd","stage":stage,"done":i+1,"total":len(self.layers)}),flush=True)
  self.stage=stage
 def post_backward(self,scale,progress):
  with torch.no_grad():
   for layer in self.layers:self.kernel["project_gradient_to_orthogonal_space"](layer.svd())
 def after_update(self,step,updates):
  with torch.no_grad():
   for layer in self.layers:self.kernel["project_parameter_to_orthogonal_space"](layer.svd())
 def method_state(self,stage):return {"method":self.name,"stage":stage,"upstream_revision":self.original_revision,"unfreeze_rank_ratio":.25,"qwen2_vl_decoder_adaptation":True}
