"""DRAFT single effective-G intervention. Original forward AST differs by one callback.
No bank/weight/dtype/expert scale edits; feature passes still execute original disabled path.
"""
import ast,copy,hashlib,types,math,json
import torch

def dump(node):return ast.dump(node,include_attributes=False)
def build_forward(source_path,expected_sha,original):
 raw=source_path.read_bytes();assert hashlib.sha256(raw).hexdigest()==expected_sha
 tree=ast.parse(raw);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CLMoELinear')
 f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');g=copy.deepcopy(f)
 # Confirm the loaded CP forward is the reviewed source function, not a rebuilt approximation.
 ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(f)],type_ignores=[])),str(source_path),'exec'),original.__globals__,ns)
 ref=ns['forward'].__code__;got=original.__code__
 assert (got.co_code,got.co_names,got.co_varnames,got.co_consts)==(ref.co_code,ref.co_names,ref.co_varnames,ref.co_consts),'CP forward/source code mismatch'
 block=next(n for n in g.body if isinstance(n,ast.With));idx=next(i for i,n in enumerate(block.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='delta' for t in n.targets))
 assert isinstance(block.body[idx-1],ast.If) and 'global_weight' in dump(block.body[idx-1])
 addition=ast.parse('w = self._cl_fixed_effective_controller.apply(self,w)').body[0]
 block.body.insert(idx,addition)
 proof=copy.deepcopy(g);next(n for n in proof.body if isinstance(n,ast.With)).body.pop(idx);assert dump(proof)==dump(f)
 out={};exec(compile(ast.fix_missing_locations(ast.Module(body=[g],type_ignores=[])),str(source_path)+'::fixed_effective_G','exec'),original.__globals__,out)
 return out['forward'],dict(original_forward_AST_sha256=hashlib.sha256(dump(f).encode()).hexdigest(),only_added_AST=dump(addition),roundtrip_original_AST_equal=True,loaded_CP_bytecode_matches_source=True)

class GateIntervention:
 def __init__(self,method,c_by_layer,source_path,source_sha256):
  self.method=method;self.model=method.model;self.layers={n:m for n,m in self.model.core.named_modules() if type(m).__name__=='CLMoELinear'}
  assert len(self.layers)==168 and set(self.layers)==set(c_by_layer)
  assert not self.model.training and all(m.enabled and not m.collect and not m.training for m in self.layers.values())
  self.names={id(m):n for n,m in self.layers.items()};self.original={n:m.forward for n,m in self.layers.items()}
  self.forward,self.AST_proof=build_forward(source_path,source_sha256,next(iter(self.original.values())).__func__)
  for f in self.original.values():
   x=f.__func__.__code__;y=next(iter(self.original.values())).__func__.__code__;assert (x.co_code,x.co_names,x.co_consts)==(y.co_code,y.co_names,y.co_consts)
  self.C={};self.C_initial={};self.coefficient_records={}
  for n,m in self.layers.items():
   c=c_by_layer[n];assert len(c)==8 and all(isinstance(x,(int,float)) and math.isfinite(x) and x>=0 for x in c)
   original=torch.tensor(c,dtype=torch.float64);v=torch.tensor(c,device=m.weight.device,dtype=torch.float32);assert bool(torch.isfinite(v).all())
   self.C[n]=v;self.C_initial[n]=v.clone();cast=v.double().cpu()
   self.coefficient_records[n]=dict(C64=c,C32=cast.tolist(),max_absolute_C64_to_C32=float((cast-original).abs().max()),sum_C64=float(original.sum()),sum_C32=float(cast.sum()),renormalized=False,expert_scale2_not_in_C=True)
  self.mode='idle';self.call_type=None;self.installed=False;self.handles=[];self.reference_outputs={};self.step=0;self.main_seen=set();self.output_seen=set()
  self.counts={k:0 for k in ['control_original_main_calls','control_passthrough_main_calls','fixed_main_calls','adapter_disabled_feature_calls','full_output_digest_comparisons']};self.fixed_per_layer={n:0 for n in self.layers}
  dec=self.model.core.language_model.model
  self.handles.append(dec.register_forward_pre_hook(self.decoder_pre));self.handles.append(dec.register_forward_hook(self.decoder_post))
  for n,m in self.layers.items():self.handles.append(m.register_forward_hook(lambda mod,args,out,n=n:self.output(n,out)))
 def install(self):
  assert not self.installed
  for n,m in self.layers.items():
   object.__setattr__(m,'_cl_fixed_effective_controller',self);m.forward=types.MethodType(self.forward,m)
  self.installed=True
 def uninstall(self):
  if not self.installed:return
  for n,m in self.layers.items():m.forward=self.original[n];object.__delattr__(m,'_cl_fixed_effective_controller')
  self.installed=False
 def begin(self,mode):
  assert mode in ['original','passthrough','fixed'] and self.call_type is None
  assert self.installed==(mode!='original');self.mode=mode;self.step=0
 def decoder_pre(self,module,args):
  assert self.call_type is None
  enabled=[m.enabled for m in self.layers.values()]
  if not any(enabled):self.call_type='feature';self.counts['adapter_disabled_feature_calls']+=1
  else:
   assert all(enabled) and all(not m.collect and not m.training for m in self.layers.values()) and self.mode!='idle'
   self.call_type='main';self.step+=1;self.counts[{'original':'control_original_main_calls','passthrough':'control_passthrough_main_calls','fixed':'fixed_main_calls'}[self.mode]]+=1
  self.main_seen=set();self.output_seen=set()
 def apply(self,m,w):
  assert self.call_type=='main' and self.mode in ['passthrough','fixed'] and m.global_weight is not None
  n=self.names[id(m)];assert n not in self.main_seen;self.main_seen.add(n)
  assert w.dtype is torch.float32 and w.ndim==3 and w.shape[0]==1 and w.shape[-1]==8
  if self.mode=='passthrough':return w # same Tensor, no arithmetic, dtype conversion or normalization
  c=self.C[n];out=c.view(1,1,8).expand_as(w)
  assert out.dtype==w.dtype and out.device==w.device and out.data_ptr()==c.data_ptr()
  self.fixed_per_layer[n]+=1
  return out
 @staticmethod
 def output_digest(out):
  # Byte I/O only, not a CPU neural forward; never retain full GPU output references.
  payload=out.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
  return dict(shape=list(out.shape),dtype=str(out.dtype),sha256=hashlib.sha256(payload).hexdigest())
 def output(self,n,out):
  if self.call_type!='main':return
  assert n not in self.output_seen;self.output_seen.add(n)
  key=(self.step,n)
  if self.mode=='original':assert key not in self.reference_outputs;self.reference_outputs[key]=self.output_digest(out)
  elif self.mode=='passthrough':
   expected=self.reference_outputs.pop(key);assert expected==self.output_digest(out),'Dynamic-preserving full-output byte digest differs: '+n
   self.counts['full_output_digest_comparisons']+=1
 def decoder_post(self,module,args,out):
  if self.call_type=='main':
   assert self.output_seen==set(self.layers)
   assert self.main_seen==(set() if self.mode=='original' else set(self.layers))
  else:assert self.call_type=='feature' and not self.main_seen
  self.call_type=None
 def end_control(self):
  assert self.call_type is None and not self.reference_outputs;self.mode='idle'
 def report(self):
  assert self.call_type is None
  assert all(torch.equal(v,self.C_initial[n]) for n,v in self.C.items())
  if self.counts['fixed_main_calls']:assert set(self.fixed_per_layer.values())=={self.counts['fixed_main_calls']}
  return dict(AST_proof=self.AST_proof,counts=self.counts,fixed_per_layer=self.fixed_per_layer,coefficients=self.coefficient_records,
   constant_tensor_bytes=sum(v.numel()*v.element_size() for v in self.C.values()),fixed_main_coverage_complete=self.counts['fixed_main_calls']>0,
   bank_weights_dtype_and_scale_unchanged=True,original_token_router_and_anchor_still_computed=True)
 def close(self):
  self.uninstall()
  for h in self.handles:h.remove()
  self.handles=[];self.reference_outputs.clear()
