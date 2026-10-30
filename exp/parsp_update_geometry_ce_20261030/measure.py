"""Read/capture native PaRSP hooks; no alternative gradients or optimizer."""
import torch

def energy(x):return float(x.double().square().sum())
def geometry(x,mask,P):
 x=x.detach().double();m=mask.double();p=P.double();e=energy(x);outside=x*(1-m);rp=x-x@p
 index=int(outside.abs().reshape(-1).argmax());ij=[index//x.shape[1],index%x.shape[1]] if bool((outside!=0).any()) else None
 return dict(squared_norm=e,outside_M_squared_norm=energy(outside),outside_M_fraction=energy(outside)/e if e else None,
  outside_M_nonzero_coefficients=int((outside!=0).sum()),outside_M_max_abs=float(outside.abs().max()),outside_M_max_index=ij,
  residual_to_P_squared_norm=energy(rp),residual_to_P_fraction=energy(rp)/e if e else None,
  P_residual_is_not_an_orthogonal_energy_partition=True)

def projector(P,mask,axis):
 p=P.double();e=energy(p);sym=energy(p-p.T);idem=energy(p@p-p)
 comm=None
 if axis==1:
  m=mask.reshape(-1).double();comm=energy((m[:,None]-m[None,:])*p)
 return dict(shape=list(P.shape),source_dtype=str(P.dtype),symmetry_residual_squared=sym,
  idempotence_residual_squared=idem,P_squared_norm=e,symmetry_relative_squared=sym/e if e else None,
  idempotence_relative_squared=idem/e if e else None,column_mask_commutator_squared_norm=comm,
  intersection_rank_measured=False)

class NativeCapture:
 def __init__(self,parameter,expected_P,axis):
  assert parameter.dtype==torch.float32 and parameter.is_cuda
  self.parameter=parameter;self.axis=axis;self.original=list(parameter._backward_hooks.items());assert len(self.original)==2
  (_,maskfn),(_,projectfn)=self.original;assert maskfn.__name__=='hook_fn' and projectfn.__name__=='project'
  env=dict(zip(maskfn.__code__.co_freevars,[c.cell_contents for c in maskfn.__closure__]));self.mask=env['mask'].detach()
  assert projectfn.__defaults__ and projectfn.__defaults__[0] is expected_P
  self.P=expected_P.detach().to(parameter.device);assert self.P.shape==(parameter.shape[1],parameter.shape[1])
  assert self.mask.shape==((parameter.shape[0],1) if axis==0 else (parameter.shape[1],))
  assert bool(((self.mask==0)|(self.mask==1)).all())
  self.sum={k:torch.zeros_like(parameter,dtype=torch.float64) for k in ['raw','masked','projected']}
  self.abs_projected=torch.zeros_like(parameter,dtype=torch.float64);self.calls=[];self.micro=None;self.pending=None
  def first(grad):
   assert grad.dtype==torch.float32 and grad.shape==parameter.shape and self.pending is None
   self.sum['raw'].add_(grad.detach().double());out=maskfn(grad)
   self.sum['masked'].add_(out.detach().double());self.pending=(self.micro,out.detach().clone());return out
  def second(grad):
   assert self.pending is not None and self.pending[0]==self.micro and torch.equal(grad,self.pending[1])
   out=projectfn(grad);self.sum['projected'].add_(out.detach().double());self.abs_projected.add_(out.detach().double().abs())
   self.calls.append(self.micro);self.pending=None;return out
  parameter._backward_hooks[self.original[0][0]]=first;parameter._backward_hooks[self.original[1][0]]=second
 def close(self):
  for key,fn in self.original:self.parameter._backward_hooks[key]=fn
 def accumulation_certificate(self,actual,scale):
  assert self.calls==list(range(16)) and self.pending is None and scale==1.
  expected=self.sum['projected']/scale;diff=actual.double()-expected
  unit=torch.finfo(torch.float32).eps/2;gamma=15*unit/(1-15*unit)
  bound=gamma*self.abs_projected/scale+32*torch.finfo(torch.float32).tiny
  return dict(micro_hook_calls=len(self.calls),raw_difference_squared=energy(diff),
   max_abs_difference=float(diff.abs().max()),max_roundoff_bound=float(bound.max()),
   elementwise_bound_violations=int((diff.abs()>bound).sum()),bound='gamma_15(sum absolute FP32 post-P micrograds) + 32*FP32tiny',scale=float(scale))
 def report(self,before,after,preclip,postclip,scale):
  delta=after.detach().double()-before.double();g=self.sum['raw']/scale;e=energy(delta)
  dot=float((g*delta).sum());gn=energy(g)
  out=dict(axis=self.axis,shape=list(before.shape),weight_dtype=str(before.dtype),mask_selected=int(self.mask.sum()),
   projector=projector(self.P,self.mask,self.axis),accumulation=self.accumulation_certificate(preclip,scale),
   gradients={k:geometry(v/scale,self.mask,self.P) for k,v in self.sum.items()},
   native_gradient_preclip=geometry(preclip,self.mask,self.P),native_gradient_postclip=geometry(postclip,self.mask,self.P),
   actual_Adam_deltaW=geometry(delta,self.mask,self.P),
   raw_current_CE_gradient_dot_actual_deltaW=dot,
   raw_current_CE_gradient_dot_unit_norm_actual_deltaW=dot/e**.5 if e else None,
   raw_gradient_delta_cosine=dot/(gn*e)**.5 if gn and e else None,
   directional_inner_products_are_target_components_not_full_loss_change=True)
  index=out['actual_Adam_deltaW']['outside_M_max_index']
  if index is None:out['max_outside_M_coordinate']=None
  else:
   i,j=index;out['max_outside_M_coordinate']=dict(index=[i,j],mask=float(self.mask[i,0] if self.axis==0 else self.mask[j]),weight_before=float(before[i,j]),weight_after=float(after[i,j]),actual_delta=float(delta[i,j]))
  return out
