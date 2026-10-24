"""A frozen independent vision prior with a single affine spatial interface."""
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer

def student(config):
 cfg=Idefics3VisionConfig(**config);cfg._attn_implementation='eager'
 return Idefics3VisionTransformer(cfg)

def spatial_features(tower,pixels,native_size=512,output_grid=27):
 assert pixels.ndim==4 and pixels.shape[1]==3
 resized=F.interpolate(pixels.float(),size=(native_size,native_size),mode='bicubic',align_corners=False,antialias=False).to(next(tower.parameters()).dtype)
 features=tower(resized,output_hidden_states=False,return_dict=True).last_hidden_state
 grid=native_size//tower.config.patch_size
 assert features.shape[1]==grid*grid
 mapped=F.interpolate(features.transpose(1,2).reshape(features.shape[0],features.shape[2],grid,grid).float(),size=(output_grid,output_grid),mode='bilinear',align_corners=False,antialias=False)
 return mapped.flatten(2).transpose(1,2).contiguous().to(features.dtype)

class VisionInterface(nn.Module):
 def __init__(self,tower,output_width=896,native_size=512,output_grid=27):
  super().__init__();self.student=tower;self.affine=nn.Linear(tower.config.hidden_size,output_width,bias=True)
  self.native_size=native_size;self.output_grid=output_grid
 def forward(self,pixel_values,output_hidden_states=True,**kwargs):
  features=self.affine(spatial_features(self.student,pixel_values,self.native_size,self.output_grid))
  return SimpleNamespace(last_hidden_state=features,hidden_states=(features,))

def install(model,interface):
 # Factory's PackedVision also owns registered references. Rebind both aliases.
 assert not hasattr(model,'_llmcl_visual_original'), 'Install before frozen_vision.install'
 model.core.vision_tower=interface;model.core.multi_modal_projector=nn.Identity()
 model.visual.tower=interface;model.visual.projector=model.core.multi_modal_projector
 assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
 model.requires_grad_(False);model.eval()

class Moments:
 def __init__(self,dx,dy):
  self.count=0;self.sx=torch.zeros(dx,dtype=torch.float64);self.sy=torch.zeros(dy,dtype=torch.float64)
  self.xx=torch.zeros(dx,dx,dtype=torch.float64);self.xy=torch.zeros(dx,dy,dtype=torch.float64)
 def add(self,x,y):
  x=x.detach().reshape(-1,self.sx.numel()).to(device='cpu',dtype=torch.float64)
  y=y.detach().reshape(-1,self.sy.numel()).to(device='cpu',dtype=torch.float64)
  assert len(x)==len(y) and len(x)>0 and torch.isfinite(x).all() and torch.isfinite(y).all()
  n=len(x);self.count+=1;self.sx+=x.mean(0);self.sy+=y.mean(0);self.xx+=(x.T@x)/n;self.xy+=(x.T@y)/n
 def solve(self):
  assert self.count>0
  mx=self.sx/self.count;my=self.sy/self.count
  cxx=self.xx/self.count-mx[:,None]*mx[None,:];cxy=self.xy/self.count-mx[:,None]*my[None,:]
  cxx=(cxx+cxx.T)/2;ridge=1e-3*torch.trace(cxx)/len(mx)
  assert torch.isfinite(ridge) and ridge>0
  b=torch.linalg.solve(cxx+ridge*torch.eye(len(mx),dtype=torch.float64),cxy);bias=my-mx@b
  assert torch.isfinite(b).all() and torch.isfinite(bias).all()
  return b.T.contiguous(),bias,float(ridge)
