"""Independent Qwen port of MedQwen equations 11,21,22,24; r is total rank.

Sources: arxiv.org/html/2604.01310v1. Eight experts, top-2, rho=10,
eta=.1; q/v targets share the comparison protocol. No author code is available.
Residual compensation is evaluated as a frozen low-rank mean, algebraically
equal to subtracting the constant dense matrix in equation 21.
"""
import math
from pathlib import Path
import torch
from torch import nn

class SpectralLinear(nn.Module):
    def __init__(self,base,cfg,name,spectral):
        super().__init__();self.base=base;self.cfg=cfg;self.name=name;self.spectral=spectral
        self.adapters=nn.ModuleList();self.dropout=nn.Dropout(cfg['dropout'])
        self.gate=None;self.balance=None;self.scale=math.sqrt(3*base.in_features*.1)
    def initialize(self):
        device=self.base.weight.device;n=self.base.in_features;m=self.base.out_features
        self.gate=nn.Linear(n,8,bias=False,device=device,dtype=torch.float32)
        nn.init.normal_(self.gate.weight,std=.01)
        if self.spectral:
            cache=Path(__file__).resolve().parent/'svd_cache';cache.mkdir(exist_ok=True)
            path=cache/(self.name.replace('.','_')+'.pt')
            if path.exists():parts=torch.load(path,map_location='cpu',weights_only=True)
            else:
                print('SVD',self.name,flush=True)
                u,s,vh=torch.linalg.svd(self.base.weight.detach().cpu().float(),full_matrices=False)
                parts=[]
                for i in range(8):
                    k=i*(min(m,n)//8);factor=math.sqrt(float(s[k])/(self.scale*10))
                    parts.append(dict(down=vh[k:k+1]*factor,up=u[:,k:k+1]*factor))
                torch.save(parts,path)
        for i in range(8):
            a=nn.ModuleDict({'down':nn.Linear(n,1,bias=False,device=device,dtype=torch.float32),
                             'up':nn.Linear(1,m,bias=False,device=device,dtype=torch.float32)})
            if self.spectral:
                a['down'].weight.data.copy_(parts[i]['down']);a['up'].weight.data.copy_(parts[i]['up'])
            else:nn.init.kaiming_uniform_(a['down'].weight,a=math.sqrt(5));nn.init.zeros_(a['up'].weight)
            a.register_buffer('initial_down',a['down'].weight.detach().clone())
            a.register_buffer('initial_up',a['up'].weight.detach().clone());self.adapters.append(a)
    def forward(self,x):
        result=self.base(x)
        if not self.adapters:return result
        shape=x.shape;raw=x.reshape(-1,shape[-1]).float()
        # Router arithmetic stays float32 even under the shared fp16 autocast.
        with torch.autocast(device_type=x.device.type,enabled=False):
            probabilities=self.gate(raw).softmax(-1)
            values,indices=probabilities.topk(2,dim=-1);values=values/values.sum(-1,keepdim=True)
            gates=torch.zeros_like(probabilities).scatter(1,indices,values)
            density=torch.nn.functional.one_hot(indices,8).float().mean((0,1))
            self.balance=8*(density*probabilities.mean(0)).sum()
            adapted=self.dropout(raw);update=torch.zeros((raw.shape[0],self.base.out_features),device=x.device)
            for i,a in enumerate(self.adapters):
                update.add_(a['up'](a['down'](adapted))*gates[:,i:i+1]*self.scale)
                if self.spectral:
                    update.sub_(torch.nn.functional.linear(torch.nn.functional.linear(raw,a.initial_down),a.initial_up)*self.scale/8)
        return result+update.reshape(*shape[:-1],self.base.out_features).to(result.dtype)

class SpectralLoRA:
    def __init__(self,model,cfg,spectral=True):
        self.model=model;self.cfg=cfg;self.spectral=spectral;self.task_count=0;self.layers={}
        model.requires_grad_(False)
        for name,module in list(model.named_modules()):
            if name.rsplit('.',1)[-1] in cfg['targets']:
                parent,child=name.rsplit('.',1);layer=SpectralLinear(module,cfg,name,spectral)
                setattr(model.get_submodule(parent),child,layer);self.layers[name]=layer
        assert len(self.layers)==56,len(self.layers)
    def begin_task(self):
        if not self.task_count:
            for layer in self.layers.values():layer.initialize()
        self.task_count+=1
    def parameters(self):return [p for p in self.model.parameters() if p.requires_grad]
    def prepare_batch(self,rows,tokenizer,batch,training):
        for layer in self.layers.values():layer.balance=None
    def penalty(self,orthogonal_weight,l2_weight):
        values=[l.balance for l in self.layers.values() if l.balance is not None]
        return sum(values)/len(values)*.001 if values else 0.
    def save(self,path):
        torch.save(dict(task_count=self.task_count,layers={name:{k:v.detach().cpu() for k,v in layer.state_dict().items()
                   if not k.startswith('base.')} for name,layer in self.layers.items()}),path)
    def load(self,path):
        saved=torch.load(path,map_location='cpu',weights_only=True)
        if saved['task_count'] and not self.task_count:self.begin_task()
        self.task_count=saved['task_count']
        for name,layer in self.layers.items():
            missing,unexpected=layer.load_state_dict(saved['layers'][name],strict=False)
            assert not unexpected and all(k.startswith('base.') for k in missing)
