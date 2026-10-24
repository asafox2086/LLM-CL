"""Two physical up-function carriers. Generic sharing is not a novelty claim."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class Carrier(nn.Module):
    def __init__(self, original, left, right, row_scale=None, master=False):
        super().__init__()
        self.gate_proj=original.gate_proj
        self.down_proj=original.down_proj
        self.act_fn=original.act_fn
        dtype=torch.float32 if master else original.gate_proj.weight.dtype
        self.left=nn.Parameter(left.detach().clone().to(dtype),requires_grad=master)
        self.right=nn.Parameter(right.detach().clone().to(dtype),requires_grad=master)
        self.register_parameter('row_scale',None if row_scale is None else nn.Parameter(row_scale.detach().clone().to(dtype),requires_grad=master))

    def forward(self,x):
        gate=self.gate_proj(x)
        up=F.linear(F.linear(x,self.right),self.left)
        if self.row_scale is not None:up=up+gate*self.row_scale.to(gate.dtype)
        return self.down_proj(self.act_fn(gate)*up)


def initialize(original,case,rank=None):
    rank=rank or {'Static':200,'Reuse':199}[case]
    g=original.gate_proj.weight.detach().float()
    u=original.up_proj.weight.detach().float()
    d=(u*g).sum(1)/g.square().sum(1).clamp_min(1e-12) if case=='Reuse' else None
    residual=u if d is None else u-d[:,None]*g
    q,s,v=torch.linalg.svd(residual,full_matrices=False,driver='gesvd' if residual.is_cuda else None)
    root=s[:rank].sqrt()
    values=(q[:,:rank]*root,root[:,None]*v[:rank],d)
    assert all(torch.isfinite(x).all() for x in values if x is not None)
    return [None if x is None else x.detach().cpu() for x in values]


def install(model,case,values,master=False):
    assert len(values)==len(model.model.layers)
    for layer,parts in zip(model.model.layers,values):
        assert not isinstance(layer.mlp,Carrier)
        layer.mlp=Carrier(layer.mlp,*[None if x is None else x.to(layer.mlp.gate_proj.weight.device) for x in parts],master=master)
    assert not any(n.endswith('up_proj.weight') for n,_ in model.core.named_parameters())


def digest(items):
    h=hashlib.sha256()
    for name,value in sorted(items):
        t=value.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((t.dtype,tuple(t.shape))).encode())
        h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def layout(model):
    def count(items):
        seen=set();n=b=0
        for _,t in items:
            st=t.untyped_storage();key=(str(t.device),st.data_ptr(),st.nbytes())
            if key in seen:continue
            seen.add(key)
            assert t.storage_offset()==0 and t.numel()*t.element_size()==st.nbytes()
            n+=t.numel();b+=st.nbytes()
        return dict(coefficients=n,bytes=b)
    return dict(weights=count(model.core.named_parameters()),ordinary_buffers=count(model.core.named_buffers()),
                trainable=sum(p.numel() for p in model.core.parameters() if p.requires_grad))


def cpu_state(model):
    out={};seen={}
    for n,t in model.core.state_dict().items():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype)
        if key not in seen:seen[key]=t.detach().cpu().clone()
        out[n]=seen[key]
    return out
