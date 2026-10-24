"""Two established rank-limited readouts; retaining nonlinear features is a hypothesis."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F


class Carrier(nn.Module):
    def __init__(self,original,left,right):
        super().__init__()
        self.gate_proj=original.gate_proj
        self.up_proj=original.up_proj
        self.act_fn=original.act_fn
        dtype=original.gate_proj.weight.dtype
        self.left=nn.Parameter(left.detach().clone().to(dtype),requires_grad=False)
        self.right=nn.Parameter(right.detach().clone().to(dtype),requires_grad=False)

    def forward(self,x):
        z=self.act_fn(self.gate_proj(x))*self.up_proj(x)
        return F.linear(F.linear(z,self.right),self.left)


def initialize(weight,moment,rank=200):
    """All decomposition/statistic arithmetic is CPU FP32, independent of autocast."""
    with torch.autocast('cpu',enabled=False):
        d=weight.detach().to('cpu',dtype=torch.float32)
        assert torch.isfinite(d).all() and d.ndim==2
        u,s,v=torch.linalg.svd(d,full_matrices=False)
        root=s[:rank].sqrt()
        weight_parts=(u[:,:rank]*root,root[:,None]*v[:rank])
        sym=(moment.float()+moment.float().T)*.5
        assert sym.device.type=='cpu' and torch.isfinite(sym).all()
        eigenvalues,eigenvectors=torch.linalg.eigh(sym)
        l=eigenvectors[:,-rank:].flip(1)
        c=l.T@d
        uc,sc,vc=torch.linalg.svd(c,full_matrices=False)
        rootc=sc.sqrt()
        response_parts=((l@uc)*rootc,rootc[:,None]*vc)
        parts={'WeightD':weight_parts,'ResponseD':response_parts}
        assert all(torch.isfinite(t).all() for pair in parts.values() for t in pair)
        evidence=dict(moment_trace=float(sym.trace()),eigenvalues=eigenvalues.flip(0).tolist(),
                      weight_singular_values=s.tolist(),response_singular_values=sc.tolist(),
                      moment_dtype=str(sym.dtype),decomposition_device=str(d.device))
        return {name:tuple(t.detach().clone() for t in pair) for name,pair in parts.items()},evidence


def add_qa_moment(moment,h,qa_count=32):
    assert h.ndim==2 and 0<len(h)<=64 and h.device.type=='cpu'
    with torch.autocast('cpu',enabled=False):
        hf=h.float();assert torch.isfinite(hf).all()
        moment.add_(hf.T@hf,alpha=1./(len(hf)*qa_count))
        assert torch.isfinite(moment).all()


def install(model,values):
    assert len(values)==len(model.model.layers)
    for layer,parts in zip(model.model.layers,values):
        assert not isinstance(layer.mlp,Carrier)
        layer.mlp=Carrier(layer.mlp,*[t.to(layer.mlp.gate_proj.weight.device) for t in parts])
    assert not any(n.endswith('down_proj.weight') for n,_ in model.core.named_parameters())


def digest(items):
    out=hashlib.sha256()
    for name,t in sorted(items):
        t=t.detach().cpu().contiguous()
        out.update(name.encode());out.update(str((t.dtype,tuple(t.shape))).encode())
        out.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
    return out.hexdigest()


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
