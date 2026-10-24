"""Physical single-branch rank replacement and ordinary q/v folding."""
import hashlib
import torch
from torch import nn


class FactorLinear(nn.Module):
    def __init__(self, left, right):
        super().__init__()
        self.left = nn.Parameter(left.detach().clone(), requires_grad=False)
        self.right = nn.Parameter(right.detach().clone(), requires_grad=False)
        self.out_features, self.in_features = left.shape[0], right.shape[1]
    def forward(self, x):
        return torch.nn.functional.linear(torch.nn.functional.linear(x, self.right), self.left)


def digest(items):
    h = hashlib.sha256()
    for name, value in sorted(items):
        v = value.detach().contiguous().cpu()
        h.update(name.encode()); h.update(str((v.dtype, tuple(v.shape))).encode())
        h.update(v.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def layout(model):
    params = list(model.core.named_parameters())
    buffers = list(model.core.named_buffers())
    def measure(items):
        seen = set(); n = b = 0
        for _, t in items:
            key = (t.untyped_storage().data_ptr(), t.untyped_storage().nbytes())
            if key in seen: continue
            seen.add(key)
            assert t.storage_offset() == 0 and t.numel()*t.element_size() == t.untyped_storage().nbytes()
            n += t.numel(); b += t.numel()*t.element_size()
        return dict(coefficients=n, bytes=b)
    return dict(weights=measure(params), ordinary_buffers=measure(buffers),
                trainable=sum(p.numel() for _,p in params if p.requires_grad))


def state_cpu(core):
    out, seen = {}, {}
    for n,t in core.state_dict().items():
        key=(t.untyped_storage().data_ptr(),t.storage_offset(),tuple(t.shape),tuple(t.stride()),t.dtype)
        if key not in seen: seen[key]=t.detach().cpu().clone()
        out[n]=seen[key]
    return out


def replace(model, case, rank=200, factors=None):
    branch = {'KeepGate':'up_proj', 'KeepUp':'gate_proj'}[case]
    stash, evidence = [], []
    for i,layer in enumerate(model.model.layers):
        base=getattr(layer.mlp,branch)
        if factors is None:
            u,s,vh=torch.linalg.svd(base.weight.detach().float(),full_matrices=False,driver='gesvd' if base.weight.is_cuda else None)
            root=s[:rank].sqrt()
            left=(u[:,:rank]*root).to(base.weight.dtype)
            right=(root[:,None]*vh[:rank]).to(base.weight.dtype)
        else:
            left,right=factors[i]
        assert left.shape==(base.out_features,rank) and right.shape==(rank,base.in_features)
        assert torch.isfinite(left).all() and torch.isfinite(right).all()
        item=FactorLinear(left,right)
        stash.append((layer.mlp,branch,base)); setattr(layer.mlp,branch,item)
        evidence.append(dict(layer=i,branch=branch,shapes=[list(left.shape),list(right.shape)],
                             sha256=digest([('left',left),('right',right)])))
    return stash,evidence


def restore(stash):
    for module,name,base in stash:setattr(module,name,base)


def fold(model):
    from methods import SeqLinear
    result=[]
    for name,module in list(model.named_modules()):
        if not isinstance(module,SeqLinear):continue
        n,m=module.weight.shape
        linear=nn.Linear(m,n,bias=module.bias is not None,device=module.weight.device,dtype=module.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_((module.weight.float()+module.scale*(module.B.weight.float()@module.A.weight.float())).to(module.weight.dtype))
            if module.bias is not None:linear.bias.copy_(module.bias)
        linear.requires_grad_(False)
        prefix,child=name.rsplit('.',1);setattr(model.get_submodule(prefix),child,linear);result.append(name)
    return result
