"""Bound wide-q/v initialization,24 response scalars and GPU-only Q/V fold."""
from pathlib import Path
import importlib.util

COORDINATES=4399128
PARAMETERS=120

def configure_gpu_math():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def make_wide(model,asset,storage):
    import torch
    from response import metadata
    path=Path(asset)/'exp/position_coordinate_control_20261022/v1/core.py'
    spec=importlib.util.spec_from_file_location('response_budget',path);budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
    method=budget.BudgetQV(model,budget.rank_plan())
    method.host_optimizer=False;method.offload_activations=False;method.vision_original=None
    assert len(method.layers)==48
    for layer in model.model.layers:layer.mlp.beta.requires_grad_(True)
    method.initialization=[]
    for name,rank in method.rank_plan.items():
        layer=model.get_submodule(name)
        assert layer.scale==4 and layer.dropout.p==.1 and layer.weight.dtype==torch.float16
        assert layer.A.weight.dtype==layer.B.weight.dtype==torch.float32 and layer.A.weight.is_cuda and layer.B.weight.is_cuda
        assert torch.count_nonzero(layer.B.weight)==0 and torch.isfinite(layer.A.weight).all()
        method.initialization.append(dict(module=name,rank=rank,scale=4,dropout=.1,A_digest=storage.digest({'A':layer.A.weight}),B_digest=storage.digest({'B':layer.B.weight}),B_zero=True))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    expected={n+s for n in method.rank_plan for s in ('.A.weight','.B.weight')}|{f'model.layers.{i}.mlp.beta' for i in range(24)}
    assert set(trainable)==expected and len(trainable)==PARAMETERS
    assert sum(p.numel() for p in trainable.values())==COORDINATES
    assert all(p.is_cuda and p.dtype==torch.float32 for p in trainable.values())
    assert all(torch.count_nonzero(layer.mlp.beta)==0 for layer in model.model.layers)
    assert not metadata(model)['final']
    return method

def fold(model,method):
    import torch
    assert len(method.rank_plan)==48 and not torch.backends.cuda.matmul.allow_tf32
    records=[];model.eval()
    for name,rank in method.rank_plan.items():
        layer=model.get_submodule(name);device=layer.weight.device
        assert rank in (65,66) and layer.scale==4 and device.type=='cuda'
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            weight=(layer.weight.detach().float()+layer.scale*(layer.B.weight.detach().float()@layer.A.weight.detach().float())).half()
            assert torch.isfinite(weight).all()
            linear=torch.nn.Linear(layer.weight.shape[1],layer.weight.shape[0],bias=layer.bias is not None,device=device,dtype=torch.float16)
            linear.weight.copy_(weight)
            if layer.bias is not None:linear.bias.copy_(layer.bias.detach())
        parent,child=name.rsplit('.',1);setattr(model.get_submodule(parent),child,linear)
        records.append(dict(module=name,rank=rank,scale=4,dtype='torch.float16',merge_dtype='torch.float32',device=str(device),autocast_enabled=False,matmul_allow_tf32=False))
    method.layers.clear();model.requires_grad_(False).eval()
    assert not any(n.endswith(('.A.weight','.B.weight')) for n in model.core.state_dict())
    return records
