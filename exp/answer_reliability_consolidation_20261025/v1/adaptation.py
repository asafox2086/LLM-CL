"""Shared-base q/v-r8 working and frozen slow buffers; GPU-only EMA/fold.

The working initializer, configure_gpu_math and fold arithmetic are inherited
from the sealed plain-r8 control. Slow routing never overwrites a working
Parameter whose autograd graph may still await backward.
"""
from pathlib import Path
import importlib.util
from contextlib import contextmanager
from types import MethodType

COORDINATES=540672
PARAMETERS=96

def configure_gpu_math():
    import torch
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')

def make_plain(model,asset,storage):
    import torch
    path=Path(asset)/'exp/position_coordinate_control_20261022/v1/core.py'
    spec=importlib.util.spec_from_file_location('response_budget',path);budget=importlib.util.module_from_spec(spec);spec.loader.exec_module(budget)
    method=budget.BudgetQV(model,{name:8 for name in budget.rank_plan()})
    assert all(item['rank']==8 and item['alpha']==32 and item['scale']==4 and item['prefix_r8_exact'] and item['extra_B_zero'] for item in method.initialization)
    method.host_optimizer=False;method.offload_activations=False;method.vision_original=None
    assert len(method.layers)==48
    method.initialization=[]
    for name,rank in method.rank_plan.items():
        layer=model.get_submodule(name)
        assert layer.scale==4 and layer.dropout.p==.1 and layer.weight.dtype==torch.float16
        assert layer.A.weight.dtype==layer.B.weight.dtype==torch.float32 and layer.A.weight.is_cuda and layer.B.weight.is_cuda
        assert torch.count_nonzero(layer.B.weight)==0 and torch.isfinite(layer.A.weight).all()
        method.initialization.append(dict(module=name,rank=rank,scale=4,dropout=.1,A_digest=storage.digest({'A':layer.A.weight}),B_digest=storage.digest({'B':layer.B.weight}),B_zero=True,prefix_r8_exact=True,alpha=32))
    trainable={n:p for n,p in model.named_parameters() if p.requires_grad}
    expected={n+s for n in method.rank_plan for s in ('.A.weight','.B.weight')}
    assert set(trainable)==expected and len(trainable)==PARAMETERS
    assert sum(p.numel() for p in trainable.values())==COORDINATES
    assert all(p.is_cuda and p.dtype==torch.float32 for p in trainable.values())
    assert all(not hasattr(layer.mlp,'beta') and not hasattr(layer.mlp,'response_sign')
               and not hasattr(layer.mlp,'calibration_scale') for layer in model.model.layers)
    install_slow(method)
    return method

def fold(model,method):
    import torch
    assert len(method.rank_plan)==48 and not torch.backends.cuda.matmul.allow_tf32
    records=[];model.eval()
    for name,rank in method.rank_plan.items():
        layer=model.get_submodule(name);device=layer.weight.device
        assert rank ==8 and layer.scale==4 and device.type=='cuda'
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


SLOW_A = '_ilora_slow_A'
SLOW_B = '_ilora_slow_B'


def _slow_routed_forward(layer, x):
    # The unchanged bound SeqLinear.forward is used on every working pass.
    if not layer._ilora_slow_active:
        return layer._ilora_working_forward(x)
    import torch.nn.functional as F
    base = F.linear(x, layer.weight, layer.bias)
    hidden = F.linear(layer.dropout(x.float()), getattr(layer, SLOW_A))
    return base + (F.linear(hidden, getattr(layer, SLOW_B)) * layer.scale).to(base.dtype)


def install_slow(method):
    import torch
    assert len(method.rank_plan) == 48 and len(method.layers) == 48
    assert not hasattr(method, 'ema_update_count')
    for name in method.rank_plan:
        layer = method.model.get_submodule(name)
        assert not any(hasattr(layer, key) for key in (SLOW_A, SLOW_B, '_ilora_working_forward', '_ilora_slow_active'))
        assert layer.A.weight.is_cuda and layer.B.weight.is_cuda
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        layer.register_buffer(SLOW_A, layer.A.weight.detach().clone(), persistent=True)
        layer.register_buffer(SLOW_B, layer.B.weight.detach().clone(), persistent=True)
        object.__setattr__(layer, '_ilora_working_forward', layer.forward)
        object.__setattr__(layer, '_ilora_slow_active', False)
        layer.forward = MethodType(_slow_routed_forward, layer)
    method.ema_update_count = 0
    method.ema_stage_sync_count = 0
    state = ema_state(method)
    assert len(state) == 96 and sum(x.numel() for x in state.values()) == COORDINATES
    assert sum(x.numel() * x.element_size() for x in state.values()) == 2162688
    assert all(not x.requires_grad and x.grad is None and x.is_cuda and x.dtype == torch.float32 for x in state.values())
    return method


def ema_state(method):
    """Exact actual registered names -> buffer references; no clone/CPU transfer."""
    result = {}
    for name in method.rank_plan:
        layer = method.model.get_submodule(name)
        result[name + '.' + SLOW_A] = getattr(layer, SLOW_A)
        result[name + '.' + SLOW_B] = getattr(layer, SLOW_B)
    assert len(result) == 96
    return result


def sync_slow(method):
    """Call once at each stage start, before any stage QA forward."""
    import torch
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for name in method.rank_plan:
            layer = method.model.get_submodule(name)
            assert not layer._ilora_slow_active
            getattr(layer, SLOW_A).copy_(layer.A.weight.detach())
            getattr(layer, SLOW_B).copy_(layer.B.weight.detach())
    method.ema_stage_sync_count += 1


def update_slow(method):
    """Call once after each successful optimizer step, never per micro-QA."""
    import torch
    with torch.no_grad(), torch.autocast('cuda', enabled=False):
        for name in method.rank_plan:
            layer = method.model.get_submodule(name)
            assert not layer._ilora_slow_active
            for slow_name, working in ((SLOW_A, layer.A.weight), (SLOW_B, layer.B.weight)):
                slow = getattr(layer, slow_name)
                assert slow.is_cuda and working.is_cuda and slow.device == working.device
                assert slow.dtype == working.dtype == torch.float32 and not slow.requires_grad
                slow.mul_(.25).add_(working.detach(), alpha=.75)
                assert torch.isfinite(slow).all()
    method.ema_update_count += 1


@contextmanager
def slow_mode(method):
    """No-grad teacher with unchanged train/dropout mode; always restores route.

    No working parameter data is swapped. The teacher uses the same base,
    vision and head, with only the q/v A/B route selecting detached buffers.
    """
    import torch
    layers = [method.model.get_submodule(name) for name in method.rank_plan]
    previous = [layer._ilora_slow_active for layer in layers]
    assert not any(previous), 'Nested slow routing is not part of this protocol'
    try:
        for layer in layers:
            layer._ilora_slow_active = True
        with torch.no_grad():
            yield
    finally:
        for layer, original in zip(layers, previous):
            layer._ilora_slow_active = original
