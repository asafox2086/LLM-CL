"""Fixed q/v-r8 shared/uniform/response-soft adapters; no Torch on import.

All three modes retain the original 96 A/B Parameter objects and one dropout
draw. Conditional scores use the *post-dropout* A response, with full autograd.
"""
from pathlib import Path
import importlib.util
from types import MethodType

ARMS = ('shared', 'uniform', 'soft')
COORDINATES = 540672
PARAMETERS = 96
SLICES = ((0, 4), (4, 6), (6, 8))


def configure_gpu_math():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def rank_plan(layers=24):
    return {f'model.layers.{i}.self_attn.{part}_proj': 8
            for i in range(layers) for part in ('q', 'v')}


def _bank_forward(self, x):
    import torch
    from torch.nn import functional as F
    base = F.linear(x, self.weight, self.bias)
    z = self.A(self.dropout(x.float()))
    if self._conditional_mode == 'uniform':
        routed = torch.cat((z[..., :4], z[..., 4:6] * .5,
                            z[..., 6:8] * .5), dim=-1)
    else:
        assert self._conditional_mode == 'soft'
        energy = torch.stack((z[..., 4:6].float().square().mean(-1),
                              z[..., 6:8].float().square().mean(-1)), dim=-1)
        gates = torch.softmax(energy, dim=-1)  # FP32, fixed tau=1; no detach.
        routed = torch.cat((z[..., :4],
                            (z[..., 4:6].float() * gates[..., 0:1]).to(z.dtype),
                            (z[..., 6:8].float() * gates[..., 1:2]).to(z.dtype)), dim=-1)
    return base + (self.B(routed) * self.scale).to(base.dtype)


def bind_mode(model, arm, plan=None):
    """Bind only Python forward metadata; do not allocate/init any Parameter."""
    assert arm in ARMS
    plan = rank_plan() if plan is None else dict(plan)
    for name, rank in plan.items():
        layer = model.get_submodule(name)
        assert rank == 8 and layer.A.weight.shape[0] == layer.B.weight.shape[1] == 8
        assert layer.scale == 4 and layer.dropout.p == .1
        assert not hasattr(layer, '_conditional_mode')
        layer._conditional_mode = arm
        if arm != 'shared':
            layer.forward = MethodType(_bank_forward, layer)
    model._conditional_arm = arm
    model._conditional_deployed = False
    return metadata(model, plan)


def metadata(model, plan=None):
    arm = model._conditional_arm
    assert arm in ARMS
    deployed = bool(model._conditional_deployed)
    plan = rank_plan() if plan is None else dict(plan)
    if not deployed or arm == 'soft':
        for name in plan:
            layer = model.get_submodule(name)
            assert layer._conditional_mode == arm
            assert layer.A.weight.shape[0] == layer.B.weight.shape[1] == 8
    return dict(arm=arm, deployed=deployed, rank_plan=plan,
                split=[4, 2, 2], scale=4, dropout=.1,
                routing=('none' if arm == 'shared' else
                         'fixed_half' if arm == 'uniform' else 'post_dropout_A_energy'),
                soft_temperature=1, extra_router_parameters=0,
                training_coordinates=COORDINATES,
                explicit_AB=(not deployed or arm == 'soft'))


def make_method(model, asset, storage, arm, inherited_source=None):
    """Use the sealed plain-r8 initializer, then only bind the conditional rule."""
    import torch
    source = (Path(inherited_source) if inherited_source is not None else
              Path(asset) / 'exp/plain_r8_gpu_control_20261025/v1/control_adaptation.py')
    spec = importlib.util.spec_from_file_location('conditional_inherited_r8', source)
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    method = old.make_plain(model, asset, storage)
    parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
    before = storage.digest(parameters)
    identities = {n: id(p) for n, p in parameters.items()}
    cpu_rng = torch.get_rng_state().clone()
    gpu_rng = torch.cuda.get_rng_state(next(model.parameters()).device).clone()
    bind_mode(model, arm, method.rank_plan)
    assert before == storage.digest({n: p for n, p in model.named_parameters() if p.requires_grad})
    assert identities == {n: id(p) for n, p in model.named_parameters() if p.requires_grad}
    assert torch.equal(cpu_rng, torch.get_rng_state())
    assert torch.equal(gpu_rng, torch.cuda.get_rng_state(next(model.parameters()).device))
    assert len(parameters) == PARAMETERS and sum(p.numel() for p in parameters.values()) == COORDINATES
    method.conditional_metadata = metadata(model, method.rank_plan)
    return method


def install_meta(model, arm, *, deployed, plan=None):
    """Rebuild the state schema on meta before strict assign=True loading."""
    from methods import SeqLinear
    plan = rank_plan() if plan is None else dict(plan)
    assert arm in ARMS
    assert all(model.get_submodule(name).weight.is_meta for name in plan)
    if not deployed or arm == 'soft':
        for name, rank in plan.items():
            assert rank == 8
            base = model.get_submodule(name)
            layer = SeqLinear(base, rank=8, alpha=32)
            parent, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), child, layer)
        bind_mode(model, arm, plan)
    else:
        model._conditional_arm = arm
    model._conditional_deployed = bool(deployed)
    return metadata(model, plan)


def deploy(model, method):
    """Shared/uniform GPU-FP32 merge; soft keeps its complete explicit r8 bank."""
    import torch
    assert not torch.backends.cuda.matmul.allow_tf32
    arm = model._conditional_arm
    assert arm in ARMS and not model._conditional_deployed
    records = []
    model.eval()
    if arm != 'soft':
        for name in method.rank_plan:
            layer = model.get_submodule(name)
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                a, b = layer.A.weight.detach().float(), layer.B.weight.detach().float()
                if arm == 'uniform':
                    a = torch.cat((a[:4], .5 * a[4:6], .5 * a[6:8]), dim=0)
                weight = (layer.weight.detach().float() + 4 * (b @ a)).half()
                assert torch.isfinite(weight).all()
                with torch.device('meta'):
                    replacement = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0],
                        bias=layer.bias is not None, dtype=torch.float16)
                replacement.weight = torch.nn.Parameter(weight, requires_grad=False)
                if layer.bias is not None:
                    replacement.bias = layer.bias
            parent, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent), child, replacement)
            records.append(dict(module=name, scale=4, private_coefficient=.5 if arm == 'uniform' else 1,
                merge_dtype='torch.float32', output_dtype='torch.float16', device=str(weight.device),
                autocast_enabled=False, matmul_allow_tf32=False))
        method.layers.clear()
    model._conditional_deployed = True
    model.requires_grad_(False).eval()
    return dict(arm=arm, folded_modules=records, explicit_bank=arm == 'soft',
                metadata=metadata(model, method.rank_plan))
