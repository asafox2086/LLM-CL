"""Common q/v path and fixed-support learning permission; no launch side effects."""
from pathlib import Path
import sys
import torch
from torch import nn
from core import Packed2of4Linear, set_ffn_trainable

ROOT = Path(__file__).resolve().parents[3]
PORT = ROOT / 'exp/two_backbone_native_cl_20261017_v2'


def add_common_qv(model, *, expected_layers=24):
    sys.path.insert(0, str(PORT))
    from methods import SeqLinear
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    candidates = [(name, layer) for name, layer in model.named_modules()
        if name.startswith('model.layers.') and name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj')
        and isinstance(layer, nn.Linear)]
    assert len(candidates) == 2 * expected_layers
    # This new paired experiment fixes its own shared q/v initialization.
    # It does not claim bitwise identity to an old anchor's RNG history.
    torch.manual_seed(42)
    for name, layer in candidates:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, SeqLinear(layer, rank=8, alpha=32))
    return [name for name, _ in candidates]


class SparseMethod:
    def __init__(self, model, case, *, expected_layers=24):
        assert case in ('Frozen', 'Plastic')
        self.model, self.case, self.name = model, case, 'SparseConnection' + case
        self.embedding = None
        self.host_optimizer = True
        self.offload_activations = True
        self.learning_rate = 1e-4
        self.expected_layers = expected_layers
        for parameter in model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        self.qv = []
        sys.path.insert(0, str(PORT))
        from methods import SeqLinear
        for name, layer in model.named_modules():
            if isinstance(layer, SeqLinear):
                assert name.startswith('model.layers.')
                assert name.rsplit('.', 1)[-1] in ('q_proj', 'v_proj')
                self.qv.extend((layer.A.weight, layer.B.weight))
        assert len(self.qv) == 4 * expected_layers
        for parameter in self.qv:
            parameter.requires_grad_(True)
        self.sparse_layers = set_ffn_trainable(model, case == 'Plastic', expected_layers=expected_layers)
        self.ffn = [layer.values for layer in self.sparse_layers] if case == 'Plastic' else []
        expected = {id(p) for p in self.qv + self.ffn}
        actual = {id(p) for p in model.parameters() if p.requires_grad}
        assert expected == actual

    def make_optimizer(self):
        sys.path.insert(0, str(PORT))
        from host_adam import HostAdamW
        groups = [dict(params=self.qv, lr=1e-4, name='common_qv')]
        if self.ffn:
            groups.append(dict(params=self.ffn, lr=1e-5, name='retained_ffn_values'))
        return HostAdamW(groups, lr=1e-4, betas=(.9, .999), eps=1e-8, weight_decay=0.)

    def set_instruction(self, task):
        pass

    def begin_stage(self, stage):
        self.stage = stage

    def method_state(self, stage):
        return dict(method=self.name, stage=stage, case=self.case,
            mask_fixed=True, qv_coordinates=sum(p.numel() for p in self.qv),
            extra_trainable_ffn_slots=sum(p.numel() for p in self.ffn))


def measured_optimizer(method, record):
    from types import MethodType
    optimizer = method.make_optimizer()
    names = {id(p): name for name, p in method.model.named_parameters() if p.requires_grad}
    expected = set(names)
    original = type(optimizer).step
    def step(self, *args, **kwargs):
        parameters = [p for group in self.param_groups for p in group['params']]
        assert {id(p) for p in parameters} == expected
        assert len(parameters) == len(expected) and all(p.grad is not None for p in parameters)
        gradient_bytes = sum(p.grad.numel() * p.grad.element_size() for p in parameters)
        lr = [group['lr'] for group in self.param_groups]
        value = original(self, *args, **kwargs)
        moments = [self.state[p][key] for p in parameters for key in ('exp_avg', 'exp_avg_sq')]
        assert all(t.device.type == 'cpu' for t in moments)
        record(dict(parameter_IDs_match=True,
            actual_parameters=[dict(id=id(p), name=names[id(p)], shape=list(p.shape), dtype=str(p.dtype)) for p in parameters],
            coordinates=sum(p.numel() for p in parameters), gradient_bytes=gradient_bytes,
            two_moment_bytes=sum(t.numel() * t.element_size() for t in moments),
            all_optimizer_tensor_state_bytes=sum(t.numel() * t.element_size()
                for parameter in parameters for t in self.state[parameter].values() if torch.is_tensor(t)),
            all_moments_CPU=True, actual_learning_rate=lr))
        return value
    optimizer.step = MethodType(step, optimizer)
    return optimizer
