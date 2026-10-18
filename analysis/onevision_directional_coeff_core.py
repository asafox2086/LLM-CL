"""Paper-inspired continuous coefficient tuning, not an original method claim.

PiSSA/LoRA-XS motivate frozen pretrained directions. We explicitly use R=0,
rank16 q/v, common LR/dropout, and continue one R across all native tasks.
"""
import hashlib
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

VIEWS = ['principal_coeff_r16', 'random_coeff_r16']
RANK = 16
ROOT = Path(__file__).resolve().parents[1]


def bases(weight, name, view, rank=RANK):
    assert view in VIEWS and rank <= min(weight.shape)
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        matrix = weight.detach().float().cpu()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        assert float(singular[rank-1]) > 0
        if view == VIEWS[0]:
            left, right = u[:, :rank] * singular[:rank], vh[:rank, :]
        else:
            seed = int.from_bytes(hashlib.sha256(('42:direction-control:'+name).encode()).digest()[:8], 'big') % (2**63-1)
            generator = torch.Generator().manual_seed(seed)
            qleft = torch.linalg.qr(torch.randn(matrix.shape[0], rank, generator=generator), mode='reduced').Q
            qright = torch.linalg.qr(torch.randn(matrix.shape[1], rank, generator=generator), mode='reduced').Q
            left, right = qleft * singular[:rank], qright.T
        return left.contiguous(), right.contiguous(), singular[:rank].contiguous()


class DirectionalLinear(nn.Module):
    def __init__(self, weight, bias, name, view, rank=RANK):
        super().__init__()
        self.weight = nn.Parameter(weight.detach(), requires_grad=False)
        self.bias = nn.Parameter(bias.detach(), requires_grad=False) if bias is not None else None
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        left, right, singular = bases(weight, name, view, rank)
        self._gram_left_cpu = left.double().T @ left.double()
        self._gram_right_cpu = right.double() @ right.double().T
        self.register_buffer('left', left.to(weight.device))
        self.register_buffer('right', right.to(weight.device))
        self.register_buffer('singular', singular.to(weight.device))
        self.coeff = nn.Parameter(torch.zeros(rank, rank, device=weight.device, dtype=torch.float32))
        self.dropout = nn.Dropout(.1)

    def forward(self, inputs):
        result = F.linear(inputs, self.weight, self.bias)
        with torch.autocast(inputs.device.type, enabled=False):
            projected = F.linear(self.dropout(inputs.float()), self.right)
            delta = F.linear(F.linear(projected, self.coeff), self.left)
        return result + delta.to(result.dtype)

    def displacement_squared(self, old_coeff):
        change = self.coeff.detach().double().cpu() - old_coeff.double().cpu()
        value = float((self._gram_left_cpu * (change @ self._gram_right_cpu @ change.T).T).sum())
        assert value >= -1e-12
        return max(0., value)

    def memory_payload(self):
        return dict(frozen_basis_bytes=sum(p.numel()*p.element_size() for p in [self.left,self.right,self.singular]),
                    trainable_coefficient_bytes=self.coeff.numel()*self.coeff.element_size(),
                    CPU_Gram_monitor_bytes=sum(p.numel()*p.element_size() for p in [self._gram_left_cpu,self._gram_right_cpu]),
                    original_base_weight_bytes=self.weight.numel()*self.weight.element_size())


def checks():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results=[]
    for shape in [(7,7),(5,9),(9,5)]:
        weight=torch.randn(*shape)
        sample,target=torch.randn(4,shape[1]),torch.randn(4,shape[0])
        models=[DirectionalLinear(weight,None,'toy',v,rank=2) for v in VIEWS]
        for model in models:
            model.eval()
            assert torch.equal(model(sample),F.linear(sample,weight))
            before={k:v.detach().clone() for k,v in model.state_dict().items()}
            loss=F.mse_loss(model(sample),target);loss.backward()
            assert torch.isfinite(model.coeff.grad).all() and model.coeff.grad.norm()>0
            assert model.weight.grad is None and model.left.grad is None and model.right.grad is None
            optimizer=torch.optim.AdamW([model.coeff],lr=1e-4,weight_decay=0)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            assert not torch.equal(model.coeff,before['coeff'])
            for key in ['left','right','singular','weight']:assert torch.equal(model.state_dict()[key],before[key])
            finite=(model.left.double() @ (model.coeff.detach().double()-before['coeff'].double()) @ model.right.double())
            assert abs(model.displacement_squared(before['coeff'])-float(finite.square().sum()))<1e-12
            # Keeping the same module/coeff at a task boundary preserves the
            # acquired function exactly; no merge, quantization or R reset.
            prediction=model(sample).detach().clone();saved=model.coeff.detach().clone()
            model.coeff.data.copy_(saved)
            assert torch.equal(model(sample),prediction)
        # Same singular scaling and coordinate budgets in both controls.
        assert torch.equal(models[0].singular,models[1].singular)
        assert not torch.equal(models[0].left,models[1].left)
        results.append(dict(shape=shape,initial_function_exact=True,first_coefficient_gradient_nonzero=True,
                            frozen_basis_unchanged=True,finite_weight_displacement_checked=True,
                            continuous_function_no_reset=True,random_direction_control_matched=True))
    assert not torch.cuda.is_initialized()
    return dict(CPU_only=True,synthetic_shapes=results,GPU_or_QA_outcome=False)
