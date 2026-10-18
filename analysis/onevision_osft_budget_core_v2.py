"""Capacity-matched SVD controls using pinned OSFT kernels.

These are controls, not a novel method or an admitted GPU worker. Fixed rank8
q/v with frozen singular values has exactly the current LoRA trainable count.
No module growth, route, QA selection, environment mutation or model loading.
"""
import ast
import __future__
import hashlib
import json
import math
from pathlib import Path
import typing as t

import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
KERNEL_NAMES = {'create_svd_dict', 'reconstruct_weight_matrix',
                'project_parameter_to_orthogonal_space', 'project_gradient_to_orthogonal_space'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def author_kernels():
    folder = ROOT/'code/OSFT'
    rel = 'src/mini_trainer/osft_utils.py'
    path = folder/'source'/rel
    expected = next(x['sha256'] for x in json.loads((folder/'files_sha256.json').read_text()) if x['path'] == rel)
    assert sha(path) == expected
    definitions = [node for node in ast.parse(path.read_text()).body
                   if isinstance(node, ast.FunctionDef) and node.name in KERNEL_NAMES]
    assert len(definitions) == 4
    env = dict(torch=torch, nn=nn, dist=dist, math=math, t=t,
               OSFT_CACHE_V=False, SVDDecompositionDict=dict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=definitions, type_ignores=[])),
                 str(path), 'exec', flags=__future__.annotations.compiler_flag), env)
    return {name: env[name] for name in KERNEL_NAMES}, expected


def displacement_squared(old_u, old_v, new_u, new_v, singular):
    """Exact FP64 Gram norm of U1*S*V1-U0*S*V0, without dense W."""
    old_u, old_v, new_u, new_v, singular = [x.double() for x in
                                           [old_u, old_v, new_u, new_v, singular]]
    left = torch.cat([(new_u-old_u)*singular, new_u*singular], dim=1)
    right = torch.cat([old_v, new_v-old_v], dim=0)
    value = float(((left.T@left)*(right@right.T).T).sum())
    assert value >= -1e-12
    return max(value, 0.)


class BudgetSVDLinear(nn.Module):
    def __init__(self, weight, bias=None, *, rank=8, protected=True, kernels=None):
        super().__init__()
        assert weight.ndim == 2 and 0 < rank < min(weight.shape)
        assert torch.isfinite(weight).all()
        self.kernels = kernels if kernels is not None else author_kernels()[0]
        self.rank, self.protected, self.output_dtype = rank, protected, weight.dtype
        self.in_features, self.out_features = weight.shape[1], weight.shape[0]
        self.register_buffer('_compiled_eval_weight', None, persistent=False)
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter('bias', None)
        self.refactor(weight)

    def refactor(self, weight):
        """Use CURRENT weights, never reset to the untouched pretraining base."""
        assert weight.shape == (self.out_features, self.in_features)
        result = self.kernels['create_svd_dict'](weight.detach(), min(weight.shape)-self.rank,
                    upcast_dtype=torch.float32, output_dtype=torch.float32)
        for name in ['U_high', 'S_high', 'V_high']:
            if name in self._buffers:
                self._buffers[name] = result[name]
            else:
                self.register_buffer(name, result[name])
        self.U_low, self.V_low = result['U_low'], result['V_low']
        # Freeze S to match exactly540672 trainable factor coordinates. This is
        # intentionally a control variant, not the author's full OSFT recipe.
        self.S_low = nn.Parameter(result['S_low'].detach(), requires_grad=False)
        self._compiled_eval_weight = None
        assert sum(p.numel() for p in self.parameters() if p.requires_grad) == self.rank*(sum(weight.shape))

    def svd(self):
        state = {name: getattr(self, name) for name in ['U_high', 'S_high', 'V_high', 'U_low', 'S_low', 'V_low']}
        state['rank_high'] = self.U_high.shape[1]
        return state

    def current_weight(self, dtype=None):
        with torch.autocast(self.U_low.device.type, enabled=False):
            return self.kernels['reconstruct_weight_matrix'](self.svd(), torch.float32, dtype or self.output_dtype)

    def forward(self, inputs):
        if not self.training and not torch.is_grad_enabled():
            if self._compiled_eval_weight is None:
                self._compiled_eval_weight = self.current_weight().detach()
            weight = self._compiled_eval_weight
        else:
            weight = self.current_weight()
        return F.linear(inputs, weight, self.bias)

    def train(self, mode=True):
        self._compiled_eval_weight = None
        return super().train(mode)

    @torch.no_grad()
    def project_gradients(self):
        if self.protected:
            self.kernels['project_gradient_to_orthogonal_space'](self.svd())

    @torch.no_grad()
    def project_parameters(self):
        self._compiled_eval_weight = None
        if self.protected:
            self.kernels['project_parameter_to_orthogonal_space'](self.svd())

    def bytes(self):
        return dict(trainable=sum(p.numel()*p.element_size() for p in self.parameters() if p.requires_grad),
                    frozen_parameters=sum(p.numel()*p.element_size() for p in self.parameters() if not p.requires_grad),
                    frozen_buffers=sum(p.numel()*p.element_size() for p in self.buffers()))


def checks():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(6)
    torch.manual_seed(42)
    kernels, kernel_sha = author_kernels()
    results = []
    for shape in [(12, 12), (8, 14), (14, 8)]:
        weight = torch.randn(*shape)
        free = BudgetSVDLinear(weight, rank=2, protected=False, kernels=kernels)
        protected = BudgetSVDLinear(weight, rank=2, protected=True, kernels=kernels)
        protected.load_state_dict(free.state_dict())
        assert torch.equal(free.current_weight(), protected.current_weight())
        exact = free.current_weight(torch.float32).detach()
        with torch.autocast('cpu', dtype=torch.bfloat16):
            assert torch.equal(exact, free.current_weight(torch.float32).detach())
        assert torch.allclose(free.current_weight(), weight, rtol=2e-5, atol=2e-5)
        original = {k: v.clone() for k, v in free.state_dict().items()}
        x, target = torch.randn(5, shape[1]), torch.randn(5, shape[0])
        raw, after = [], []
        for model in [free, protected]:
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
            optimizer.zero_grad(set_to_none=True)
            F.mse_loss(model(x), target).backward()
            assert model.U_low.grad is not None and model.V_low.grad is not None and model.S_low.grad is None
            raw.append(torch.cat([model.U_low.grad.flatten(), model.V_low.grad.flatten()]).clone())
            model.project_gradients()
            after.append(torch.cat([model.U_low.grad.flatten(), model.V_low.grad.flatten()]).clone())
            optimizer.step()
            model.project_parameters()
            assert all(torch.equal(model.state_dict()[k], original[k]) for k in ['U_high', 'S_high', 'V_high', 'S_low'])
            assert not torch.equal(model.U_low, original['U_low']) or not torch.equal(model.V_low, original['V_low'])
        assert torch.equal(raw[0], raw[1]) and not torch.equal(after[0], after[1])
        actual_squared = displacement_squared(original['U_low'], original['V_low'],
            protected.U_low.detach(), protected.V_low.detach(), protected.S_low)
        dense = (protected.U_low.detach().double()*protected.S_low.double()@protected.V_low.detach().double()
                 - original['U_low'].double()*original['S_low'].double()@original['V_low'].double())
        assert math.isclose(actual_squared, float(dense.square().sum()), rel_tol=1e-8, abs_tol=1e-14)
        assert float((protected.U_high.T@protected.U_low).norm()) < 2e-5
        assert float((protected.V_low@protected.V_high.T).norm()) < 2e-5
        protected.eval()
        with torch.no_grad():
            prediction = protected(x)
            assert protected._compiled_eval_weight is not None
            assert torch.equal(prediction, protected(x))
        protected.train()
        assert protected._compiled_eval_weight is None
        # Re-factorization continues the resulting CURRENT function, not W0.
        changed = free.current_weight().detach().clone()
        free.refactor(changed)
        assert torch.allclose(free.current_weight(), changed, rtol=2e-5, atol=2e-5)
        results.append(dict(shape=shape, raw_gradients_exact_before_intervention=True,
            actual_projection_changes_gradient=True, frozen_high_and_S_exact=True,
            protected_subspace_residuals_bounded=True, current_function_refactor_passed=True))
    return dict(CPU_only=True, author_kernel_sha256=kernel_sha, synthetic_cases=results,
                GPU_started=False, model_or_QA_performance_evidence=False)


if __name__ == '__main__':
    print(json.dumps(checks()))
