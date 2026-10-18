"""Function-neutral initialization mechanics; separate diagnostic, not a method.

Keep one trainable A/B pair and one frozen reference pair. The reference path
shares the SAME dropped input and remains differentiable with respect to input.
No model loading, GPU orchestration, routing or TRAIN/TEST selection here.
"""
import copy
import torch
from torch import nn
import torch.nn.functional as F


def neutral_increment(dropped, a, b, reference_a, reference_b):
    # Do NOT detach/no_grad the reference output: upstream input derivatives
    # also need its subtractive path. Frozen reference parameters suffice.
    current = F.linear(F.linear(dropped, a), b)
    reference = F.linear(F.linear(dropped, reference_a), reference_b)
    return current-reference


class NeutralLoRA(nn.Module):
    def __init__(self, original, scale=4.):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.A, self.B, self.drop = original.A, original.B, original.drop
        self.enabled = original.enabled
        self.scale = scale
        # Parameters rather than buffers so ordinary model-parameter byte
        # accounting includes the reference cost; they are NOT optimized.
        self.reference_A = nn.Parameter(self.A.weight.detach().clone(), requires_grad=False)
        self.reference_B = nn.Parameter(self.B.weight.detach().clone(), requires_grad=False)

    def forward(self, x):
        y = F.linear(x, self.weight, self.bias)
        if not self.enabled:
            return y
        with torch.autocast(x.device.type, enabled=False):
            dropped = self.drop(x.float())
            delta = neutral_increment(dropped, self.A.weight, self.B.weight,
                                      self.reference_A, self.reference_B)
            delta = self.scale*delta
        return y+delta.to(y.dtype)


class ToyLoRA(nn.Module):
    """CPU test fixture only; never an experiment backbone."""
    def __init__(self, zero_b):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(6, 9), requires_grad=False)
        self.bias = None;self.enabled = True
        self.A = nn.Linear(9, 3, bias=False)
        self.B = nn.Linear(3, 6, bias=False)
        if zero_b:
            nn.init.zeros_(self.B.weight)
        self.drop = nn.Dropout(.1)

    def forward(self, x):
        return F.linear(x, self.weight)+4*self.B(self.A(self.drop(x)))


def checks():
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        model = NeutralLoRA(ToyLoRA(False));x = torch.randn(11, 9)
        base = F.linear(x, model.weight)
        for training in [False, True]:
            model.train(training)
            assert torch.equal(model(x), base), 'One shared dropout mask must cancel exactly'
        x = x.requires_grad_();model.zero_grad(set_to_none=True)
        model(x).square().mean().backward()
        assert model.A.weight.grad.norm() > 0 and model.B.weight.grad.norm() > 0
        assert model.reference_A.grad is None and model.reference_B.grad is None
        assert not model.reference_A.requires_grad and not model.reference_B.requires_grad
        # Directly test correct vs incorrectly detached input-gradient paths.
        z = torch.randn(7, 9, requires_grad=True);probe = torch.randn(7, 6)
        d = neutral_increment(z, model.A.weight, model.B.weight, model.reference_A, model.reference_B)
        assert torch.count_nonzero(d) == 0
        correct = torch.autograd.grad((d*probe).sum(), z)[0]
        assert float(correct.abs().max()) < 1e-6
        wrong = F.linear(F.linear(z, model.A.weight), model.B.weight)-F.linear(
            F.linear(z, model.reference_A), model.reference_B).detach()
        wrong_grad = torch.autograd.grad((wrong*probe).sum(), z)[0]
        assert wrong_grad.norm() > 1e-4, 'Detached reference negative control must be detected'
        # Independent dropout masks must NOT pass the equality control.
        left = F.linear(F.linear(F.dropout(z, .1, True), model.A.weight), model.B.weight)
        right = F.linear(F.linear(F.dropout(z, .1, True), model.reference_A), model.reference_B)
        assert torch.count_nonzero(left-right) > 0
        # B0=0 reference path is identically zero: verify BOTH actual Adam steps
        # match ordinary cold LoRA, with shared seed and identical dropout.
        cold = ToyLoRA(True);neutral = NeutralLoRA(copy.deepcopy(cold))
        states = []
        for branch in [cold, neutral]:
            opt = torch.optim.AdamW([p for p in branch.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
            torch.manual_seed(901)
            for step in range(2):
                opt.zero_grad(set_to_none=True)
                branch(x.detach()).square().mean().backward()
                torch.nn.utils.clip_grad_norm_([p for p in branch.parameters() if p.requires_grad],1.)
                opt.step()
                for group in opt.param_groups:group['lr'] = 5e-5
            states.append((branch.A.weight.detach().clone(),branch.B.weight.detach().clone()))
        assert all(torch.equal(a,b) for a,b in zip(*states))
        assert torch.equal(neutral.reference_B, torch.zeros_like(neutral.reference_B))
        assert sum(p.numel() for p in neutral.parameters() if p.requires_grad) == 45
        assert neutral.reference_A.numel()+neutral.reference_B.numel() == 45
    return dict(CPU_only=True,checks=8,
        cases=['eval exact initial function','TRAIN shared-dropout exact initial function',
               'nonzero live A/B gradients and frozen reference',
               'correct input-gradient cancellation at fixed FP32 tolerance1e-6',
               'detached-reference negative control','independent-mask negative control',
               'two cold Adam updates exactly equal ordinary LoRA','reference parameter storage counted'],
        model_performance_evidence=False,GPU_started=False)


if __name__ == '__main__':
    import json
    print(json.dumps(checks()))
