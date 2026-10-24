"""One fixed shared, sparse carrier. No data or model loading at import."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Basis(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.weight = nn.Parameter(value.detach().clone().float())


class Factor(nn.Module):
    def __init__(self, basis, value, expanding):
        super().__init__()
        self.basis = basis
        self.value = nn.Parameter(value.detach().clone().float())
        self.register_buffer('mask', torch.ones_like(self.value, dtype=torch.bool))
        self.expanding = bool(expanding)

    def forward(self, x):
        v = self.value * self.mask
        u = self.basis.weight
        if self.expanding:
            return F.linear(F.linear(x, u.T), v.T)
        return F.linear(F.linear(x, v), u)


class PackedFactor(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.basis = source.basis
        self.expanding = source.expanding
        self.shape = tuple(source.value.shape)
        mask = source.mask.detach().reshape(-1)
        assert mask.numel() % 8 == 0
        assert bool(torch.isfinite(source.value).all())
        assert bool((source.value.detach()[~source.mask] == 0).all())
        values = source.value.detach().reshape(-1)[mask].to(torch.float16).clone()
        assert bool(torch.isfinite(values).all())
        bits = (mask.reshape(-1, 8).to(torch.int16) * (2 ** torch.arange(8, device=mask.device))).sum(1).to(torch.uint8)
        self.register_buffer('values', values)
        self.register_buffer('bits', bits)

    def unpack(self):
        mask = ((self.bits[:, None] >> torch.arange(8, device=self.bits.device)) & 1).to(torch.bool).reshape(-1)
        assert int(mask.sum()) == self.values.numel()
        v = torch.zeros(math.prod(self.shape), dtype=self.values.dtype, device=self.values.device)
        v[mask] = self.values
        return v.reshape(self.shape)

    def forward(self, x):
        v = self.unpack()
        u = self.basis.weight
        if self.expanding:
            return F.linear(F.linear(x, u.T), v.T)
        return F.linear(F.linear(x, v), u)


def full_initialize(mlps, rank=1152):
    """All common short-axis SVD directions; zero extra U, nonzero V/32."""
    matrices = []
    for m in mlps:
        assert m.gate_proj.bias is None and m.up_proj.bias is None and m.down_proj.bias is None
        matrices.extend([m.gate_proj.weight.detach().float().cpu().T,
                         m.up_proj.weight.detach().float().cpu().T,
                         m.down_proj.weight.detach().float().cpu()])
    d, p = matrices[0].shape
    assert all(tuple(w.shape) == (d, p) for w in matrices)
    assert d < rank <= 2 * d
    w = torch.cat(matrices, dim=1)
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    assert u.shape == (d, d) and bool(torch.isfinite(s).all())
    v = s[:, None] * vh
    extra = rank - d
    basis = Basis(torch.cat([u, torch.zeros(d, extra)], dim=1))
    v = torch.cat([v, v[:extra] / 32], dim=0)
    factors = []
    for i, m in enumerate(mlps):
        for j, name in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            value = v[:, (3 * i + j) * p:(3 * i + j + 1) * p]
            layer = Factor(basis, value, name != 'down_proj')
            setattr(m, name, layer)
            factors.append(layer)
    return basis, factors


def prune(factor, sparse_fraction):
    """Monotone exact count; magnitude ties are resolved by flat index."""
    with torch.no_grad():
        flat = factor.value.reshape(-1)
        mask = factor.mask.reshape(-1)
        wanted = int(math.ceil(flat.numel() * (1 - sparse_fraction)))
        available = int(mask.sum())
        assert 0 < wanted <= available
        if wanted == available:
            return available
        scores = flat.abs().masked_fill(~mask, float('-inf'))
        cutoff = torch.topk(scores, wanted, sorted=False).values.min()
        new = (scores > cutoff)
        required = wanted - int(new.sum())
        ties = torch.nonzero((scores == cutoff) & mask, as_tuple=False).reshape(-1)
        assert 0 <= required <= ties.numel()
        new[ties[:required]] = True
        assert int(new.sum()) == wanted and not bool((new & ~mask).any())
        mask.copy_(new)
        flat.mul_(mask)
        return wanted


def enforce(factors, optimizer):
    with torch.no_grad():
        for f in factors:
            f.value.mul_(f.mask)
            if f.value.grad is not None:
                f.value.grad.mul_(f.mask)
            for key in ('exp_avg', 'exp_avg_sq'):
                if key in optimizer.state[f.value]:
                    optimizer.state[f.value][key].mul_(f.mask)


def recover_group(mlps, cached, device, emit, *, epochs=40, update_interval=50, final_at=960, audit=None):
    """One optimizer update jointly includes every group layer for one QA."""
    assert len(cached) > 0 and all(len(c) == len(mlps) for c in cached)
    basis = mlps[0].gate_proj.basis
    factors = [getattr(m, n) for m in mlps for n in ('gate_proj', 'up_proj', 'down_proj')]
    assert all(f.basis is basis for f in factors)
    for m in mlps:
        m.to(device)
    parameters = [basis.weight, *[f.value for f in factors]]
    assert len(parameters) == len({id(p) for p in parameters})
    optimizer = torch.optim.AdamW(parameters, lr=2e-5, betas=(.9, .999), eps=1e-8, weight_decay=.1)
    total = epochs * len(cached)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=1e-6)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda', init_scale=1)
    completed = [0]

    def actual_step(opt, args, kwargs):
        completed[0] += 1

    handle = optimizer.register_step_post_hook(actual_step)
    try:
        for step in range(1, total + 1):
            sample = cached[(step - 1) % len(cached)]
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for m, (xc, yc) in zip(mlps, sample):
                x, y = xc.to(device), yc.to(device)
                with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
                    predicted = m(x)
                denom = y.float().square().mean().clamp_min(1e-8)
                losses.append((predicted.float() - y.float()).square().mean() / denom)
            loss = torch.stack(losses).mean()
            assert bool(torch.isfinite(loss))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            assert all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters)
            grad_bytes = sum(p.grad.numel() * p.grad.element_size() for p in parameters)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1)
            assert bool(torch.isfinite(grad_norm))
            before = completed[0]
            lr = optimizer.param_groups[0]['lr']
            scaler.step(optimizer)
            scaler.update()
            assert completed[0] == before + 1
            event = None
            if step % update_interval == 0 or step == final_at:
                fraction = .75 * (1 - (1 - min(step / final_at, 1)) ** 3)
                kept = [prune(f, fraction) for f in factors]
                event = dict(fraction=fraction, kept=kept)
            enforce(factors, optimizer)
            if audit is not None:audit(factors,optimizer,step)
            assert all(bool(torch.isfinite(p).all()) for p in parameters)
            moment_bytes = sum(v.numel() * v.element_size() for st in optimizer.state.values()
                               for k, v in st.items() if k in ('exp_avg', 'exp_avg_sq'))
            emit(dict(step=step,actual_optimizer_steps=completed[0],loss=float(loss.detach()),lr=lr,
                      raw_grad_norm=float(grad_norm),gradient_bytes=grad_bytes,moment_bytes=moment_bytes,
                      optimizer_parameter_ids=[id(p) for p in parameters],trainable_coordinates=sum(p.numel() for p in parameters),
                      mask_event=event))
            scheduler.step()
        assert completed[0] == total
        assert all(int(f.mask.sum()) == f.mask.numel() // 4 for f in factors)
    finally:
        handle.remove()
        optimizer.zero_grad(set_to_none=True)
    return dict(updates=completed[0],trainable_coordinates=sum(p.numel() for p in parameters))


def pack_group(mlps):
    basis = mlps[0].gate_proj.basis
    basis.to(dtype=torch.float16)
    assert bool(torch.isfinite(basis.weight).all())
    basis.weight.requires_grad_(False)
    for m in mlps:
        for name in ('gate_proj', 'up_proj', 'down_proj'):
            source = getattr(m, name)
            assert source.basis is basis
            setattr(m, name, PackedFactor(source))
    return basis
