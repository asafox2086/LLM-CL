"""Small TRAIN-quality router and smooth residual gate; no model expansion.

Inspired by D-MoLE input/task recognition and MVP relevance aggregation. This
is a new controlled adaptation, NOT a reproduction of either published method.
"""
import torch
import torch.nn.functional as F

STEPS = 400
LR = .03
DECAY = .001
VIEWS = ["original_hard", "quality_hard", "geometry_soft", "quality_soft"]


def fit(cosines, losses):
    assert cosines.ndim == losses.ndim == 2
    n, k = cosines.shape
    assert losses.shape == (n, k+1) and n > 0
    assert cosines.device.type == losses.device.type == "cpu"
    assert torch.isfinite(cosines).all() and torch.isfinite(losses).all()
    inputs = torch.cat([cosines.float(), torch.ones(n, 1)], dim=1)
    target = F.softmax(-losses.float(), dim=1).detach()  # base first, then fixed expert order
    w = torch.zeros(k+1, k+1)
    for i in range(k):
        w[i+1, i] = 10.
    initial = w.clone()
    w.requires_grad_(True)
    optimizer = torch.optim.AdamW([w], lr=LR, weight_decay=DECAY)
    events, first = [], None
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = -(target * F.log_softmax(inputs @ w.T, dim=1)).sum(1).mean()
        loss.backward()
        assert torch.isfinite(loss) and torch.isfinite(w.grad).all()
        if first is None:
            first = w.grad.detach().clone()
        last = w.grad.detach().clone()
        events.append({"step": step+1, "loss": float(loss.detach()), "grad_l2": float(w.grad.norm()), "lr": LR})
        optimizer.step()
    result = w.detach().clone()
    return result, {"parameters": result.numel(), "changed": not torch.equal(initial, result),
                    "events": events, "raw_first_gradient": first, "raw_last_gradient": last,
                    "initial": initial, "target_probabilities": target}


def probabilities(cosines, weights):
    x = torch.cat([cosines.float(), torch.ones(1)])
    p = F.softmax(weights.float() @ x, dim=0)
    assert torch.isfinite(p).all() and abs(float(p.sum())-1) < 1e-5
    return p


def choose(view, cosines, weights, threshold):
    assert view in VIEWS and cosines.ndim == 1
    best = int(cosines.argmax())
    similarity = float(cosines[best])
    if view == "original_hard":
        return (best if similarity >= threshold else None), 1., None
    if view == "geometry_soft":
        return best, max(0., min(1., (1.+similarity)/2.)), None
    p = probabilities(cosines, weights)
    if view == "quality_hard":
        winner = int(p.argmax())
        return (winner-1 if winner else None), 1., p.tolist()
    return int(p[1:].argmax()), 1.-float(p[0]), p.tolist()
