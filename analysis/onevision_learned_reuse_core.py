"""TRAIN-only cosine metric and bounded expert allocation; experimental, not optimal.

No domain/task labels or answer text enter this module. Expert labels are past
allocation decisions, NOT externally verified evidence that a task is mastered.
"""
import hashlib
import math
import torch
import torch.nn.functional as F

DIM = 32
MEMORY = 16  # Frozen input vectors per expert, not language-model replay.
STEPS = 40
LR = .01
TEMPERATURE = .1
NEW_COST = .1  # TRAIN calibration: false-novel weight relative to false-reuse.


def initialize(input_dim):
    generator = torch.Generator(device="cpu").manual_seed(42)
    return torch.randn(DIM, input_dim, generator=generator) / math.sqrt(input_dim)


def embed(features, weight):
    assert features.device.type == weight.device.type == "cpu"
    z = F.normalize(F.linear(features.float(), weight.float()), dim=-1)
    assert torch.isfinite(z).all()
    return z


def grouped(memory):
    assert memory
    keys = sorted({r["expert"] for r in memory})
    vectors = torch.stack([r["feature"] for r in memory]).float()
    labels = torch.tensor([keys.index(r["expert"]) for r in memory])
    assert len({r["id"] for r in memory}) == len(memory)
    return keys, vectors, labels


def prototypes(memory, weight):
    keys, vectors, labels = grouped(memory)
    z = embed(vectors, weight)
    p = torch.stack([F.normalize(z[labels == i].mean(0), dim=0) for i in range(len(keys))])
    return keys, p


def fit(memory, weight, learned):
    keys, vectors, labels = grouped(memory)
    old = weight.detach().clone()
    if not learned or len(keys) < 2:
        return old, {"optimizer_updates": 0, "parameter_changed": False,
                     "reason": "fixed_control" if not learned else "only_one_expert_no_discriminative_labels",
                     "events": [], "memory_vectors": len(memory)}
    w = old.clone().requires_grad_(True)
    optimizer = torch.optim.AdamW([w], lr=LR, weight_decay=0)
    events = []
    first_gradient = None
    for step in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        z = embed(vectors, w)
        p = torch.stack([F.normalize(z[labels == i].mean(0), dim=0) for i in range(len(keys))])
        # Leave the query out of its own positive prototype wherever possible.
        logits = z @ p.T
        for i in range(len(keys)):
            ix = (labels == i).nonzero().flatten()
            if len(ix) > 1:
                leave = F.normalize(z[ix].sum(0)[None] - z[ix], dim=1)
                logits[ix, i] = (z[ix] * leave).sum(1)
        loss = F.cross_entropy(logits / TEMPERATURE, labels) + .001 * (w-old).square().mean()
        loss.backward()
        assert w.grad is not None and torch.isfinite(w.grad).all() and torch.isfinite(loss)
        if first_gradient is None:
            first_gradient = w.grad.detach().clone()
        last_gradient = w.grad.detach().clone()
        events.append({"step": step+1, "loss": float(loss.detach()),
                       "gradient_l2": float(w.grad.norm()), "lr": LR})
        optimizer.step()
    result = w.detach().clone()
    return result, {"optimizer_updates": STEPS, "parameter_changed": not torch.equal(result, old),
                    "delta_l2": float((result-old).norm()), "memory_vectors": len(memory), "events": events,
                    "raw_gradient_first": first_gradient, "raw_gradient_last": last_gradient}


def calibrate(memory, weight):
    """Known TRAIN leave-one-out similarities vs other-expert similarities.

    This calibrates an input-cluster novelty heuristic, NOT unseen-task accuracy.
    Lower thresholds win ties, expressing a declared preference for reuse.
    """
    keys, vectors, labels = grouped(memory)
    z = embed(vectors, weight)
    _, p = prototypes(memory, weight)
    positive, negative = [], []
    for n in range(len(memory)):
        same = (labels == labels[n]).nonzero().flatten()
        other = same[same != n]
        if len(other):
            positive.append(float(z[n] @ F.normalize(z[other].mean(0), dim=0)))
        if len(keys) > 1:
            negative.append(float((z[n] @ p.T)[torch.arange(len(keys)) != labels[n]].max()))
    if not positive:
        threshold, reason = .5, "singleton_default"
    elif not negative:
        threshold = float(torch.quantile(torch.tensor(positive), .1))
        reason = "single_expert_TRAIN_positive_10th_percentile"
    else:
        candidates = sorted(set([-1., 1.] + positive + negative))
        def cost(t):
            return NEW_COST * sum(v < t for v in positive)/len(positive) + sum(v >= t for v in negative)/len(negative)
        threshold = min(candidates, key=lambda t: (cost(t), t))
        reason = "TRAIN_false_novel_cost_0.1_false_reuse_cost_1"
    return max(-1., min(1., threshold)), {"reason": reason, "positive_cosines": positive,
                                        "negative_cosines": negative, "new_cost": NEW_COST}


def similarities(features, memory, weight):
    keys, p = prototypes(memory, weight)
    return keys, embed(features, weight) @ p.T


def allocate(features, memory, weight, threshold, size, cap):
    assert features.ndim == 2 and 1 <= cap and 0 <= size <= cap
    if not size:
        assert not memory
        return "e0", {"action": "new_from_base", "novel_fraction": 1., "threshold": threshold}
    keys, sim = similarities(features, memory, weight)
    assert len(keys) == size
    best, which = sim.max(1)
    novel = float((best < threshold).float().mean())
    chosen = min(range(len(keys)), key=lambda i: (-int((which == i).sum()), -float(sim[:, i].mean()), keys[i]))
    new = novel > .5 and size < cap
    return ("e"+str(size) if new else keys[chosen]), {
        "action": "new_from_base" if new else "reuse", "threshold": threshold,
        "novel_fraction": novel, "cap_forced_reuse": novel > .5 and size == cap,
        "expert_keys": keys, "cosines": sim.tolist(), "row_winners": [keys[i] for i in which.tolist()]}


def infer(feature, memory, weight, threshold):
    keys, sim = similarities(feature[None], memory, weight)
    value, index = sim[0].max(0)
    return (keys[int(index)] if float(value) >= threshold else None), float(value)


def update_memory(memory, rows, features, expert):
    assert len(rows) == len(features)
    records = {r["id"]: r for r in memory}
    for row, vector in zip(rows, features):
        assert row["id"] not in records, "Native TRAIN QA unexpectedly repeated"
        records[row["id"]] = {"id": row["id"], "expert": expert, "feature": vector.detach().float().cpu().clone()}
    keys = sorted({r["expert"] for r in records.values()})
    return [r for key in keys for r in sorted((r for r in records.values() if r["expert"] == key),
            key=lambda r: hashlib.sha256(("42:learned-reuse-memory:"+r["id"]).encode()).digest())[:MEMORY]]
