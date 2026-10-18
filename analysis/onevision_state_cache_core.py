"""Bounded functional state retention and input-only TRAIN-quality calling."""
import itertools
import math
import torch
import torch.nn.functional as F

PENALTY = .02
TEMPERATURE = .1
CAP = 3
VIEWS = ('seq_geometry', 'seq_quality', 'cache_geometry', 'cache_quality')


def retain(current, losses, cap=CAP):
    """losses maps only past/current candidate states to arrived TRAIN losses."""
    keys = sorted(losses)
    assert current in keys and all(1 <= k <= current for k in keys)
    n = len(losses[current])
    assert n and 1 <= cap <= CAP
    assert all(len(v) == n and all(math.isfinite(x) for x in v) for v in losses.values())
    options = []
    for count in range(1, min(cap, len(keys))+1):
        for subset in itertools.combinations(keys, count):
            if current not in subset:
                continue
            coverage = sum(min(losses[s][i] for s in subset) for i in range(n))/n
            options.append((coverage+PENALTY*(count-1), count,
                            tuple(-s for s in sorted(subset, reverse=True)), subset, coverage))
    best = min(options)
    return list(best[3]), {'TRAIN_coverage_mean_NLL': best[4], 'regularized_objective': best[0]}


def route(feature, keys, prototypes, memory_features, train_losses, quality):
    """No row, domain/task identity, answers, or TEST loss accepted by this API."""
    assert feature.device.type == 'cpu' and feature.ndim == 1
    assert keys == sorted(set(keys)) and 1 <= len(keys) <= CAP
    assert prototypes.shape == (len(keys), feature.numel())
    assert memory_features.ndim == 2 and memory_features.shape[1] == feature.numel()
    assert train_losses.shape == (len(keys), memory_features.shape[0])
    assert memory_features.shape[0] > 0
    assert all(torch.isfinite(t).all() for t in [feature, prototypes, memory_features, train_losses])
    f = F.normalize(feature.float(), dim=0)
    if quality:
        similarities = F.normalize(memory_features.float(), dim=1) @ f
        weights = F.softmax(similarities/TEMPERATURE, dim=0)
        expected = train_losses.float() @ weights
        index = int(expected.argmin())
        details = {'estimated_TRAIN_NLL': expected.tolist(),
                   'memory_similarity_max': float(similarities.max())}
    else:
        similarities = F.normalize(prototypes.float(), dim=1) @ f
        index = int(similarities.argmax())
        details = {'prototype_cosines': similarities.tolist()}
    return keys[index], details


def checks():
    checks = []
    chosen, _ = retain(2, {1: [0., 2.], 2: [2., 0.]})
    checks.append(chosen == [1, 2])
    chosen, _ = retain(2, {1: [1., 1.], 2: [1., 1.]})
    checks.append(chosen == [2])
    chosen, _ = retain(2, {1: [0., 0.], 2: [2., 2.]}, 1)
    checks.append(chosen == [2])
    try:
        retain(2, {2: [1.], 3: [0.]})
    except AssertionError:
        checks.append(True)
    else:
        checks.append(False)
    memory = torch.eye(2)
    losses = torch.tensor([[0., 5.], [5., 0.]])
    prototypes = torch.flip(memory, dims=[0])
    a, _ = route(memory[0], [1, 2], prototypes, memory, losses, True)
    b, _ = route(memory[1], [1, 2], prototypes, memory, losses, True)
    c, _ = route(memory[0], [1, 2], prototypes, memory, losses, False)
    checks.extend([a == 1, b == 2, c == 2])
    for quality in [False, True]:
        k, _ = route(memory[0], [2], prototypes[:1], memory, losses[1:], quality)
        checks.append(k == 2)
    assert all(checks)
    return len(checks)
