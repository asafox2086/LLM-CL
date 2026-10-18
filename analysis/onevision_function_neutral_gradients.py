"""Pure gradient profile helper; no device/environment side effects."""
import math
import torch


def gradient_values(obj):
    vector = obj['preclip_gradient'].double();offset = 0
    sums = {'A': 0., 'B': 0.};nonzero = {'A': 0, 'B': 0};counts = {'A': 0, 'B': 0}
    assert torch.isfinite(vector).all()
    for block in obj['layout']:
        name, size = block['name'], block['count']
        part = vector[offset:offset+size];offset += size
        assert part.numel() == size == math.prod(block['shape'])
        factor = 'A' if name.endswith('.A.weight') else 'B' if name.endswith('.B.weight') else None
        assert factor is not None, name
        sums[factor] += float(part.square().sum());counts[factor] += 1
        nonzero[factor] += int(bool(torch.count_nonzero(part)))
    assert offset == vector.numel() and counts == {'A': 48, 'B': 48}
    return dict(TRAIN_ids=obj['train_ids'],
        preclip_A_gradient_l2=math.sqrt(sums['A']),preclip_B_gradient_l2=math.sqrt(sums['B']),
        nonzero_gradient_layers=nonzero,
        zero_vector_does_not_distinguish_missing_gradient_from_exact_zero=True)
