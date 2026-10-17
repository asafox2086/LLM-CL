"""Shared scheduler and actual-parameter checks, independent of method kernels."""
import hashlib
import math


def warmup_updates(updates):
    assert updates >= 1
    return min(math.floor(.03 * updates), updates - 1)


def parameter_digest(parameters):
    import torch
    h = hashlib.sha256()
    for index, parameter in enumerate(parameters):
        value = parameter.detach().cpu().contiguous()
        h.update(str((index, tuple(value.shape), str(value.dtype))).encode())
        raw = value.reshape(-1).view(torch.uint8).numpy().reshape(-1)
        # Chunking avoids a second full parameter-sized byte-string allocation.
        for begin in range(0, raw.size, 8 * 1024**2):
            h.update(raw[begin:begin + 8 * 1024**2].tobytes())
        del value, raw
    return h.hexdigest()


def validate_stage(updates, positive_updates, before, after):
    if not positive_updates:
        raise RuntimeError('No positive-lr optimizer update; stage cannot enter formal results')
    if before == after:
        raise RuntimeError('Trainable parameters unchanged: review numerical/author mechanism before admission')
    assert 0 < positive_updates <= updates
    return {'passed': True, 'updates': updates, 'positive_lr_updates': positive_updates,
            'parameters_before_sha256': before, 'parameters_after_sha256': after,
            'actual_parameters_changed': True}
