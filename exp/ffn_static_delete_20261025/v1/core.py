"""Physical FFN deletion and temporary differentiable scalar importance gates."""
import hashlib
import json

import torch
from torch import nn


class ZeroFFN(nn.Module):
    """Keep the decoder's residual path, with no retained FFN parameters."""
    def forward(self, x):
        return torch.zeros_like(x)


class GatedFFN(nn.Module):
    """Freeze the original FFN values; only the FP32 scalar gate is trainable.

    Install after the model's final dtype/device conversion so gate stays FP32.
    """
    def __init__(self, original):
        super().__init__()
        self.original = original
        self.original.requires_grad_(False)
        first = next(original.parameters(), None)
        self.gate = nn.Parameter(torch.ones((), dtype=torch.float32,
                                           device=first.device if first is not None else None))

    def forward(self, x):
        output = self.original(x)
        # Keep the scalar's derivative out of FP16 reduction/rounding. At gate=1
        # the FP16 -> FP32 -> FP16 conversion preserves every output value.
        return (output.float() * self.gate).to(output.dtype)


def _core(model):
    return getattr(model, 'core', model)


def _measure(items):
    seen = set()
    coefficients = size = 0
    for _, tensor in items:
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes())
        if key in seen:
            continue
        seen.add(key)
        # Count allocated storage once, including tied embedding/head weights.
        assert storage.nbytes() % tensor.element_size() == 0
        coefficients += storage.nbytes() // tensor.element_size()
        size += storage.nbytes()
    return {'coefficients': coefficients, 'bytes': size}


def layout(model):
    core = _core(model)
    return {'weights': _measure(core.named_parameters(remove_duplicate=False)),
            'ordinary_buffers': _measure(core.named_buffers(remove_duplicate=False)),
            'trainable': _measure((n, p) for n, p in core.named_parameters(remove_duplicate=False)
                                  if p.requires_grad)['coefficients']}


def delete(model, indexes):
    """Physically remove selected FFNs; return actual whole-model storage savings.

    No original module is retained. Duplicate/out-of-range selections are rejected
    before any mutation. This also handles shared/tied tensors conservatively.
    """
    selected = list(indexes)
    layers = model.model.layers
    assert len(selected) == len(set(selected)), 'duplicate layer index'
    assert all(isinstance(i, int) and 0 <= i < len(layers) for i in selected)
    assert all(not isinstance(layers[i].mlp, ZeroFFN) for i in selected)
    before = layout(model)['weights']
    for i in selected:
        layers[i].mlp = ZeroFFN()
    after = layout(model)['weights']
    return {'indexes': selected,
            'coefficients': before['coefficients'] - after['coefficients'],
            'bytes': before['bytes'] - after['bytes']}


def digest(items):
    """Hash every named tensor, including persistent buffers and tied aliases."""
    if isinstance(items, nn.Module):
        items = _core(items).state_dict()
    if hasattr(items, 'items'):
        items = items.items()
    h = hashlib.sha256()
    for name, value in sorted(items):
        value = value.detach().cpu().contiguous()
        metadata = json.dumps([name, str(value.dtype), list(value.shape)],
                              ensure_ascii=False, separators=(',', ':')).encode()
        h.update(len(metadata).to_bytes(8, 'little'))
        h.update(metadata)
        h.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def state_cpu(model):
    """Detached CPU snapshot preserving exact aliases such as tied embeddings."""
    out, seen = {}, {}
    for name, tensor in _core(model).state_dict().items():
        storage = tensor.untyped_storage()
        key = (str(tensor.device), storage.data_ptr(), storage.nbytes(),
               tensor.storage_offset(), tuple(tensor.shape), tuple(tensor.stride()), tensor.dtype)
        if key not in seen:
            seen[key] = tensor.detach().cpu().clone()
        out[name] = seen[key]
    return out
