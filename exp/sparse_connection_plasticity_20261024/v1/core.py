"""Physical 2:4 FFN storage. Independent WIP; this file does not launch a model.

SparseGPT chooses and compensates the weights before this conversion. This
module stores two slots plus one pattern byte per four original coefficients.
It makes no claim about function preservation, full rank, speed or peak RAM.
"""
import torch
from torch import nn
import torch.nn.functional as F


PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
TARGETS = frozenset(('gate_proj', 'up_proj', 'down_proj'))


def pack(weight, keep_mask):
    """Pack the author's explicit support; zero slots still count as weights."""
    assert weight.ndim == 2 and weight.shape[1] % 4 == 0
    assert bool(torch.isfinite(weight).all())
    groups = weight.detach().reshape(weight.shape[0], -1, 4)
    assert keep_mask.shape == weight.shape and keep_mask.dtype == torch.bool
    support = keep_mask.to(weight.device).reshape_as(groups)
    assert bool((support.sum(-1) == 2).all()), 'Expected two retained slots'
    assert bool((groups.masked_select(~support) == 0).all())
    # Distinct integer priority, so value ties/zero values cannot alter support.
    priority = support.to(torch.int64) * 4 + torch.arange(3, -1, -1, device=weight.device)
    indices = priority.topk(2, dim=-1).indices.sort(dim=-1).values
    encode = torch.full((16,), -1, dtype=torch.int64, device=weight.device)
    for code, (left, right) in enumerate(PAIRS):
        encode[4 * left + right] = code
    pattern = encode[4 * indices[..., 0] + indices[..., 1]]
    assert bool((pattern >= 0).all())
    values = groups.gather(-1, indices).clone()
    return values, pattern.to(torch.uint8)


class Packed2of4Linear(nn.Module):
    def __init__(self, base, keep_mask, *, master_dtype=torch.float32):
        super().__init__()
        assert isinstance(base, nn.Linear)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.output_dtype = base.weight.dtype
        values, pattern = pack(base.weight, keep_mask)
        self.values = nn.Parameter(values.to(master_dtype), requires_grad=False)
        self.register_buffer('pattern', pattern)
        self.register_buffer('pattern_offsets', torch.tensor(
            PAIRS, dtype=torch.int64, device=base.weight.device))
        self.bias = base.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

    def materialize(self):
        """Transient dense allocation; never stored as a model field/buffer."""
        indices = self.pattern_offsets[self.pattern.long()]
        groups = self.values.new_zeros((self.out_features, self.in_features // 4, 4))
        return groups.scatter(-1, indices, self.values).reshape(
            self.out_features, self.in_features).to(self.output_dtype)

    def forward(self, x):
        return F.linear(x, self.materialize(), self.bias)

    def metadata(self):
        return dict(shape=[self.out_features, self.in_features],
            stored_weight_slots=self.values.numel(),
            values_dtype=str(self.values.dtype),
            values_bytes=self.values.numel() * self.values.element_size(),
            pattern_bytes=self.pattern.numel() * self.pattern.element_size(),
            lookup_bytes=self.pattern_offsets.numel() * self.pattern_offsets.element_size(),
            temporary_dense_is_not_a_persistent_weight=True)


def replace_language_ffn(model, keep_masks, *, expected_layers=24):
    candidates = [(name, module) for name, module in model.named_modules()
        if name.startswith('model.layers.') and name.rsplit('.', 1)[-1] in TARGETS
        and isinstance(module, nn.Linear)]
    assert len(candidates) == 3 * expected_layers
    assert set(keep_masks) == {name for name, _ in candidates}
    for name, base in candidates:
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, Packed2of4Linear(base, keep_masks[name]))
    return [name for name, _ in candidates]


def set_ffn_trainable(model, enabled, *, expected_layers=24):
    layers = [layer for layer in model.modules() if isinstance(layer, Packed2of4Linear)]
    assert len(layers) == 3 * expected_layers
    for layer in layers:
        assert layer.values.dtype == torch.float32
        layer.values.requires_grad_(enabled)
        layer.values.grad = None
    return layers


def finalize_ffn_values(model, *, expected_layers=24):
    """Evaluate only after this prescribed final cast and complete reload."""
    layers = [layer for layer in model.modules() if isinstance(layer, Packed2of4Linear)]
    assert len(layers) == 3 * expected_layers
    for layer in layers:
        layer.values.grad = None
        layer.values.requires_grad_(False)
        layer.values.data = layer.values.data.to(torch.float16)
        assert layer.output_dtype == torch.float16
    return layers
