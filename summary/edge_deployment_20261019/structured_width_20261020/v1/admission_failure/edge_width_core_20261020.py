"""FFN channel removal with weight or QA-balanced fluctuation ranking.

Borrowed mechanisms: FLAP (AAAI 2024) WIFV and mean compensation. Uniform
half-width, multimodal scope and QA-balanced moments are pilot adaptations,
not the author's global adaptive structure search or a new algorithm claim.
"""
import copy
import hashlib
import json
import torch
from torch import nn


def sites(core):
    result = {}
    for name, module in core.named_modules():
        if all(hasattr(module, x) for x in ['gate_proj', 'up_proj', 'down_proj']):
            result[name] = ('text', module, module.down_proj)
        elif name.startswith('vision_tower.') and all(hasattr(module, x) for x in ['fc1', 'fc2']):
            result[name] = ('vision', module, module.fc2)
    return result


def parameter_digest(core):
    h = hashlib.sha256()
    for name, p in core.named_parameters():
        h.update(json.dumps([name, list(p.shape), str(p.dtype)]).encode())
        h.update(p.detach().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


class QAMoments:
    """Equal mass per QA; all selected positions within one QA have equal mass."""
    def __init__(self):
        self.mean_sum = None
        self.second_sum = None
        self.ids = []
        self.positions = []

    def add(self, x, qid):
        assert qid not in self.ids and x.ndim == 2 and len(x) > 0
        z = x.detach().double()
        assert torch.isfinite(z).all()
        mean, second = z.mean(0), z.square().mean(0)
        if self.mean_sum is None:
            self.mean_sum, self.second_sum = mean, second
        else:
            self.mean_sum += mean
            self.second_sum += second
        self.ids.append(qid)
        self.positions.append(len(x))

    def finish(self):
        assert self.ids
        mean = self.mean_sum / len(self.ids)
        variance = (self.second_sum / len(self.ids) - mean.square()).clamp_min(0)
        return {'mean': mean, 'variance': variance, 'ids': self.ids, 'positions': self.positions}


def subset_linear(old, rows=None, columns=None, compensation=None):
    w = old.weight.detach()
    bias = old.bias.detach() if old.bias is not None else None
    if rows is not None:
        w = w[rows]
        if bias is not None:
            bias = bias[rows]
    if columns is not None:
        w = w[:, columns]
    if compensation is not None:
        bias = compensation if bias is None else bias.double() + compensation
    new = nn.Linear(w.shape[1], w.shape[0], bias=bias is not None,
                    device=w.device, dtype=w.dtype)
    new.weight = nn.Parameter(w.contiguous().clone(), requires_grad=False)
    if bias is not None:
        new.bias = nn.Parameter(bias.to(w.dtype).contiguous().clone(), requires_grad=False)
    return new


def prune(core, moments, rule):
    assert rule in ['weight', 'fluctuation']
    found = sites(core)
    assert set(found) == set(moments)
    changes = []
    with torch.inference_mode():
        for name, (kind, module, output) in found.items():
            stat = moments[name]
            n = output.in_features
            assert n % 2 == 0 and stat['mean'].numel() == n and stat['variance'].numel() == n
            metric = output.weight.detach().double().square().sum(0)
            if rule == 'fluctuation':
                metric *= stat['variance']
            assert torch.isfinite(metric).all()
            keep = torch.argsort(metric, descending=True, stable=True)[:n // 2].sort().values
            removed = torch.ones(n, dtype=torch.bool)
            removed[keep] = False
            # Same mean-compensation rule for BOTH rankings: criterion is isolated.
            compensation = output.weight.detach().double()[:, removed] @ stat['mean'][removed]
            reduced = subset_linear(output, columns=keep, compensation=compensation)
            if kind == 'text':
                module.gate_proj = subset_linear(module.gate_proj, rows=keep)
                module.up_proj = subset_linear(module.up_proj, rows=keep)
                module.down_proj = reduced
                module.intermediate_size = n // 2
            else:
                module.fc1 = subset_linear(module.fc1, rows=keep)
                module.fc2 = reduced
            changes.append({'name': name, 'kind': kind, 'original_width': n,
                            'remaining_width': n // 2, 'kept': keep.tolist(),
                            'compensation_norm': float(compensation.norm()),
                            'calibration_positions_per_QA': stat['positions']})
    core.config.text_config.intermediate_size //= 2
    core.config.vision_config.intermediate_size //= 2
    core.config.llmcl_compensated_text_down_bias = True
    return changes


def controls(folder):
    """Small actual HF structure, serialization and known recovery controls."""
    from pathlib import Path
    from transformers import LlavaOnevisionConfig, Qwen2Config, SiglipVisionConfig
    from edge_width_runtime_20261020 import WidthOneVision
    torch.set_num_threads(2)
    torch.manual_seed(42)
    config = LlavaOnevisionConfig(
        text_config=Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, tie_word_embeddings=True),
        vision_config=SiglipVisionConfig(hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        image_size=28, patch_size=14),
        image_token_index=63, image_grid_pinpoints=[[28, 28]])
    model = WidthOneVision(config).eval()
    found = sites(model)
    assert sum(k == 'text' for k, _, _ in found.values()) == 2
    assert sum(k == 'vision' for k, _, _ in found.values()) == 2
    moments = {name: {'mean': torch.zeros(o.in_features, dtype=torch.float64),
                      'variance': torch.linspace(.01, 2, o.in_features, dtype=torch.float64),
                      'positions': [3, 9]} for name, (_, _, o) in found.items()}
    original = {k: v.detach().clone() for k, v in model.state_dict().items()}
    sizes = []
    for rule in ['weight', 'fluctuation']:
        candidate = WidthOneVision(copy.deepcopy(config)).eval()
        candidate.load_state_dict(original, strict=True)
        changes = prune(candidate, moments, rule)
        assert len(changes) == 4
        ids = torch.tensor([[1, 2, 3]])
        pixels = torch.randn(1, 3, 28, 28)
        with torch.inference_mode():
            a = candidate.language_model(ids).logits
            b = candidate.vision_tower(pixels).last_hidden_state
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        destination = Path(folder) / rule
        assert not destination.exists()
        candidate.save_pretrained(destination, safe_serialization=True)
        loaded = WidthOneVision.from_pretrained(destination, local_files_only=True).eval()
        assert parameter_digest(candidate) == parameter_digest(loaded)
        with torch.inference_mode():
            assert torch.equal(a, loaded.language_model(ids).logits)
            assert torch.equal(b, loaded.vision_tower(pixels).last_hidden_state)
        sizes.append(sum(p.numel() for p in candidate.parameters()))
    assert sizes[0] == sizes[1]
    # Removing a truly constant input column plus mean compensation is exact.
    old = nn.Linear(2, 2, bias=False).double()
    old.weight.data.copy_(torch.tensor([[2., 3.], [5., 7.]], dtype=torch.float64))
    x = torch.tensor([[1., 4.], [3., 4.]], dtype=torch.float64)
    kept = torch.tensor([0])
    compressed = subset_linear(old, columns=kept, compensation=old.weight[:, 1] * 4)
    assert torch.equal(old(x), compressed(x[:, kept]))
    equal_QA = QAMoments()
    equal_QA.add(torch.zeros(1, 2), 'a')
    equal_QA.add(torch.ones(9, 2) * 2, 'b')
    assert torch.equal(equal_QA.finish()['mean'], torch.ones(2, dtype=torch.float64))
    return {'actual_small_HF_conditions': 2, 'fresh_reload_state_and_text_vision_exact': True,
            'same_parameter_count': True, 'constant_channel_compensation_exact': True,
            'equal_QA_not_equal_token_weighting': True, 'CUDA_initialized': torch.cuda.is_initialized(),
            'not_full_model_or_quality_result': True}
