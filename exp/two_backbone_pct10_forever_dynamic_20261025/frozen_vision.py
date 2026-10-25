"""Identical bounded CPU feature cache for both frozen vision backbones."""
from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
LIMIT = 128 * 1024**2


def install(model):
    assert not any(p.requires_grad for p in model.visual.parameters())
    model._llmcl_visual_cache = OrderedDict()
    model._llmcl_visual_cache_bytes = 0
    model._llmcl_visual_cache_hits = 0
    model._llmcl_visual_cache_misses = 0
    original = model.visual.forward
    model._llmcl_visual_original = original

    def forward(hidden_states, grid_thw=None):
        if hidden_states.ndim == 2 and hidden_states.shape[-1] == model.config.hidden_size:
            return hidden_states
        return original(hidden_states, grid_thw)

    model.visual.forward = forward
    return original


def stats(model):
    return {'visual_cache_cpu_bytes': model._llmcl_visual_cache_bytes,
            'visual_cache_cpu_limit': LIMIT, 'visual_cache_hits': model._llmcl_visual_cache_hits,
            'visual_cache_misses': model._llmcl_visual_cache_misses}


def prepare(batch, row, model, original=None):
    if 'pixel_values' not in batch:
        return False
    pixels = batch['pixel_values']
    if pixels.ndim == 2 and pixels.shape[-1] == model.config.hidden_size:
        return False
    assert not any(p.requires_grad for p in model.visual.parameters())
    images = []
    for rel in row['images']:
        p = ROOT / rel
        st = p.stat()
        images.append((rel, st.st_size, st.st_mtime_ns))
    dtype = model.visual.get_dtype()
    key = hashlib.sha256(json.dumps({'images': images, 'native_layout': batch['image_grid_thw'].tolist(),
                                    'dtype': str(dtype)}).encode()).hexdigest()
    cache = model._llmcl_visual_cache
    if key in cache:
        features = cache[key]
        cache.move_to_end(key)
        model._llmcl_visual_cache_hits += 1
        hit = True
    else:
        extractor = original or model._llmcl_visual_original
        with torch.no_grad(), torch.autocast(pixels.device.type, dtype=dtype):
            features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach().cpu()
        size = features.numel() * features.element_size()
        assert features.ndim == 2 and features.shape[-1] == model.config.hidden_size
        assert torch.isfinite(features).all()
        if size <= LIMIT:
            while model._llmcl_visual_cache_bytes + size > LIMIT:
                _, dropped = cache.popitem(last=False)
                model._llmcl_visual_cache_bytes -= dropped.numel() * dropped.element_size()
            cache[key] = features
            model._llmcl_visual_cache_bytes += size
        model._llmcl_visual_cache_misses += 1
        hit = False
    batch['pixel_values'] = features.to(model.device, dtype=dtype)
    if getattr(model, 'llmcl_backbone', None) != 'onevision':
        assert len(features) == int((batch['input_ids'] == model.config.image_token_id).sum())
    assert model._llmcl_visual_cache_bytes <= LIMIT
    return hit
