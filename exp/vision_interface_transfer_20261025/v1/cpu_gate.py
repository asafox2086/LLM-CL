"""One synthetic CPU admission: native vision, spatial pack, storage and QA weighting.

No pretrained weights, images, dataset rows, CUDA or optimization are used.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import copy
import gc
import io
import json
import sys
import time
import weakref
from pathlib import Path

import torch
from transformers import (LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration,
                          Qwen2Config, SiglipVisionConfig)

from vision_interface import Moments, VisionInterface, install, student


SMALL = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
             num_attention_heads=2, num_channels=3, image_size=12, patch_size=2,
             attention_dropout=0., layer_norm_eps=1e-6)


def storages(module):
    found = {}
    for _, parameter in module.named_parameters(remove_duplicate=False):
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr(), storage.nbytes())
        found[key] = storage.nbytes()
    return found


def parameter_count(module):
    seen = set()
    count = 0
    for _, parameter in module.named_parameters(remove_duplicate=False):
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr(), storage.nbytes())
        if key not in seen:
            seen.add(key)
            count += storage.nbytes() // parameter.element_size()
    return count


def serialized_state(module):
    buffer = io.BytesIO()
    torch.save(module.state_dict(), buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location='cpu', weights_only=True)


def moments_check():
    # Unequal lengths and group-specific shifts make pooled-token fitting wrong.
    generator = torch.Generator().manual_seed(812)
    samples = []
    for index, length in enumerate((2, 7, 31)):
        x = torch.randn(length, 3, generator=generator, dtype=torch.float64) + index
        y = torch.randn(length, 2, generator=generator, dtype=torch.float64)
        y = y + x @ torch.tensor([[1., -2.], [.5, 1.], [-1., .25]], dtype=torch.float64)
        y = y + torch.tensor([index * 3., -index], dtype=torch.float64)
        samples.append((x, y))
    moments = Moments(3, 2)
    for x, y in samples:
        moments.add(x, y)
    weight, bias, ridge = moments.solve()
    x = torch.cat([x for x, _ in samples])
    y = torch.cat([y for _, y in samples])
    row_weights = torch.cat([torch.full((len(a),), 1. / (len(samples) * len(a)),
                                         dtype=torch.float64) for a, _ in samples])
    mean_x = (row_weights[:, None] * x).sum(0)
    centered = x - mean_x
    explicit_covariance = centered.T @ (row_weights[:, None] * centered)
    expected_ridge = 1e-3 * explicit_covariance.trace() / x.shape[1]
    assert torch.allclose(torch.tensor(ridge, dtype=torch.float64), expected_ridge,
                          rtol=1e-11, atol=1e-13)
    # Solve the augmented weighted least-squares problem independently, with an
    # unpenalized intercept, rather than reimplementing the centered solve.
    design = torch.cat((x, torch.ones(len(x), 1, dtype=torch.float64)), dim=1)
    penalty = torch.zeros(3, 4, dtype=torch.float64)
    penalty[:, :3] = torch.eye(3, dtype=torch.float64) * expected_ridge.sqrt()
    augmented = torch.cat((design * row_weights.sqrt()[:, None], penalty))
    targets = torch.cat((y * row_weights.sqrt()[:, None], torch.zeros(3, 2, dtype=torch.float64)))
    reference = torch.linalg.lstsq(augmented, targets).solution
    actual = torch.cat((weight.T, bias[None, :]))
    assert torch.allclose(actual, reference, rtol=1e-10, atol=1e-10)
    pooled = torch.linalg.lstsq(torch.cat((design / len(x) ** .5, penalty)),
                                torch.cat((y / len(x) ** .5, torch.zeros(3, 2, dtype=torch.float64)))).solution
    assert (pooled - reference).abs().max() > 1e-3, 'Fixture failed to distinguish QA and token weights'
    # Duplicating positions inside one QA must preserve its total weight of one.
    repeated = Moments(3, 2)
    for index, (a, b) in enumerate(samples):
        repeated.add(a.repeat_interleave(3, dim=0) if index == 0 else a,
                     b.repeat_interleave(3, dim=0) if index == 0 else b)
    w2, b2, r2 = repeated.solve()
    assert torch.allclose(weight, w2, rtol=1e-10, atol=1e-10)
    assert torch.allclose(bias, b2, rtol=1e-10, atol=1e-10)
    assert abs(ridge - r2) < 1e-12


def bridge_class():
    root_hint = os.environ.get('LLMCL_ROOT')
    root = Path(root_hint) if root_hint else Path(__file__).resolve().parents[3]
    candidates = [root / '.deployment/paired_image_support_20261024_view/exp/two_backbone_native_cl_20261017_v2',
                  root / 'exp/two_backbone_native_cl_20261017_v2']
    port = next(path for path in candidates if (path / 'onevision_bridge.py').is_file())
    sys.path.insert(0, str(port))
    from onevision_bridge import OneVisionBridge
    return OneVisionBridge


def main():
    tick = time.monotonic()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    moments_check()

    # Exercise the actual installed Idefics3 class, including FP16 LayerNorm,
    # attention and both interpolation operations, without CPU autocast.
    standalone = VisionInterface(student(SMALL), output_width=16, native_size=12, output_grid=4).half().eval()
    assert standalone.student.config._attn_implementation == 'eager'
    pixels = torch.randn(3, 3, 8, 8, dtype=torch.float16)
    with torch.inference_mode():
        expected = standalone(pixels).hidden_states[-1]
    assert expected.shape == (3, 16, 16) and expected.dtype == torch.float16
    assert torch.isfinite(expected).all()
    independent = VisionInterface(student(SMALL), output_width=16, native_size=12, output_grid=4).half().eval()
    independent.load_state_dict(serialized_state(standalone), strict=True)
    with torch.inference_mode():
        assert torch.equal(independent(pixels).hidden_states[-1], expected)
    del standalone, independent

    vc = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                            num_attention_heads=2, image_size=8, patch_size=2,
                            vision_use_head=False, attention_dropout=0.)
    tc = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                    num_attention_heads=2, num_key_value_heads=1, vocab_size=64,
                    max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True)
    cfg = LlavaOnevisionConfig(vision_config=vc, text_config=tc, image_token_index=63,
                              image_grid_pinpoints=[[8, 8], [8, 16], [16, 8]],
                              vision_feature_layer=-1, vision_feature_select_strategy='full',
                              vision_aspect_ratio='anyres_max_9')
    Bridge = bridge_class()
    def fresh():
        return Bridge(LlavaOnevisionForConditionalGeneration(copy.deepcopy(cfg))).half().eval()
    model = fresh()
    # Non-square source has one global tile plus two local tiles.
    non_square_pixels = pixels.unsqueeze(0)
    sizes = torch.tensor([[8, 16]])
    with torch.inference_mode():
        old_packed = model.visual(non_square_pixels, sizes)
    old_count = parameter_count(model)
    removed_count = parameter_count(model.core.vision_tower) + parameter_count(model.core.multi_modal_projector)
    old_storage = set(storages(model.core.vision_tower)) | set(storages(model.core.multi_modal_projector))
    old_tower_ref = weakref.ref(model.core.vision_tower)
    old_projector_ref = weakref.ref(model.core.multi_modal_projector)
    interface = VisionInterface(student(SMALL), output_width=16, native_size=12, output_grid=4).half().eval()
    added_count = parameter_count(interface)
    install(model, interface)
    del interface
    gc.collect()
    assert old_tower_ref() is None and old_projector_ref() is None, 'Old registered visual aliases survive'
    assert model.visual.tower is model.core.vision_tower
    assert model.visual.projector is model.core.multi_modal_projector
    assert isinstance(model.core.multi_modal_projector, torch.nn.Identity)
    assert set(storages(model)) == set(storages(model.core)), 'Core-only accounting omits bridge weights'
    assert set(storages(model)).isdisjoint(old_storage)
    assert parameter_count(model) == old_count - removed_count + added_count
    assert not any(p.requires_grad for p in model.parameters())
    assert (model.core.config.vision_config.image_size, model.core.config.vision_config.patch_size) == (8, 2)
    assert model.core.config.image_grid_pinpoints == [[8, 8], [8, 16], [16, 8]]
    assert model.core.config.vision_feature_layer == -1
    assert model.core.config.vision_feature_select_strategy == 'full'
    with torch.inference_mode():
        packed = model.visual(non_square_pixels, sizes)
        per_image = model.core.get_image_features(non_square_pixels, sizes, -1, 'full')
        packed_direct, lengths = model.core.pack_image_features(per_image, sizes, model.core.image_newline,
                                                               model.core.config.vision_aspect_ratio)
    assert torch.equal(packed, packed_direct) and torch.isfinite(packed).all()
    assert packed.shape == old_packed.shape and packed.shape[1] == 16
    assert lengths.tolist() == [len(packed)]
    # Original text IDs and positions remain legal for the unchanged pack length.
    ids = torch.tensor([[1] + [63] * len(packed) + [2, 3]])
    attention = torch.ones_like(ids)
    assert (ids == 63).sum().item() == len(packed)
    with torch.inference_mode():
        embeddings = model.get_input_embeddings()(ids)
        mask = (ids == 63).unsqueeze(-1).expand_as(embeddings)
        scatter = embeddings.masked_scatter(mask, packed)
        assert torch.equal(scatter[:, ids[0] != 63], embeddings[:, ids[0] != 63])
        native = model.core(input_ids=ids, attention_mask=attention, pixel_values=non_square_pixels,
                            image_sizes=sizes, use_cache=False).logits
        generated = model.generate(input_ids=ids, attention_mask=attention, pixel_values=non_square_pixels,
                                   image_grid_thw=sizes, do_sample=False, max_new_tokens=1,
                                   pad_token_id=0, eos_token_id=62)
    assert torch.isfinite(native).all() and native.shape[:2] == ids.shape
    assert torch.equal(generated[:, :ids.shape[1]], ids) and generated.shape[1] == ids.shape[1] + 1

    # A fresh independent architecture with the same interface must strict-load
    # the complete graph, without retaining an old tower in either registry.
    snapshot = serialized_state(model.core)
    restored = fresh()
    install(restored, VisionInterface(student(SMALL), output_width=16, native_size=12, output_grid=4).half())
    restored.core.load_state_dict(snapshot, strict=True)
    assert parameter_count(restored) == parameter_count(model)
    assert set(storages(restored)) == set(storages(restored.core))
    with torch.inference_mode():
        assert torch.equal(restored.visual(non_square_pixels, sizes), packed)
        reloaded_native = restored.core(input_ids=ids, attention_mask=attention, pixel_values=non_square_pixels,
                                         image_sizes=sizes, use_cache=False).logits
        assert torch.equal(reloaded_native, native)
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(passed=True, real_tiny_Idefics3_eager_FP16=True,
                          independent_strict_reload_exact=True, original_visual_aliases_removed=True,
                          core_bridge_storage_sets_equal=True, outer_pack_config_preserved=True,
                          non_square_pack_and_text_positions_valid=True,
                          QA_equal_weight_centered_ridge_matches_explicit_WLS=True,
                          unequal_QA_lengths_and_token_duplication_checked=True,
                          packed_tokens=len(packed), real_QA=0, CUDA_initialized=False,
                          seconds=time.monotonic() - tick), indent=2))


if __name__ == '__main__':
    main()
