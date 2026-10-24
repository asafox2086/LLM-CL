"""One synthetic CPU integration check of adaptation.py, without pretrained data."""
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


def storage_layout(module):
    result = {}
    for parameter in module.parameters():
        storage = parameter.untyped_storage()
        key = (str(parameter.device), storage.data_ptr(), storage.nbytes())
        result[key] = storage.nbytes() // parameter.element_size()
    return result


def parameters_half(module):
    # Match the production loading rule; preserve FP32 nonpersistent RoPE buffers.
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.data = parameter.data.to(dtype=torch.float16)


def expected_fold(model, plan):
    result = {}
    refs = []
    with torch.no_grad():
        for name in plan:
            layer = model.get_submodule(name)
            layer.B.weight.fill_(.001)
            result[name] = (layer.weight.detach().cpu().float() + layer.scale *
                            (layer.B.weight.detach().cpu().float() @ layer.A.weight.detach().cpu().float())).to(layer.weight.dtype)
            refs.append(weakref.ref(layer))
    return result, refs


def main():
    tick = time.monotonic()
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    root = Path(os.environ['LLMCL_ROOT'])
    candidates = ([Path(os.environ['LLMCL_ASSET'])] if os.environ.get('LLMCL_ASSET') else [])
    candidates += [root / '.deployment/paired_image_support_20261024_view', root]
    assets = next(path for path in candidates
                  if (path / 'exp/osft_increment_screen_20261019/v1/study.py').is_file())
    sys.path.insert(0, str(assets / 'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0, str(assets / 'exp/osft_increment_screen_20261019/v1'))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if not (Path(__file__).resolve().parent / 'vision_interface.py').is_file():
        sys.path.insert(0, str(root / 'exp/vision_interface_transfer_20261025/v1'))
    from onevision_bridge import OneVisionBridge
    import adaptation
    import vision_interface as vi

    run = Path(os.environ.get('LLMCL_RUN', str(Path(__file__).resolve().parent)))
    budget_source = next(path for path in [run / 'inherited_budget_core.py',
                        Path(__file__).resolve().parent / 'inherited_budget_core.py',
                        root / 'exp/position_coordinate_control_20261022/v1/core.py']
                         if path.is_file())
    small = dict(hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                 num_attention_heads=2, num_channels=3, image_size=12, patch_size=2,
                 attention_dropout=0., layer_norm_eps=1e-6)
    vc = SiglipVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                           num_attention_heads=2, image_size=8, patch_size=2,
                           vision_use_head=False, attention_dropout=0.)
    tc = Qwen2Config(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, vocab_size=64,
                    max_position_embeddings=128, attention_dropout=0., tie_word_embeddings=True)
    cfg = LlavaOnevisionConfig(vision_config=vc, text_config=tc, image_token_index=63,
                              image_grid_pinpoints=[[8, 8]], vision_feature_layer=-1,
                              vision_feature_select_strategy='full', vision_aspect_ratio='anyres_max_9')
    cfg._attn_implementation = 'sdpa'
    cfg.text_config._attn_implementation = 'sdpa'
    cfg.vision_config._attn_implementation = 'sdpa'
    torch.manual_seed(7)
    model = OneVisionBridge(LlavaOnevisionForConditionalGeneration(copy.deepcopy(cfg)))
    parameters_half(model)
    interface = vi.VisionInterface(vi.student(small), output_width=16,
                                   native_size=12, output_grid=4)
    parameters_half(interface)
    old_tower = weakref.ref(model.core.vision_tower)
    old_projector = weakref.ref(model.core.multi_modal_projector)
    removed = sum(storage_layout(model.core.vision_tower).values()) + sum(storage_layout(model.core.multi_modal_projector).values())
    added = sum(storage_layout(interface).values())
    plan = {f'model.layers.{i}.self_attn.{kind}_proj': 12 + (kind == 'v')
            for i in range(2) for kind in ('q', 'v')}
    # The interface is constructed before the original adapter seed reset.
    torch.manual_seed(42)
    method = adaptation.make_budget(model, assets, budget_source, plan=plan)
    before_install = sum(storage_layout(model).values())
    originals = {name: (id(parameter), parameter.detach().clone())
                 for name, parameter in model.named_parameters() if parameter.requires_grad}
    adapter_coordinates = sum(value.numel() for _, value in originals.values())
    rng = torch.get_rng_state().clone()
    report = adaptation.install_interface_preserving_adapters(model, interface, method)
    assert torch.equal(rng, torch.get_rng_state())
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert set(current) == set(originals)
    assert all(id(current[name]) == identity and torch.equal(current[name], value)
               for name, (identity, value) in originals.items())
    assert len(current) == 8 and report['adapter_coordinates'] == adapter_coordinates
    assert all(parameter.dtype == torch.float32 for parameter in current.values())
    del interface
    gc.collect()
    assert old_tower() is None and old_projector() is None
    assert set(storage_layout(model)) == set(storage_layout(model.core))
    assert sum(storage_layout(model).values()) == before_install - removed + added
    assert not any(parameter.requires_grad for parameter in model.visual.parameters())
    inv_freq = {name: value.detach().clone() for name, value in model.core.named_buffers()
                if name.endswith('inv_freq')}
    assert inv_freq and all(value.dtype == torch.float32 for value in inv_freq.values())
    assert model.core.language_model.model.embed_tokens.weight is model.core.language_model.lm_head.weight

    # New integration only: the real compact vision/decoder graph carries a
    # synthetic CE gradient to the restored A/B parameters, and nowhere else.
    model.eval()
    # AnyRes supplies one global tile plus one local tile for this square image.
    pixels = torch.linspace(-1, 1, 2 * 3 * 8 * 8, dtype=torch.float16).reshape(1, 2, 3, 8, 8)
    sizes = torch.tensor([[8, 8]])
    with torch.no_grad():
        packed = model.visual(pixels, sizes)
    ids = torch.tensor([[1] + [63] * len(packed) + [2, 3]])
    mask = torch.ones_like(ids)
    logits = model.core(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                        image_sizes=sizes, use_cache=False).logits
    loss = torch.nn.functional.cross_entropy(logits[:, -1, :].float(), torch.tensor([4]))
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in current.values())
    assert any(parameter.grad.abs().sum() > 0 for name, parameter in current.items() if name.endswith('.B.weight'))
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    model.zero_grad(set_to_none=True)
    del logits, loss, current, originals

    expected, retired_refs = expected_fold(model, plan)
    with_adapters = sum(storage_layout(model).values())
    folded = adaptation.fold(model, method)
    assert len(folded) == 4 and not method.layers
    assert all(record['merge_device'] == 'cpu' and record['merge_dtype'] == 'torch.float32'
               and record['output_device'] == 'cpu' for record in folded)
    assert all(torch.equal(model.get_submodule(name).weight, value) for name, value in expected.items())
    assert not any('.A.' in name or '.B.' in name for name in model.core.state_dict())
    assert sum(storage_layout(model).values()) == with_adapters - adapter_coordinates
    assert not any(parameter.requires_grad for parameter in model.parameters())
    gc.collect()
    assert all(reference() is None for reference in retired_refs), 'Retired adapters retained by the model or method'
    with torch.no_grad():
        folded_logits = model.core(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                                   image_sizes=sizes, use_cache=False).logits
    package = adaptation.export_package(model, small)
    assert package['interface_config'] == dict(output_width=16, native_size=12, output_grid=4)
    buffer = io.BytesIO()
    torch.save(package, buffer)
    buffer.seek(0)
    restored = adaptation.load_export(torch.load(buffer, map_location='cpu', weights_only=True), device='cpu')
    assert sum(storage_layout(restored).values()) == sum(storage_layout(model).values())
    assert set(storage_layout(restored)) == set(storage_layout(restored.core))
    assert not any('.A.' in name or '.B.' in name for name in restored.core.state_dict())
    assert restored.core.language_model.model.embed_tokens.weight is restored.core.language_model.lm_head.weight
    for name, value in restored.core.named_buffers():
        if name in inv_freq:
            assert value.dtype == torch.float32 and torch.equal(value, inv_freq[name])
    with torch.no_grad():
        actual = restored.core(input_ids=ids, attention_mask=mask, pixel_values=pixels,
                               image_sizes=sizes, use_cache=False).logits
    assert torch.equal(actual, folded_logits)
    assert torch.isfinite(actual).all() and not torch.cuda.is_initialized()
    print(json.dumps(dict(passed=True, seconds=time.monotonic() - tick,
                          production_adapter_install_and_fold_and_reload=True,
                          adapter_identity_bytes_RNG_preserved=True,
                          only_adapter_finite_gradients=True,
                          old_visual_aliases_removed=True,
                          folded_adapter_modules_released=True,
                          fold_matches_FP32_merge_then_FP16_storage=True,
                          serialized_strict_reload_logits_exact=True,
                          preserved_FP32_RoPE_and_embedding_tie=True,
                          adapter_coordinates=adapter_coordinates,
                          final_coefficients=sum(storage_layout(model).values()),
                          synthetic_backwards=1, optimizer_updates=0,
                          real_QA=0, pretrained_weights=0, CUDA_initialized=False), allow_nan=False))


if __name__ == '__main__':
    main()
