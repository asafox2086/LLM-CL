"""Fixed native FP16 deployment and unscored readouts; no model work on import.

All model construction/loading uses the standard pretrained loader with GPU
placement. CPU tensor copies serve checkpoint I/O and proof hashes only.
The caller binds source/assets, original raw inputs and the phase order.
"""
from pathlib import Path
import gc
import hashlib
import json

DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
NATIVE_WEIGHTS = {'coefficients': 507482304, 'bytes': 1014964608}
PARENT_WEIGHTS = {'coefficients': 893675552, 'bytes': 1787351104}
PARENT_DIGEST = '681645dbc1ea4a1f5c2ac8ffe033278748408fd9439e0dbee1f568184784fcc8'
FORMAT = 'native-prior-learning-complete-v1'


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def _json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _functional(value):
    if isinstance(value, dict):
        return {k: _functional(v) for k, v in value.items() if k != '_name_or_path'}
    if isinstance(value, (list, tuple)):
        return [_functional(v) for v in value]
    return value


def _device(device):
    import torch
    value = torch.device(device)
    assert value.type == 'cuda', 'CPU is restricted to I/O/statistics'
    return torch.device('cuda', torch.cuda.current_device() if value.index is None else value.index)


def _aliases(items):
    groups = {}
    for name, tensor in items:
        key = (str(tensor.device), tensor.untyped_storage().data_ptr(),
               tensor.untyped_storage().nbytes(), tensor.storage_offset(),
               tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))
        groups.setdefault(key, []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _buffer_receipt(core, storage):
    import torch
    values = list(core.named_buffers(remove_duplicate=False))
    for name, value in values:
        assert value.is_cuda, name
        if name.endswith('inv_freq'):
            assert value.dtype == torch.float32, name
    return dict(all_buffers_digest=storage.digest(values),
                buffers=[dict(name=n, shape=list(v.shape), dtype=str(v.dtype),
                              bytes=v.numel() * v.element_size()) for n, v in values])


def receipt(model, storage, kind='native', include_digest=True):
    import torch
    assert kind in ('native', 'parent')
    core = model if kind == 'native' else model.core
    device = _device(next(core.parameters()).device)
    assert all(p.device == device and p.dtype == torch.float16 for p in core.parameters())
    assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    assert not any('.A.' in n or '.B.' in n for n in core.state_dict())
    layout = storage.layout(core)
    assert layout['weights'] == (NATIVE_WEIGHTS if kind == 'native' else PARENT_WEIGHTS)
    assert layout['trainable'] == 0
    if kind == 'native':
        language = core.model.text_model
        vision = core.model.vision_model
        assert len(language.layers) == 32
        assert vision.config._attn_implementation == 'eager'
        assert language.config._attn_implementation == 'sdpa'
        # Native head/embedding sharing is observed, never assumed.
        assert all(isinstance(getattr(layer.self_attn, n), torch.nn.Linear)
                   for layer in language.layers for n in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
    else:
        language, vision = core.language_model.model, core.vision_tower
        assert len(language.layers) == 24
        assert vision.config._attn_implementation == 'sdpa'
        assert core.language_model.config._attn_implementation == 'sdpa'
        assert model.model is language and model.lm_head is core.language_model.lm_head
        assert model.visual.tower is vision and model.visual.projector is core.multi_modal_projector
        assert storage._measure(model.named_parameters(remove_duplicate=False)) == layout['weights']
    result = dict(kind=kind, device=str(device), layout=layout,
                  parameter_alias_groups=_aliases(core.named_parameters(remove_duplicate=False)),
                  language_attention_classes=sorted({type(x.self_attn).__name__ for x in language.layers}),
                  vision_attention_classes=sorted({type(x.self_attn).__name__ for x in
                                                   (vision.vision_model.encoder.layers if kind == 'parent'
                                                    else vision.encoder.layers)}),
                  language_attention_backend='sdpa',
                  vision_attention_backend='eager' if kind == 'native' else 'sdpa',
                  functional_config_digest=_json_digest(_functional(core.config.to_dict())),
                  generation_config_digest=_json_digest(core.generation_config.to_dict()),
                  **_buffer_receipt(core, storage))
    if kind == 'parent':
        result['language_generation_config_digest'] = _json_digest(core.language_model.generation_config.to_dict())
    if include_digest:
        result['digest'] = storage.digest(core)
    return result


def load_native(base_path, storage, device, package_path=None, expected_digest=None):
    """Fresh standard GPU load, optionally overridden by one complete export.

    The native nonpersistent buffers originate from this same pretrained-loader
    path. They must match before/after strict override and the saved export.
    """
    import torch
    from transformers import Idefics3Config, Idefics3ForConditionalGeneration, GenerationConfig
    device = _device(device)
    package = None
    options = {}
    if package_path is not None:
        package = torch.load(Path(package_path), map_location='cpu', weights_only=True)
        assert package['format'] == FORMAT
        options['config'] = Idefics3Config.from_dict(package['base_config'])
    model, loading = Idefics3ForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.float16, local_files_only=True,
        attn_implementation={'vision_config': 'eager', 'text_config': 'sdpa'},
        device_map={'': str(device)}, low_cpu_mem_usage=True, output_loading_info=True, **options)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    before = _buffer_receipt(model, storage)
    if package is not None:
        model.load_state_dict(package['state'], strict=True)
        # Retain the standard loader's complete config object, including its
        # framework hash sentinel. Public configuration must equal the export.
        assert model.generation_config.to_dict() == package['generation_config']
        assert hasattr(model.generation_config, '_original_object_hash')
        assert _buffer_receipt(model, storage) == before
        assert before['all_buffers_digest'] == package['receipt']['all_buffers_digest']
    model.requires_grad_(False).eval()
    result = receipt(model, storage)
    if package is not None:
        for key in ('digest', 'layout', 'parameter_alias_groups', 'all_buffers_digest',
                    'functional_config_digest', 'generation_config_digest',
                    'language_attention_classes', 'vision_attention_classes'):
            assert result[key] == package['receipt'][key], key
    if expected_digest is not None:
        assert result['digest'] == expected_digest
    result.update(base_pretrained_weight_IO=True, full_random_CUDA_initialization=False,
                  strict_complete_state_override=package is not None,
                  same_loader_buffers_before_after_equal=True)
    del package
    gc.collect()
    return model, result


def export_native(model, path, storage):
    """Write a complete folded graph only; preserve exact CPU tensor aliases."""
    import torch
    path = Path(path)
    assert not path.exists()
    value = receipt(model, storage)
    state = storage.state_cpu(model)
    assert storage.digest(state) == value['digest']
    parameter_names = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    assert _aliases((n, v) for n, v in state.items() if n in parameter_names) == value['parameter_alias_groups']
    package = dict(format=FORMAT, state=state, base_config=model.config.to_dict(),
                   generation_config=model.generation_config.to_dict(), receipt=value)
    torch.save(package, path)
    del package, state
    return dict(**value, path=str(path), bytes=path.stat().st_size, sha256=_sha(path),
                complete_native_graph=True, no_teacher_optimizer_adapter=True,
                CPU_exact_aliases_preserved=True)


def load_parent_static(path, base_path, storage, device):
    """Original full parent for fresh same-card resource windows, not scoring.

    No comparison is made to buffers from the discarded random-CUDA loader.
    The original sealed complete parameter/state digest remains mandatory.
    """
    import torch
    from transformers import GenerationConfig, LlavaOnevisionConfig
    import onevision_bridge as bridge
    import frozen_vision as fv
    device = _device(device)
    package = torch.load(Path(path), map_location='cpu', weights_only=True)
    assert package['indexes'] == [] and package['digest'] == PARENT_DIGEST
    assert package['layout']['weights'] == PARENT_WEIGHTS
    model, loading = _parent_factory(base_path, device)
    assert not any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), loading
    before = _buffer_receipt(model.core, storage)
    model.core.load_state_dict(package['state'], strict=True)
    assert _buffer_receipt(model.core, storage) == before
    # Original direct readout specified tokenizer EOS/pad explicitly. Retain
    # from-config generation semantics for other native generation defaults.
    config = LlavaOnevisionConfig.from_pretrained(base_path, local_files_only=True)
    model.core.generation_config = GenerationConfig.from_model_config(config)
    model.core.language_model.generation_config = GenerationConfig.from_model_config(config.text_config)
    model.requires_grad_(False).eval()
    bridge.bind(model); bridge.LAST_NATIVE = None; bridge.NATIVE_GATE_MODE = False
    fv.LIMIT = 0
    fv.install(model)
    model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
    result = receipt(model, storage, 'parent')
    assert result['digest'] == PARENT_DIGEST and result['layout'] == package['layout']
    result.update(base_pretrained_weight_IO=True, strict_complete_state_override=True,
                  same_loader_buffers_before_after_equal=True, visual_cache_limit_bytes=0)
    del package
    gc.collect()
    return model, result


def _parent_factory(base_path, device):
    # Factory wraps a model only, so loading diagnostics use its underlying
    # standard HF class, then the identical bridge wrapper; no CPU forward.
    import torch
    import onevision_bridge as bridge
    core, loading = bridge.LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_path, torch_dtype=torch.float16, attn_implementation='sdpa',
        local_files_only=True, device_map={'': str(device)}, low_cpu_mem_usage=True,
        output_loading_info=True)
    return bridge.OneVisionBridge(core), loading


def _rows(rows):
    assert len(rows) == len({r['id'] for _, r in rows}) == 100
    assert all(sum(t['domain'] == d for t, _ in rows) == 20 for d in DOMAINS)


def _batch(model, task, row, storage, batch_provider):
    import torch
    batch, details = batch_provider(model, task, row)
    assert isinstance(details, dict) and isinstance(details['raw_input_sha256'], str)
    payload = {n: v for n, v in batch.items() if not n.startswith('_')}
    assert all(isinstance(v, torch.Tensor) and v.is_cuda for v in payload.values())
    assert all(not v.is_floating_point() or v.dtype == torch.float16 for v in payload.values())
    assert payload['input_ids'].shape[0] == 1
    prepared = storage.digest(payload)
    if 'prepared_sha256' in details:
        assert details['prepared_sha256'] == prepared
    binding = dict(details)
    binding.update(prepared_sha256=prepared, prompt_tokens=payload['input_ids'].shape[1],
                   input_shape=list(payload['input_ids'].shape),
                   input_tensor_shapes={n: list(v.shape) for n, v in payload.items()})
    return payload, binding


def _generate(model, payload, cap, generation_policy, fixed=False):
    import torch

    class FiniteScores:
        def __call__(self, input_ids, scores):
            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
            assert torch.isfinite(scores).any(dim=-1).all()
            return scores

    length = payload['input_ids'].shape[1]
    limit = generation_policy.get('context_limit')
    assert limit is None or length + cap <= limit
    options = dict(do_sample=False, num_beams=1, use_cache=True, max_new_tokens=cap,
                   eos_token_id=generation_policy['generation_stop_ids'],
                   pad_token_id=generation_policy['pad_id'], logits_processor=[FiniteScores()])
    if fixed:
        options['min_new_tokens'] = cap
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        output = model.generate(**payload, **options)
    tokens = output[0, length:].tolist()
    assert len(tokens) == cap if fixed else len(tokens) <= cap
    return tokens


def evaluate_dev(model, processor, rows, storage, *, batch_provider, generation_policy, kind='native'):
    """Exactly 100 original DEV generations; no NLL or scoring."""
    assert kind == 'native'
    _rows(rows)
    model.eval()
    records = []
    for task, row in rows:
        batch, binding = _batch(model, task, row, storage, batch_provider)
        tokens = _generate(model, batch, task['max_new_tokens'], generation_policy)
        records.append(dict(binding, id=row['id'], task=task['task'], domain=task['domain'],
                            prediction=processor.tokenizer.decode(tokens, skip_special_tokens=True),
                            references=row['answers'], generation_cap=task['max_new_tokens'],
                            output_tokens=tokens))
        del batch, tokens
    return records, dict(new_generations=100, new_scores=0, NLL_requests=0,
                         backwards=0, optimizer_updates=0, native_template=True)


def resource_windows(model, processor, rows, storage, *, batch_provider, generation_policy,
                     kind='native', prior_windows=None):
    """Five hash-min raw IDs, 32 tokens, fresh-load order fixed by caller.

    Call before any 100-DEV readout in the parent/final-native processes.
    Allocation windows include preparation and actual image/decoder execution.
    """
    import torch
    _rows(rows)
    device = _device(next(model.parameters()).device)
    model.eval()
    prior = None if prior_windows is None else {x['id']: x for x in prior_windows}
    if prior is not None:
        assert len(prior) == 5
    values = []
    for domain in DOMAINS:
        task, row = min(((t, r) for t, r in rows if t['domain'] == domain),
                        key=lambda tr: hashlib.sha256(tr[1]['id'].encode()).digest())
        if kind == 'parent':
            import onevision_bridge as bridge
            bridge.LAST_NATIVE = None
            assert model._llmcl_visual_cache_bytes == 0
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache(); torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            batch, binding = _batch(model, task, row, storage, batch_provider)
            if prior is not None:
                old = prior[row['id']]
                assert old['domain'] == domain and old['new_tokens'] == 32
                assert binding['raw_input_sha256'] == old['raw_input_sha256']
            tokens = _generate(model, batch, 32, generation_policy, True)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        values.append(dict(binding, id=row['id'], domain=domain, new_tokens=32,
                           baseline_allocated=baseline, peak_allocated=peak,
                           peak_reserved=reserved))
        del batch, tokens
    return values, dict(new_generations=5, new_scores=0, NLL_requests=0,
                        fixed_new_tokens=32, actual_device=str(device),
                        window_order='fresh_reload_before_any_DEV',
                        hash_min_raw_ids=True, cross_backbone_prepared_equality_required=False)


def finish_proof(model, initial_receipt, storage, kind='native'):
    result = receipt(model, storage, kind)
    for key in ('digest', 'layout', 'parameter_alias_groups', 'all_buffers_digest',
                'functional_config_digest', 'generation_config_digest',
                'language_attention_classes', 'vision_attention_classes', 'device'):
        assert result[key] == initial_receipt[key], key
    if kind == 'parent':
        assert result['language_generation_config_digest'] == initial_receipt['language_generation_config_digest']
    result.update(complete_state_and_buffers_unchanged=True, all_parameters_frozen_no_grad=True)
    return result
