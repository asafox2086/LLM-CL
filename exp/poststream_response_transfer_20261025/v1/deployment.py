"""WIP: complete GPU deployment graphs and fixed, unscored DEV readouts.

The caller binds the unchanged storage helper and runtime sources in READY.
Only tensors/configuration are serialized. No method, optimizer, teacher, cache
or original removed FFN is part of a newly exported package. CPU tensor copies
and hashes are serialization/verification I/O, never model execution.
"""
from pathlib import Path
import gc
import hashlib
import json

INDEXES = [2, 5, 13, 16, 17, 18, 20, 21]
FULL = {'coefficients': 893675552, 'bytes': 1787351104}
PRUNED = {'coefficients': 789080096, 'bytes': 1578160192}
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
FORMAT = 'llmcl-poststream-pruned-native-v1'


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            value.update(block)
    return value.hexdigest()


def _device(device):
    import torch
    device = torch.device(device)
    assert device.type == 'cuda', 'CPU is restricted to I/O and scalar statistics'
    return torch.device('cuda', torch.cuda.current_device() if device.index is None else device.index)


def _buffers(model, storage):
    values = list(model.core.named_buffers(remove_duplicate=False))
    for name, value in values:
        assert value.device.type == 'cuda', name
        if name.endswith('inv_freq'):
            import torch
            assert value.dtype == torch.float32, name
    nonvision = [(name, value) for name, value in values
                 if not name.startswith(('vision_tower.', 'multi_modal_projector.'))]
    return dict(all_buffers_digest=storage.digest(values),
                nonvision_buffers_digest=storage.digest(nonvision),
                buffer_digest=storage.digest(nonvision))


def _receipt(model, storage, indexes, digest=None):
    import torch
    core = model.core
    assert indexes in ([], INDEXES)
    assert len(model.model.layers) == 24
    assert model.model is core.language_model.model and model.lm_head is core.language_model.lm_head
    assert model.visual.tower is core.vision_tower
    assert model.visual.projector is core.multi_modal_projector
    assert model.visual.image_newline is core.image_newline
    assert core.language_model.model.embed_tokens.weight is core.language_model.lm_head.weight
    assert storage._measure(model.named_parameters(remove_duplicate=False)) == storage.layout(model)['weights']
    assert all(p.dtype == torch.float16 and p.device == model.device for p in core.parameters())
    assert all(not p.requires_grad and p.grad is None for p in core.parameters())
    keys = core.state_dict().keys()
    assert not any('.A.' in name or '.B.' in name for name in keys)
    for i, layer in enumerate(model.model.layers):
        if i in indexes:
            assert isinstance(layer.mlp, storage.ZeroFFN)
            assert not list(layer.mlp.parameters()) and not list(layer.mlp.buffers())
            assert not any(f'language_model.model.layers.{i}.mlp.' in name for name in keys)
        else:
            assert all(isinstance(getattr(layer.mlp, name), torch.nn.Linear)
                       for name in ('gate_proj', 'up_proj', 'down_proj'))
        assert all(isinstance(getattr(layer.self_attn, name), torch.nn.Linear)
                   for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'))
    layout = storage.layout(model)
    assert layout['weights'] == (PRUNED if indexes else FULL) and layout['trainable'] == 0
    assert core.language_model.config._attn_implementation == 'sdpa'
    assert core.vision_tower.config._attn_implementation == 'sdpa'
    result = dict(indexes=list(indexes), layout=layout,
                  language_attention_classes=sorted({type(x.self_attn).__name__ for x in model.model.layers}),
                  language_attention_backend=core.language_model.config._attn_implementation,
                  vision_attention_backend=core.vision_tower.config._attn_implementation,
                  embedding_head_tied=True, bridge_aliases=True, device=str(model.device),
                  **_buffers(model, storage))
    if digest is not None:
        result['digest'] = digest
    return result


def _install_inputs(model):
    import onevision_bridge as bridge
    import frozen_vision as fv
    bridge.bind(model)
    bridge.LAST_NATIVE = None
    bridge.NATIVE_GATE_MODE = False
    fv.LIMIT = 0
    if not hasattr(model, '_llmcl_visual_original'):
        fv.install(model)
    model._llmcl_visual_cache.clear()
    model._llmcl_visual_cache_bytes = 0
    model.eval()


def load_static(path, base_path, storage, device='cuda', expected_digest=None, expected_layout=None):
    """Strict reload without rehashing the whole checkpoint file.

    Legacy parent/pruned packages require state/indexes/digest/layout; their
    native configuration comes from the caller's bound original base path.
    New packages additionally preserve native and generation configurations.
    No pretrained matrix is loaded and no CPU model is constructed.
    """
    import torch
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration, GenerationConfig
    import onevision_bridge as bridge
    device = _device(device)
    package = torch.load(Path(path), map_location='cpu', weights_only=True)
    assert isinstance(package, dict) and {'state', 'indexes', 'digest', 'layout'} <= set(package)
    indexes = package['indexes']
    assert indexes in ([], INDEXES)
    if 'format' in package:
        assert package['format'] == FORMAT and indexes == INDEXES
    if expected_digest is not None:
        assert package['digest'] == expected_digest
    if expected_layout is not None:
        assert package['layout'] == expected_layout
    if 'base_config' in package:
        config = LlavaOnevisionConfig(**package['base_config'])
    else:
        config = LlavaOnevisionConfig.from_pretrained(base_path, local_files_only=True)
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    with torch.device(device):
        core = LlavaOnevisionForConditionalGeneration(config)
        model = bridge.OneVisionBridge(core)
    if indexes:
        removed = storage.delete(model, indexes)
        # Fresh initialization was FP32, so only coefficient removal is checked.
        assert removed['coefficients'] == 104595456
    # Preserve the original FP32, nonpersistent RoPE buffers and tied Parameters.
    with torch.no_grad():
        for parameter in core.parameters():
            parameter.data = parameter.data.to(device=device, dtype=torch.float16)
    core.load_state_dict(package['state'], strict=True)
    if 'language_generation_config' in package:
        core.language_model.generation_config = GenerationConfig.from_dict(package['language_generation_config'])
        core.generation_config = GenerationConfig.from_dict(package['generation_config'])
    model.requires_grad_(False).eval()
    digest = storage.digest(model)
    assert digest == package['digest']
    result = _receipt(model, storage, indexes, digest)
    assert result['layout'] == package['layout']
    if 'all_buffers_digest' in package:
        assert result['all_buffers_digest'] == package['all_buffers_digest']
        assert result['nonvision_buffers_digest'] == package['nonvision_buffers_digest']
        assert result['language_attention_classes'] == package['language_attention_classes']
    del package
    gc.collect()
    _install_inputs(model)
    result.update(path=str(path), bytes=Path(path).stat().st_size,
                  strict_complete_state_reload=True, no_pretrained_weight_reload=True,
                  parameter_only_FP16_preserved_FP32_RoPE=True)
    return model, result


def export_static(model, path, storage):
    """Export only the actual fixed pruned graph, with tied CPU state aliases.

    GPU folding must already be complete. Hash the new file once after save;
    subsequent reload checks actual complete tensors, not another file hash.
    """
    import torch
    path = Path(path)
    assert not path.exists()
    model.eval()
    receipt = _receipt(model, storage, INDEXES)
    state = storage.state_cpu(model)
    assert all(isinstance(v, torch.Tensor) and v.device.type == 'cpu' for v in state.values())
    assert state['language_model.model.embed_tokens.weight'] is state['language_model.lm_head.weight']
    digest = storage.digest(state)
    config = model.core.config.to_dict()
    generation = model.core.generation_config.to_dict()
    language_generation = model.core.language_model.generation_config.to_dict()
    json.dumps([config, generation, language_generation], allow_nan=False)
    package = dict(format=FORMAT, state=state, indexes=list(INDEXES), digest=digest,
                   layout=receipt['layout'], base_config=config,
                   generation_config=generation, language_generation_config=language_generation,
                   all_buffers_digest=receipt['all_buffers_digest'],
                   nonvision_buffers_digest=receipt['nonvision_buffers_digest'],
                   language_attention_classes=receipt['language_attention_classes'])
    torch.save(package, path)
    del package, state
    receipt.update(digest=digest, path=str(path), bytes=path.stat().st_size, sha256=_sha(path),
                   package_format=FORMAT, aliases_preserved_in_CPU_snapshot=True,
                   no_teacher_optimizer_adapter_or_removed_FFN_payload=True)
    return receipt


def _prior_map(records):
    if isinstance(records, dict):
        assert all(key == value['id'] for key, value in records.items())
        return records
    result = {record['id']: record for record in records}
    assert len(result) == len(records)
    return result


def _rows(rows, prior):
    assert len(rows) == len(prior) == 100
    assert len({row['id'] for _, row in rows}) == 100
    assert {row['id'] for _, row in rows} == set(prior)
    assert all(sum(task['domain'] == domain for task, _ in rows) == 20 for domain in DOMAINS)


def _batch(model, task, row, processor, reference, storage, batch_provider):
    raw = {}

    def capture(module, args):
        assert not raw
        raw.update(raw_pixels_sha256=storage.digest({'pixels': args[0]}), tiles=args[0].shape[0])

    handle = model.core.vision_tower.register_forward_pre_hook(capture)
    try:
        if batch_provider is None:
            from runtime_utils import input_batch
            batch, details = input_batch(row, task, processor, model.device)
        else:
            batch, details = batch_provider(model, task, row)
    finally:
        handle.remove()
    assert raw and model._llmcl_visual_cache_bytes == 0
    prepared = storage.digest(batch)
    transport = storage.digest({name: value for name, value in batch.items() if name != 'pixel_values'})
    expected = reference.get('prepared_sha256', reference.get('input_sha256'))
    assert isinstance(expected, str) and prepared == expected
    assert row['answers'] == reference['references'] and task['max_new_tokens'] == reference['generation_cap']
    for name, value in (('prepared_sha256', prepared), ('input_sha256', prepared),
                        ('transport_sha256', transport), ('packed_tokens', len(batch['pixel_values']))):
        if name in details:
            assert details[name] == value
        if name in reference:
            assert reference[name] == value
    for name, value in raw.items():
        if name in reference:
            assert reference[name] == value
    return batch, dict(prepared_sha256=prepared, input_sha256=prepared,
                       transport_sha256=transport, packed_tokens=len(batch['pixel_values']),
                       input_shape=list(batch['input_ids'].shape),
                       text_sha256=storage.digest({'input_ids': batch['input_ids']}), **raw)


def _generate(model, processor, task, batch, fixed=False):
    import torch

    class FiniteScores:
        def __call__(self, input_ids, scores):
            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
            assert torch.isfinite(scores).any(dim=-1).all()
            return scores

    options = dict(do_sample=False, num_beams=1, max_new_tokens=32 if fixed else task['max_new_tokens'],
                   use_cache=True, pad_token_id=processor.tokenizer.pad_token_id,
                   eos_token_id=processor.tokenizer.eos_token_id, logits_processor=[FiniteScores()])
    if fixed:
        options['min_new_tokens'] = 32
    with torch.inference_mode(), torch.autocast('cuda', dtype=model.dtype):
        output = model.generate(**batch, **options)
    result = output[0, batch['input_ids'].shape[1]:].tolist()
    assert len(result) == 32 if fixed else len(result) <= task['max_new_tokens']
    return result


def evaluate_dev(model, processor, rows, prior_predictions, storage, *, batch_provider=None):
    """Generate original 100 DEV answers, without NLL or any quality scoring."""
    prior = _prior_map(prior_predictions)
    _rows(rows, prior)
    _install_inputs(model)
    records = []
    for task, row in rows:
        batch, binding = _batch(model, task, row, processor, prior[row['id']], storage, batch_provider)
        tokens = _generate(model, processor, task, batch)
        records.append(dict(id=row['id'], task=task['task'], domain=task['domain'], **binding,
                            prediction=processor.tokenizer.decode(tokens, skip_special_tokens=True),
                            output_tokens=tokens, references=row['answers'], generation_cap=task['max_new_tokens']))
        del batch
    assert model._llmcl_visual_cache_bytes == 0
    return records, dict(new_generations=100, NLL_requests=0, new_scores=0,
                         backwards=0, optimizer_updates=0, original_prepared_inputs_exact=True)


def resource_windows(model, processor, rows, prior_predictions, prior_windows, storage, *, batch_provider=None):
    """Call after 100 DEV to match the original direct-ablation parent windows."""
    import torch
    prior = _prior_map(prior_predictions)
    windows = _prior_map(prior_windows)
    _rows(rows, prior)
    assert len(windows) == 5 and {v['domain'] for v in windows.values()} == set(DOMAINS)
    _install_inputs(model)
    result = []
    device = _device(model.device)
    for domain in DOMAINS:
        task, row = min(((task, row) for task, row in rows if task['domain'] == domain),
                        key=lambda pair: hashlib.sha256(pair[1]['id'].encode()).digest())
        old = windows[row['id']]
        assert old['domain'] == domain and old['new_tokens'] == 32
        import onevision_bridge as bridge
        bridge.LAST_NATIVE = None
        gc.collect()
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
            batch, binding = _batch(model, task, row, processor, prior[row['id']], storage, batch_provider)
            for key in ('input_sha256', 'prepared_sha256', 'transport_sha256', 'text_sha256',
                        'input_shape', 'raw_pixels_sha256', 'tiles', 'packed_tokens'):
                if key in old:
                    assert binding[key] == old[key]
            if 'prompt_tokens' in old:
                assert batch['input_ids'].shape[1] == old['prompt_tokens']
            tokens = _generate(model, processor, task, batch, True)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
        result.append(dict(id=row['id'], domain=domain, **binding,
                           prompt_tokens=batch['input_ids'].shape[1], new_tokens=32,
                           baseline_allocated=baseline, peak_allocated=peak, peak_reserved=reserved))
        del batch, tokens
    assert model._llmcl_visual_cache_bytes == 0
    return result, dict(new_generations=5, new_scores=0, NLL_requests=0,
                        actual_device=str(device), same_original_hash_min_ids=True,
                        fixed_new_tokens=32, cache_limit_bytes=0,
                        window_order='after_original_100_DEV')


def finish_proof(model, initial_receipt, storage):
    """One final full tensor digest, plus all nonpersistent buffers and aliases."""
    result = _receipt(model, storage, initial_receipt['indexes'], storage.digest(model))
    for key in ('digest', 'layout', 'all_buffers_digest', 'nonvision_buffers_digest',
                'language_attention_classes', 'language_attention_backend', 'vision_attention_backend',
                'embedding_head_tied', 'bridge_aliases', 'device'):
        assert result[key] == initial_receipt[key], key
    result.update(digest_before=initial_receipt['digest'], digest_after=result['digest'],
                  complete_state_and_buffers_unchanged=True, all_parameters_frozen_no_grad=True)
    return result
