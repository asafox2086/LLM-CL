"""Complete OneVision/QV-only control packages; no Torch work on import.

Numerical model work stays on GPU. CPU tensors are checkpoint I/O snapshots.
Reload constructs an empty meta graph and requires no external base weights.
The caller owns bridge.bind, frozen-vision installation, batches and evaluation.
"""
from pathlib import Path
import gc
import hashlib
import json

FORMAT = 'native-response-qv-control-complete-v1'
BASE_WEIGHTS = {'coefficients': 893675552, 'bytes': 1787351104}
RESPONSE_WEIGHTS = {'coefficients': 893675576, 'bytes': 1787351200}


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def jd(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def functional(value):
    if isinstance(value, dict):
        return {key: functional(item) for key, item in value.items() if key != '_name_or_path'}
    if isinstance(value, (list, tuple)):
        return [functional(item) for item in value]
    return value


def _alias_key(tensor):
    return (str(tensor.device), tensor.untyped_storage().data_ptr(),
            tensor.untyped_storage().nbytes(), tensor.storage_offset(),
            tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


def aliases(items):
    groups = {}
    for name, tensor in items:
        groups.setdefault(_alias_key(tensor), []).append(name)
    return sorted(sorted(names) for names in groups.values() if len(names) > 1)


def _rebind(model):
    core = model.core
    model.model = core.language_model.model
    model.lm_head = core.language_model.lm_head
    model.visual.tower = core.vision_tower
    model.visual.projector = core.multi_modal_projector
    # assign=True replaces this Parameter; the bridge's registered alias must
    # follow the new object rather than keep its meta constructor Parameter.
    model.visual.image_newline = core.image_newline


def receipt(model, storage, response_installed=False):
    import torch
    if response_installed:
        import response
    core = model.core
    device = next(core.parameters()).device
    assert device.type == 'cuda'
    parameters = list(core.named_parameters(remove_duplicate=False))
    beta_names = {'language_model.model.layers.%d.mlp.beta' % i for i in range(24)}
    actual_beta = {name for name, _ in parameters if name.endswith('.mlp.beta')}
    assert actual_beta == (beta_names if response_installed else set())
    for name, parameter in parameters:
        assert parameter.device == device and not parameter.requires_grad and parameter.grad is None, name
        assert parameter.dtype == (torch.float32 if name in beta_names else torch.float16), name
        assert not name.endswith(('.A.weight', '.B.weight')), name
        if name in beta_names:
            assert parameter.shape == () and torch.isfinite(parameter).all(), name
    buffers = list(core.named_buffers(remove_duplicate=False))
    assert all(buffer.device == device for _, buffer in buffers)
    for name, buffer in buffers:
        if name.endswith('inv_freq'):
            assert buffer.dtype == torch.float32, name
    layout = storage.layout(core)
    assert layout['weights'] == (RESPONSE_WEIGHTS if response_installed else BASE_WEIGHTS)
    assert layout['trainable'] == 0
    assert model.model is core.language_model.model and model.lm_head is core.language_model.lm_head
    assert model.visual.tower is core.vision_tower and model.visual.projector is core.multi_modal_projector
    assert model.visual.image_newline is core.image_newline
    # No additional bridge-only tensors may escape the core layout/digest.
    assert {id(p) for p in model.parameters()} == {id(p) for p in core.parameters()}
    assert {id(b) for b in model.buffers()} == {id(b) for b in core.buffers()}
    assert core.language_model.config._attn_implementation == 'sdpa'
    assert core.vision_tower.config._attn_implementation == 'sdpa'
    metadata = response.metadata(model) if response_installed else None
    if not response_installed:
        assert layout['ordinary_buffers']['bytes'] == 9032
        assert not any(name.endswith(('.mlp.response_sign', '.mlp.calibration_scale')) for name, _ in buffers)
    if response_installed:
        signs = [(name, b) for name, b in buffers if name.endswith('.mlp.response_sign')]
        calibration = [(name, b) for name, b in buffers if name.endswith('.mlp.calibration_scale')]
        assert len(signs) == 24 and sum(b.numel() * b.element_size() for _, b in signs) == 233472
        expected_calibration = metadata['arm'] == 'static' and not metadata['final']
        assert len(calibration) == (24 if expected_calibration else 0)
        assert sum(b.numel() * b.element_size() for _, b in calibration) == metadata['calibration_buffer_bytes']
        for name, b in calibration:
            assert b.shape == () and b.dtype == torch.float32 and torch.isfinite(b).all() and b > 0, name
    return dict(layout=layout, digest=storage.digest(core),
                all_buffers_digest=storage.digest(buffers),
                buffer_manifest={name: {'shape': list(b.shape), 'dtype': str(b.dtype),
                                        'bytes': b.numel() * b.element_size()} for name, b in buffers},
                parameter_alias_groups=aliases(parameters), buffer_alias_groups=aliases(buffers),
                device=str(device), functional_config_digest=jd(functional(core.config.to_dict())),
                generation_config_digest=jd(core.generation_config.to_dict()),
                language_generation_config_digest=jd(core.language_model.generation_config.to_dict()),
                language_attention_classes=sorted({type(layer.self_attn).__name__ for layer in model.model.layers}),
                vision_attention_classes=sorted({type(layer.self_attn).__name__
                                                 for layer in core.vision_tower.vision_model.encoder.layers}),
                response=metadata)


def load_base(base, device, storage):
    """One standard GPU weight load; response installation is the caller's job."""
    import torch
    import onevision_bridge as bridge
    device = torch.device(device)
    assert device.type == 'cuda'
    core, info = bridge.LlavaOnevisionForConditionalGeneration.from_pretrained(
        base, torch_dtype=torch.float16, attn_implementation='sdpa', local_files_only=True,
        device_map={'': str(device)}, low_cpu_mem_usage=True, output_loading_info=True)
    assert not any(info.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')), info
    model = bridge.OneVisionBridge(core).requires_grad_(False).eval()
    return model, receipt(model, storage, response_installed=False)


def _cpu_buffers(core, state):
    """Preserve exact aliases within buffers and between buffers and state."""
    bank = {_alias_key(tensor): state[name] for name, tensor in core.state_dict().items()}
    result = {}
    for name, tensor in core.named_buffers(remove_duplicate=False):
        key = _alias_key(tensor)
        if key not in bank:
            bank[key] = tensor.detach().cpu().clone()
        result[name] = bank[key]
    return result


def export(model, path, storage, response_installed=False):
    import torch
    value = receipt(model, storage, response_installed=response_installed)
    path = Path(path)
    assert not path.exists()
    core = model.core
    state = storage.state_cpu(core)
    assert storage.digest(state) == value['digest']
    buffers = _cpu_buffers(core, state)
    assert storage.digest(buffers) == value['all_buffers_digest']
    for config in (core.generation_config, core.language_model.generation_config):
        assert hasattr(config, '_original_object_hash') and config._original_object_hash == hash(config)
    metadata = value['response']
    scales = ([float(layer.mlp.calibration_scale.detach().item()) for layer in model.model.layers]
              if metadata is not None and metadata['arm'] == 'static' and not metadata['final'] else None)
    package = dict(format=FORMAT, state=state, buffers=buffers,
                   base_config=core.config.to_dict(), response_metadata=metadata,
                   scales=scales, receipt=value, generation_config=core.generation_config.to_dict(),
                   language_generation_config=core.language_model.generation_config.to_dict())
    torch.save(package, path)
    del package, state, buffers
    gc.collect()
    return dict(value, path=str(path), bytes=path.stat().st_size, sha256=sha(path),
                complete_graph=True, external_base_weights_required=False,
                no_adapter_optimizer_or_teacher=True, alias_preserving_cpu_snapshot=True)


def _transfer(items, device, bank):
    result = {}
    for name, tensor in items.items():
        assert tensor.device.type == 'cpu'
        key = _alias_key(tensor)
        if key not in bank:
            bank[key] = tensor.to(device=device)
        result[name] = bank[key]
    return result


def _generation(values):
    from transformers import GenerationConfig
    config = GenerationConfig.from_dict(values)
    # HF4.46's ordinary GenerationConfig loaders restore this private sentinel.
    # It is absent from to_dict; public fields and _from_model_config stay fixed.
    config._original_object_hash = hash(config)
    assert config.to_dict() == values
    return config


def load_complete(path, device, storage):
    import torch
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    import onevision_bridge as bridge
    device = torch.device(device)
    assert device.type == 'cuda'
    package = torch.load(path, map_location='cpu', weights_only=True)
    assert package['format'] == FORMAT
    metadata = package['response_metadata']
    assert metadata == package['receipt']['response']
    expected_scales = metadata is not None and metadata['arm'] == 'static' and not metadata['final']
    assert (package['scales'] is not None) == expected_scales
    config = LlavaOnevisionConfig.from_dict(package['base_config'])
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    with torch.device('meta'), no_init_weights():
        core = LlavaOnevisionForConditionalGeneration(config)
        model = bridge.OneVisionBridge(core)
        if metadata is not None:
            import response
            response.install(model, metadata['arm'], package['scales'], device='meta', final=metadata['final'])
    if metadata is not None:
        assert response.metadata(model) == metadata
    bank = {}
    core.load_state_dict(_transfer(package['state'], device, bank), strict=True, assign=True)
    assert set(package['buffers']) == {name for name, _ in core.named_buffers(remove_duplicate=False)}
    for name, buffer in _transfer(package['buffers'], device, bank).items():
        parent, _, leaf = name.rpartition('.')
        core.get_submodule(parent)._buffers[leaf] = buffer
    for module in core.modules():
        if hasattr(module, 'original_inv_freq'):
            assert getattr(module, 'rope_type', 'default') == 'default'
            module.original_inv_freq = module.inv_freq
    core.language_model.tie_weights()
    core.generation_config = _generation(package['generation_config'])
    core.language_model.generation_config = _generation(package['language_generation_config'])
    _rebind(model)
    model.requires_grad_(False).eval()
    actual = receipt(model, storage, response_installed=metadata is not None)
    for key, expected in package['receipt'].items():
        if key != 'device':
            assert actual[key] == expected, key
    assert all(parameter.device == device for parameter in core.parameters())
    del bank, package
    gc.collect()
    return model, dict(actual, self_contained_meta_construction=True,
                       all_buffers_explicitly_restored=True, strict_complete_reload=True,
                       mixed_parameter_dtypes_preserved=True)


def finish(model, before, storage, response_installed=False):
    after = receipt(model, storage, response_installed=response_installed)
    assert all(before[key] == value for key, value in after.items())
    return dict(after, complete_state_and_buffers_unchanged=True)
