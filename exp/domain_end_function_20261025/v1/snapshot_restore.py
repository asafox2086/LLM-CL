"""Restore one complete cumulative BudgetQV snapshot on its unchanged base.

No model construction, folding, scoring, optimizer, or device selection occurs
here. The caller supplies the fresh base and fixed interface, then freezes and
folds only after this function has checked the original parameter mask.
"""


ADAPTER_COORDINATES = 4399104
ADAPTER_PARAMETERS = 96


def restore_snapshot(model, method, snapshot, stage_record, initialization, storage):
    import torch

    modules = {f'model.layers.{layer}.self_attn.{projection}_proj'
               for layer in range(24) for projection in ('q', 'v')}
    expected = {name + suffix for name in modules for suffix in ('.A.weight', '.B.weight')}
    assert len(expected) == ADAPTER_PARAMETERS
    assert set(method.rank_plan) == modules
    assert method.rank_plan == initialization['rank_plan']
    assert len(method.layers) == len(modules) == 48
    assert {id(layer) for layer in method.layers} == {id(model.get_submodule(name)) for name in modules}
    assert model.visual.tower is model.core.vision_tower
    assert model.visual.projector is model.core.multi_modal_projector
    assert isinstance(stage_record['stage'], int) and not isinstance(stage_record['stage'], bool)
    assert 1 <= stage_record['stage'] <= 19 and stage_record['updates'] > 0
    assert stage_record['frozen_digest'] == initialization['frozen_digest']
    assert stage_record['vision_digest'] == initialization['vision_digest']
    assert initialization['layout']['trainable'] == ADAPTER_COORDINATES
    assert initialization['receipt']['adapter_parameters'] == ADAPTER_PARAMETERS
    assert initialization['receipt']['adapter_coordinates'] == ADAPTER_COORDINATES
    assert initialization['receipt']['identities_bytes_and_RNG_preserved'] is True

    parameters = dict(model.named_parameters())
    actual_adapters = {name for name in parameters if name.endswith(('.A.weight', '.B.weight'))}
    assert actual_adapters == expected
    before_fold_trainable_names = sorted(name for name, parameter in parameters.items()
                                         if parameter.requires_grad)
    assert set(before_fold_trainable_names) == expected, 'Check the original A/B mask before freezing the whole graph'
    assert isinstance(snapshot, dict) and set(snapshot) == expected
    assert len(snapshot) == ADAPTER_PARAMETERS

    shapes = {}
    dtypes = {}
    tensor_bytes = {}
    identities = {name: id(parameters[name]) for name in expected}
    for module_name in sorted(modules):
        layer = model.get_submodule(module_name)
        rank = method.rank_plan[module_name]
        assert layer.scale == 4
        assert tuple(layer.A.weight.shape) == (rank, layer.weight.shape[1])
        assert tuple(layer.B.weight.shape) == (layer.weight.shape[0], rank)
    for name in sorted(expected):
        value = snapshot[name]
        target = parameters[name]
        assert isinstance(value, torch.Tensor) and value.ndim == 2
        assert value.dtype == target.dtype == torch.float32, (name, value.dtype, target.dtype)
        assert tuple(value.shape) == tuple(target.shape), name
        assert bool(torch.isfinite(value).all()), name
        shapes[name] = list(value.shape)
        dtypes[name] = str(value.dtype)
        tensor_bytes[name] = value.numel() * value.element_size()
    coordinates = sum(snapshot[name].numel() for name in expected)
    total_bytes = sum(tensor_bytes.values())
    assert coordinates == ADAPTER_COORDINATES
    assert total_bytes == ADAPTER_COORDINATES * 4 == stage_record['adapter_bytes']
    snapshot_digest = storage.digest(snapshot)
    assert snapshot_digest == stage_record['adapter_digest']

    # Use the fixed canonical A/B names, not the current requires_grad flags, to
    # define the immutable part. A readonly caller must not expand this mask by
    # freezing A/B before checking the original training receipt.
    frozen_names = sorted(set(parameters) - expected)
    assert all(not parameters[name].requires_grad for name in frozen_names)

    def frozen_digest():
        return storage.digest({name: parameters[name] for name in frozen_names})

    def buffer_digest():
        return storage.digest({name: value for name, value in model.core.named_buffers()
                               if not name.startswith(('vision_tower.', 'multi_modal_projector.'))})

    frozen_before = frozen_digest()
    assert frozen_before == initialization['frozen_digest'] == stage_record['frozen_digest']
    vision_before = storage.digest(model.core.vision_tower)
    assert vision_before == initialization['vision_digest'] == stage_record['vision_digest']
    buffers_before = buffer_digest()
    assert buffers_before == initialization['buffer_digest']

    # Every saved tensor is the cumulative state at this stage. Replace all A/B
    # values; never add snapshots or modify an already-folded base matrix.
    with torch.no_grad():
        for name in sorted(expected):
            parameters[name].copy_(snapshot[name].to(device=parameters[name].device))
    adapter_digest = storage.digest({name: parameters[name] for name in expected})
    assert adapter_digest == snapshot_digest == stage_record['adapter_digest']
    assert all(id(parameters[name]) == identities[name] for name in expected)
    assert sorted(name for name, parameter in model.named_parameters()
                  if parameter.requires_grad) == before_fold_trainable_names
    frozen_after = frozen_digest()
    assert frozen_after == frozen_before
    assert storage.digest(model.core.vision_tower) == vision_before
    assert buffer_digest() == buffers_before
    assert storage.layout(model)['trainable'] == ADAPTER_COORDINATES

    return dict(stage=stage_record['stage'], adapter_parameters=ADAPTER_PARAMETERS,
                adapter_coordinates=coordinates, adapter_digest=adapter_digest,
                snapshot_digest=snapshot_digest, snapshot_keys=sorted(expected),
                snapshot_shapes=shapes, snapshot_dtypes=dtypes,
                snapshot_bytes=total_bytes, snapshot_tensor_bytes=tensor_bytes,
                before_fold_trainable_names=before_fold_trainable_names,
                fixed_frozen_parameter_count=len(frozen_names),
                frozen_digest=frozen_after, frozen_digest_before=frozen_before,
                frozen_digest_after=frozen_after, vision_digest=vision_before,
                buffer_digest=buffers_before,
                adapter_parameter_identities_preserved=True,
                original_trainable_mask_preserved=True,
                complete_cumulative_snapshot_copied_not_added=True)
