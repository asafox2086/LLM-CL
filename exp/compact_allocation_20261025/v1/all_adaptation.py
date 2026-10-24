"""WIP: inherited seven-location LoRA and cumulative snapshot restoration.

The allocation is the existing TargetLoRA recipe, not a new rank search.
Deployment export/fold/load are imported unchanged from the sealed compact-first
adaptation.py, which must be included in READY.sources by the caller.
"""
from pathlib import Path
import hashlib

from adaptation import (install_interface_preserving_adapters, fold,
                        export_package, load_export)

ADAPTER_COORDINATES = 4399104
ADAPTER_PARAMETERS = 336
TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj',
           'gate_proj', 'up_proj', 'down_proj')


def expected_modules():
    return {f'model.layers.{i}.{family}.{projection}'
            for i in range(24)
            for family, projections in (
                ('self_attn', TARGETS[:4]), ('mlp', TARGETS[4:]))
            for projection in projections}


def make_all(model, assets_path, storage):
    """Use the original TargetLoRA constructor and its exact RNG policy."""
    import torch
    import study
    assert Path(study.__file__).resolve() == (
        Path(assets_path) / 'exp/osft_increment_screen_20261019/v1/study.py').resolve()
    assert study.QV == ['q_proj', 'v_proj']
    assert study.EXTRA == ['k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    method = study.TargetLoRA(model, study.QV + study.EXTRA)
    wanted = expected_modules()
    modules = {name: module for name, module in model.named_modules() if name in wanted}
    assert set(modules) == wanted and len(modules) == len(method.layers) == 168
    assert {id(layer) for layer in method.layers} == {id(layer) for layer in modules.values()}
    method.rank_plan = {name: 8 for name in modules}
    method.initialization = []
    for name, layer in modules.items():
        assert layer.scale == 4 and layer.dropout.p == .1
        assert layer.A.weight.shape == (8, layer.weight.shape[1])
        assert layer.B.weight.shape == (layer.weight.shape[0], 8)
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert torch.isfinite(layer.A.weight).all() and torch.isfinite(layer.B.weight).all()
        assert torch.count_nonzero(layer.B.weight).item() == 0
        projection = name.rsplit('.', 1)[-1]
        extra_seed = (None if projection in study.QV else int(hashlib.sha256(
            ('42:osft-extra-init:' + name).encode()).hexdigest()[:8], 16))
        method.initialization.append(dict(module=name, rank=8, alpha=32, scale=4,
            dropout=.1, weight_shape=list(layer.weight.shape),
            A_shape=list(layer.A.weight.shape), B_shape=list(layer.B.weight.shape),
            dtype='torch.float32', B_zero=True, extra_initialization_seed=extra_seed,
            A_digest=storage.digest({'A': layer.A.weight}),
            B_digest=storage.digest({'B': layer.B.weight})))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    expected = {n + suffix for n in wanted for suffix in ('.A.weight', '.B.weight')}
    assert set(trainable) == expected and len(trainable) == ADAPTER_PARAMETERS
    assert sum(p.numel() for p in trainable.values()) == ADAPTER_COORDINATES
    return method


def restore_snapshot(model, method, snapshot, stage_record, initialization, storage):
    """Copy one full cumulative all-r8 snapshot under the original mask."""
    import torch
    modules = expected_modules()
    expected = {name + suffix for name in modules for suffix in ('.A.weight', '.B.weight')}
    assert set(method.rank_plan) == modules
    assert method.rank_plan == initialization['rank_plan']
    assert all(rank == 8 for rank in method.rank_plan.values())
    assert len(method.layers) == len(modules) == 168
    assert {id(layer) for layer in method.layers} == {id(model.get_submodule(n)) for n in modules}
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
    assert {n for n in parameters if n.endswith(('.A.weight', '.B.weight'))} == expected
    before_fold_trainable_names = sorted(n for n, p in parameters.items() if p.requires_grad)
    assert set(before_fold_trainable_names) == expected
    assert isinstance(snapshot, dict) and set(snapshot) == expected
    assert len(snapshot) == ADAPTER_PARAMETERS
    identities = {n: id(parameters[n]) for n in expected}
    shapes, dtypes, tensor_bytes = {}, {}, {}
    for name in sorted(expected):
        value, target = snapshot[name], parameters[name]
        assert isinstance(value, torch.Tensor) and value.ndim == 2
        assert value.dtype == target.dtype == torch.float32
        assert tuple(value.shape) == tuple(target.shape) and torch.isfinite(value).all()
        shapes[name] = list(value.shape)
        dtypes[name] = str(value.dtype)
        tensor_bytes[name] = value.numel() * value.element_size()
    coordinates = sum(value.numel() for value in snapshot.values())
    total_bytes = sum(tensor_bytes.values())
    assert coordinates == ADAPTER_COORDINATES
    assert total_bytes == ADAPTER_COORDINATES * 4 == stage_record['adapter_bytes']
    snapshot_digest = storage.digest(snapshot)
    assert snapshot_digest == stage_record['adapter_digest']
    frozen_names = sorted(set(parameters) - expected)
    assert all(not parameters[n].requires_grad for n in frozen_names)

    def frozen_digest():
        return storage.digest({n: parameters[n] for n in frozen_names})

    def buffer_digest():
        return storage.digest({n: v for n, v in model.core.named_buffers()
                               if not n.startswith(('vision_tower.', 'multi_modal_projector.'))})

    frozen_before = frozen_digest()
    assert frozen_before == initialization['frozen_digest'] == stage_record['frozen_digest']
    vision_before = storage.digest(model.core.vision_tower)
    buffers_before = buffer_digest()
    assert vision_before == initialization['vision_digest'] == stage_record['vision_digest']
    assert buffers_before == initialization['buffer_digest']
    with torch.no_grad():
        for name in sorted(expected):
            parameters[name].copy_(snapshot[name].to(device=parameters[name].device))
    adapter_digest = storage.digest({n: parameters[n] for n in expected})
    assert adapter_digest == snapshot_digest
    assert all(id(parameters[n]) == identities[n] for n in expected)
    assert sorted(n for n, p in model.named_parameters() if p.requires_grad) == before_fold_trainable_names
    frozen_after = frozen_digest()
    assert frozen_after == frozen_before and storage.digest(model.core.vision_tower) == vision_before
    assert buffer_digest() == buffers_before
    assert storage.layout(model)['trainable'] == ADAPTER_COORDINATES
    return dict(stage=stage_record['stage'], adapter_parameters=ADAPTER_PARAMETERS,
        adapter_coordinates=coordinates, adapter_digest=adapter_digest,
        snapshot_digest=snapshot_digest, snapshot_keys=sorted(expected),
        snapshot_shapes=shapes, snapshot_dtypes=dtypes, snapshot_bytes=total_bytes,
        snapshot_tensor_bytes=tensor_bytes, before_fold_trainable_names=before_fold_trainable_names,
        fixed_frozen_parameter_count=len(frozen_names), frozen_digest=frozen_after,
        frozen_digest_before=frozen_before, frozen_digest_after=frozen_after,
        vision_digest=vision_before, buffer_digest=buffers_before,
        adapter_parameter_identities_preserved=True, original_trainable_mask_preserved=True,
        complete_cumulative_snapshot_copied_not_added=True)
