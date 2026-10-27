"""Task-direction q/v-r8 reuse; no Torch import or execution on import.

Native SeqLinear forward, alpha32/scale4/dropout.1 and FP32 A/B are retained.
Every stage merges the previous update into the FP16 base, then reuses the same
96 Parameter objects. The active coordinate budget is fixed; cumulative delta
rank is not constrained to eight. No original LoRA-GA equivalence is claimed.
"""
import hashlib
import importlib.util
import math
from pathlib import Path

ARMS = ('aligned', 'random')
COORDINATES = 540672
PARAMETERS = 96
RANK = 8
ALPHA = 32
SCALE = 4
DROPOUT = .1
ROW_NORM = 1 / math.sqrt(3)


def configure_gpu_math():
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')


def rank_plan(layers=24):
    return {f'model.layers.{i}.self_attn.{part}_proj': RANK
            for i in range(layers) for part in ('q', 'v')}


def _check_layer(layer):
    import torch
    assert layer.weight.is_cuda and layer.weight.dtype == torch.float16
    assert layer.A.weight.is_cuda and layer.B.weight.is_cuda
    assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
    assert layer.weight.device == layer.A.weight.device == layer.B.weight.device
    assert layer.A.weight.shape == (RANK, layer.weight.shape[1])
    assert layer.B.weight.shape == (layer.weight.shape[0], RANK)
    assert layer.scale == SCALE and layer.dropout.p == DROPOUT


def metadata(model, plan=None):
    import torch
    plan = rank_plan() if plan is None else dict(plan)
    arm, deployed = model._direction_arm, bool(model._direction_deployed)
    assert arm in ARMS and plan == rank_plan()
    expected = {name+suffix for name in plan for suffix in ('.A.weight','.B.weight')}
    actual = {name:p for name,p in model.named_parameters() if p.requires_grad}
    if not deployed:
        for name in plan: _check_layer(model.get_submodule(name))
        structural = {name:p for name,p in model.named_parameters() if name in expected}
        assert set(structural) == expected and len(structural) == PARAMETERS
        assert sum(p.numel() for p in structural.values()) == COORDINATES
        assert all(p.dtype == torch.float32 and p.is_cuda for p in structural.values())
        # End-of-training receipts legitimately freeze the still-explicit AB
        # graph. A nonempty training mask must contain all and only native AB.
        assert not actual or set(actual) == expected
    else:
        assert not any(name.endswith(('.A.weight','.B.weight')) for name,_ in model.named_parameters())
        assert not any(p.requires_grad for p in model.parameters())
    return dict(arm=arm,deployed=deployed,rank_plan=plan,rank=RANK,alpha=ALPHA,
        scale=SCALE,dropout=DROPOUT,A_B_dtype='torch.float32',base_dtype='torch.float16',
        A_row_norm=ROW_NORM,explicit_AB=not deployed,
        structural_AB_coordinates=0 if deployed else COORDINATES,
        structural_AB_parameter_tensors=0 if deployed else PARAMETERS,
        active_training_coordinates=sum(p.numel() for p in actual.values()),
        active_training_parameter_tensors=len(actual),
        model_parameter_coefficients=sum(p.numel() for p in model.parameters()),
        model_parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters()),
        parameter_counting='unique Parameter objects; ordinary buffers are separately caller-accounted',
        extra_router_parameters=0,cumulative_delta_rank_is_not_fixed_r8=True)


def make_method(model, asset, storage, arm, inherited_source=None):
    """Install the exact original shared-r8 layers; reset_stage sets task A."""
    assert arm in ARMS
    source = (Path(inherited_source) if inherited_source is not None else
              Path(asset)/'exp/plain_r8_gpu_control_20261025/v1/control_adaptation.py')
    spec = importlib.util.spec_from_file_location('task_direction_inherited_r8',source)
    old = importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    method = old.make_plain(model,asset,storage)
    assert method.rank_plan == rank_plan() and len(method.layers) == 48
    method._direction_storage = storage
    method.direction_arm = arm
    method.direction_last_fold_stage = 15
    method.direction_last_reset_stage = 15
    method.host_optimizer = False
    method.offload_activations = False
    method.vision_original = None
    model._direction_arm = arm
    model._direction_deployed = False
    method.direction_metadata = metadata(model,method.rank_plan)
    return method


def install_meta(model, arm, *, deployed, plan=None):
    """State-schema construction only; strict state assign is caller-owned."""
    from methods import SeqLinear
    plan = rank_plan() if plan is None else dict(plan)
    assert arm in ARMS and plan == rank_plan()
    assert all(model.get_submodule(name).weight.is_meta for name in plan)
    if not deployed:
        for name in plan:
            base = model.get_submodule(name)
            layer = SeqLinear(base,rank=RANK,alpha=ALPHA)
            parent,child = name.rsplit('.',1)
            setattr(model.get_submodule(parent),child,layer)
    model._direction_arm = arm
    model._direction_deployed = bool(deployed)
    # Meta state may have temporary requires_grad/dtypes before strict assign.
    return dict(arm=arm,deployed=bool(deployed),rank_plan=plan,rank=RANK,
                alpha=ALPHA,scale=SCALE,dropout=DROPOUT,explicit_AB=not deployed)


def _merge_weight(layer, storage):
    """GPU FP32 merge followed by original base dtype; no random allocation."""
    import torch
    _check_layer(layer)
    assert not torch.backends.cuda.matmul.allow_tf32
    with torch.no_grad(),torch.autocast('cuda',enabled=False):
        a,b = layer.A.weight.detach(),layer.B.weight.detach()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert torch.isfinite(layer.weight).all()
        a_digest = storage.digest({'A':a});b_digest = storage.digest({'B':b})
        before_digest = storage.digest({'weight':layer.weight})
        merged = layer.weight.detach().float()+SCALE*(b.float()@a.float())
        rounded = merged.to(layer.weight.dtype)
        assert torch.isfinite(merged).all() and torch.isfinite(rounded).all()
        error = rounded.float()-merged
        receipt = dict(A_digest=a_digest,B_digest=b_digest,base_before_digest=before_digest,
            merged_base_digest=storage.digest({'weight':rounded}),
            rounding_max_abs=float(error.abs().max().item()),
            rounding_rms=float(error.square().mean().sqrt().item()),
            update_max_abs=float((merged-layer.weight.detach().float()).abs().max().item()),
            merge_dtype='torch.float32',output_dtype=str(rounded.dtype),device=str(rounded.device),
            autocast_enabled=False,matmul_allow_tf32=False,scale=SCALE)
    return rounded,receipt


def fold_stage(model, method, stage):
    """Before current-task calibration: merge old BA and set B=0 in place."""
    import torch
    assert not model._direction_deployed and method.direction_arm == model._direction_arm
    assert type(stage) is int and stage == 16
    assert stage == method.direction_last_fold_stage+1
    assert method.direction_last_reset_stage == stage-1
    storage = method._direction_storage
    identities = {name:id(p) for name,p in model.named_parameters() if p.requires_grad}
    records = []
    for name in method.rank_plan:
        layer = model.get_submodule(name)
        weight,record = _merge_weight(layer,storage)
        with torch.no_grad():
            layer.weight.copy_(weight)
            layer.B.weight.zero_()
            layer.A.weight.grad = None
            layer.B.weight.grad = None
        records.append(dict(record,module=name,B_zero_after=True,
                            B_after_digest=storage.digest({'B':layer.B.weight})))
    assert identities == {name:id(p) for name,p in model.named_parameters() if p.requires_grad}
    method.direction_last_fold_stage = stage
    return dict(stage=stage,folded_modules=records,parameter_objects_preserved=True,
        base_function_reference='current stage FP16 folded base; not bitwise pre-fold explicit graph',
        metadata=metadata(model,method.rank_plan))


def reset_stage(model, method, gradients, stage):
    """Both arms perform GPU SVD; random uses a private per-layer CUDA RNG."""
    import torch
    assert not model._direction_deployed and method.direction_arm == model._direction_arm
    assert method.direction_last_fold_stage == stage
    assert stage == method.direction_last_reset_stage+1
    assert set(gradients) == set(method.rank_plan)
    assert not torch.backends.cuda.matmul.allow_tf32
    storage = method._direction_storage
    records = []
    for name in method.rank_plan:
        layer = model.get_submodule(name);_check_layer(layer)
        gradient = gradients[name]
        assert isinstance(gradient,torch.Tensor) and gradient.is_cuda
        assert gradient.device == layer.weight.device and gradient.shape == layer.weight.shape
        assert gradient.dtype == torch.float32 and not gradient.requires_grad
        assert torch.count_nonzero(layer.B.weight).item() == 0
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            assert torch.isfinite(gradient).all()
            # Required common decomposition even in the randomized arm.
            u,s,vh = torch.linalg.svd(gradient,full_matrices=False)
            assert min(gradient.shape) >= RANK and torch.isfinite(s).all()
            assert torch.isfinite(vh).all()
            seed = int(hashlib.sha256(f'42:task-direction:{stage}:{name}'.encode()).hexdigest()[:16],16)
            if method.direction_arm == 'aligned':
                basis = vh[:RANK]
            else:
                generator = torch.Generator(device=gradient.device)
                generator.manual_seed(seed)
                gaussian = torch.randn((gradient.shape[1],RANK),device=gradient.device,
                                       dtype=torch.float32,generator=generator)
                q,_ = torch.linalg.qr(gaussian,mode='reduced')
                basis = q.T
            initialized = (ROW_NORM*basis).contiguous()
            assert initialized.shape == layer.A.weight.shape and torch.isfinite(initialized).all()
            layer.A.weight.copy_(initialized)
            layer.B.weight.zero_()
            layer.A.weight.grad = None;layer.B.weight.grad = None
            squared = s.square();energy = squared.sum();top_energy = squared[:RANK].sum()
            gram = basis@basis.T
            records.append(dict(module=name,gradient_shape=list(gradient.shape),
                gradient_dtype=str(gradient.dtype),gradient_digest=storage.digest({'G':gradient}),
                gradient_nonzero=bool(energy.item()>0),singular_values_top_r=s[:RANK].tolist(),
                total_squared_singular_energy=float(energy.item()),
                top_r_squared_singular_energy=float(top_energy.item()),
                top_r_energy_fraction=float((top_energy/energy).item()) if energy.item()>0 else 0.0,
                A_digest=storage.digest({'A':layer.A.weight}),B_digest=storage.digest({'B':layer.B.weight}),
                B_zero=True,A_row_norm_target=ROW_NORM,
                A_row_norm_max_error=float((initialized.norm(dim=1)-ROW_NORM).abs().max().item()),
                basis_orthogonality_max_error=float((gram-torch.eye(RANK,device=gram.device)).abs().max().item()),
                private_random_seed=seed if method.direction_arm == 'random' else None,
                direction='task_gradient_top_right_singular_vectors' if method.direction_arm == 'aligned' else 'random_orthogonal_rows',
                device=str(gradient.device),svd_dtype='torch.float32',svd_full_matrices=False,
                main_RNG_used_for_random_direction=False))
            # Do not retain full SVD/G or a hidden reference adapter on method.
            del u,s,vh,basis,initialized,gram,squared,energy,top_energy
            if method.direction_arm == 'random':del gaussian,q,generator
    method.direction_last_reset_stage = stage
    return dict(stage=stage,arm=method.direction_arm,modules=records,
        new_branch_zero_for_every_dropout_mask=True,
        first_step_A_gradient_zero=True,cumulative_delta_rank_is_not_fixed_r8=True,
        metadata=metadata(model,method.rank_plan))


def deploy(model, method):
    """Final GPU FP32 fold, replacing all AB layers with frozen dense linears."""
    import torch
    assert not model._direction_deployed and method.direction_last_reset_stage == 16
    storage = method._direction_storage;records = []
    model.eval()
    for name in method.rank_plan:
        layer = model.get_submodule(name)
        weight,record = _merge_weight(layer,storage)
        with torch.no_grad(),torch.autocast('cuda',enabled=False):
            with torch.device('meta'):
                replacement = torch.nn.Linear(layer.weight.shape[1],layer.weight.shape[0],
                    bias=layer.bias is not None,dtype=layer.weight.dtype)
            replacement.weight = torch.nn.Parameter(weight,requires_grad=False)
            if layer.bias is not None:replacement.bias = layer.bias
        parent,child = name.rsplit('.',1)
        setattr(model.get_submodule(parent),child,replacement)
        records.append(dict(record,module=name,rank=RANK))
    method.layers.clear()
    model._direction_deployed = True
    model.requires_grad_(False).eval()
    return dict(arm=method.direction_arm,folded_modules=records,explicit_bank=False,
                metadata=metadata(model,method.rank_plan))


fold = deploy
