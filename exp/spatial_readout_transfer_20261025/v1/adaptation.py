"""Original seven-position all-r8 recipe, plus explicit GPU deployment fold.

Both fixed spatial readout arms share this known recipe. It is not the old
wide-q/v allocation and does not change final complete base-graph dimensions.
"""
from pathlib import Path

COORDINATES = 4399104
PARAMETERS = 336


def expected_modules():
    return {f'model.layers.{i}.{family}.{projection}' for i in range(24)
        for family, projections in (('self_attn', ('q_proj', 'k_proj', 'v_proj', 'o_proj')),
            ('mlp', ('gate_proj', 'up_proj', 'down_proj'))) for projection in projections}


def configure_gpu_math():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return dict(base_storage='torch.float16', outer_autocast='torch.float16',
        adapter_storage='torch.float32', adapter_forward='unchanged native SeqLinear under outer FP16 autocast',
        fold='GPU FP32 product/add then FP16; autocast/TF32 disabled',
        host_optimizer=False, offload_activations=False)


def make_all(model, assets_path=None, storage=None):
    import torch
    import study
    assert next(model.parameters()).is_cuda
    if assets_path is not None:
        assert Path(study.__file__).resolve() == (Path(assets_path) /
            'exp/osft_increment_screen_20261019/v1/study.py').resolve()
    assert study.QV == ['q_proj', 'v_proj']
    assert study.EXTRA == ['k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    method = study.TargetLoRA(model, study.QV + study.EXTRA)
    wanted = expected_modules()
    modules = {n: m for n, m in model.named_modules() if n in wanted}
    assert set(modules) == wanted and len(modules) == len(method.layers) == 168
    assert {id(m) for m in modules.values()} == {id(m) for m in method.layers}
    method.rank_plan = {name: 8 for name in modules}
    method.initialization = []
    method.host_optimizer = False
    method.offload_activations = False
    method.vision_original = None  # Native input preparation is owned by caller; AST prepare is a no-op.
    for name, layer in modules.items():
        assert layer.weight.is_cuda and layer.weight.dtype == torch.float16
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert layer.A.weight.device == layer.B.weight.device == layer.weight.device
        assert layer.scale == 4 and layer.dropout.p == .1
        assert tuple(layer.A.weight.shape) == (8, layer.weight.shape[1])
        assert tuple(layer.B.weight.shape) == (layer.weight.shape[0], 8)
        assert torch.isfinite(layer.A.weight).all() and torch.count_nonzero(layer.B.weight).item() == 0
        value = dict(module=name, rank=8, alpha=32, scale=4, dropout=.1,
            A_shape=list(layer.A.weight.shape), B_shape=list(layer.B.weight.shape),
            dtype='torch.float32', device=str(layer.weight.device), B_zero=True,
            initialization='original TargetLoRA shared q/v stream and named-fork extra positions')
        if storage is not None:
            value.update(A_digest=storage.digest({'A': layer.A.weight}), B_digest=storage.digest({'B': layer.B.weight}))
        method.initialization.append(value)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert set(trainable) == {n + suffix for n in wanted for suffix in ('.A.weight', '.B.weight')}
    assert len(trainable) == PARAMETERS and sum(p.numel() for p in trainable.values()) == COORDINATES
    return method


def fold(model, method):
    import torch
    assert set(method.rank_plan) == expected_modules() and len(method.layers) == 168
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == 'highest'
    model.eval()
    records = []
    for name, rank in method.rank_plan.items():
        layer = model.get_submodule(name)
        device = layer.weight.device
        assert rank == 8 and layer.scale == 4 and device.type == 'cuda'
        assert layer.weight.dtype == torch.float16
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        with torch.cuda.device(device), torch.no_grad(), torch.autocast('cuda', enabled=False):
            weight = (layer.weight.detach().float() + layer.scale *
                (layer.B.weight.detach().float() @ layer.A.weight.detach().float())).half()
            assert torch.isfinite(weight).all()
            linear = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0],
                bias=layer.bias is not None, device=device, dtype=torch.float16)
            linear.weight.copy_(weight)
            if layer.bias is not None:
                linear.bias.copy_(layer.bias.detach())
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, linear)
        records.append(dict(module=name, rank=8, scale=4, shape=list(weight.shape),
            dtype='torch.float16', merge_device=str(device), merge_dtype='torch.float32',
            output_device=str(device), autocast_enabled=False, matmul_allow_tf32=False))
    method.layers.clear()
    model.requires_grad_(False).eval()
    assert not any(n.endswith(('.A.weight', '.B.weight')) for n in model.core.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return records
