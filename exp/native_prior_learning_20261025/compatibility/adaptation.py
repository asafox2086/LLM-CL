"""Fixed native q/v-r8 adaptation and GPU FP32 deployment fold.

No model work on import. Original SeqLinear is used under the caller's outer
FP16 autocast; FP32 parameter storage does not imply FP32 adapter compute.
The caller owns the original training AST, GPU AdamW and per-stage reset.
"""
import hashlib

PREFIX = 'model.text_model.layers.'
COORDINATES = 819200
PARAMETERS = 128
ATTENTION = {'vision_config': 'eager', 'text_config': 'sdpa'}


def configure_gpu_math():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return dict(base_storage='torch.float16', outer_autocast='torch.float16',
        adapter_storage='torch.float32', adapter_forward='original SeqLinear under outer FP16 autocast',
        optimizer='GPU AdamW; weight_decay0; per-stage reset; caller original schedule',
        fold='GPU FP32 product/add with autocast and TF32 disabled, then FP16',
        matmul_allow_tf32=False, cudnn_allow_tf32=False,
        attention=dict(ATTENTION), vision_backend_native_implementation_exception=True)


def expected_modules():
    return {f'{PREFIX}{i}.self_attn.{projection}' for i in range(32) for projection in ('q_proj', 'v_proj')}


def make_qv(model, storage=None):
    import torch
    from methods import SeqLinear
    assert next(model.parameters()).is_cuda
    assert len(model.model.text_model.layers) == 32
    assert model.model.vision_model.config._attn_implementation == 'eager'
    assert model.model.text_model.config._attn_implementation == 'sdpa'
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    wanted = expected_modules()
    candidates = [(name, layer) for name, layer in model.named_modules()
                  if name in wanted and isinstance(layer, torch.nn.Linear)]
    assert len(candidates) == 64 and {name for name, _ in candidates} == wanted
    layers, initialization, rank_plan = [], [], {}
    for name, base in candidates:
        assert base.weight.is_cuda and base.weight.dtype == torch.float16
        seed = int(hashlib.sha256(('42:native-reuse:' + name).encode()).hexdigest()[:8], 16)
        with torch.random.fork_rng(devices=[base.weight.device.index]):
            torch.manual_seed(seed)
            layer = SeqLinear(base, rank=8, alpha=32)
        parent, child = name.rsplit('.', 1)
        setattr(model.get_submodule(parent), child, layer)
        assert layer.scale == 4 and layer.dropout.p == .1
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert layer.A.weight.device == layer.B.weight.device == base.weight.device
        assert torch.isfinite(layer.A.weight).all() and torch.count_nonzero(layer.B.weight).item() == 0
        record = dict(module=name, rank=8, alpha=32, scale=4, dropout=.1,
            initialization_seed=seed, A_shape=list(layer.A.weight.shape),
            B_shape=list(layer.B.weight.shape), dtype='torch.float32', B_zero=True,
            device=str(base.weight.device))
        if storage is not None:
            record.update(A_digest=storage.digest({'A': layer.A.weight}),
                          B_digest=storage.digest({'B': layer.B.weight}))
        initialization.append(record)
        layers.append(layer)
        rank_plan[name] = 8
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert set(trainable) == {name + suffix for name in wanted for suffix in ('.A.weight', '.B.weight')}
    assert len(trainable) == PARAMETERS and sum(p.numel() for p in trainable.values()) == COORDINATES

    class NativeQV:
        name = 'SeqFT_LoRA'
        learning_rate = 1e-4
        host_optimizer = False
        offload_activations = False
        vision_original = None

        def set_instruction(self, task):
            pass

        def begin_stage(self, stage):
            pass

    method = NativeQV()
    method.model, method.layers, method.rank_plan = model, layers, rank_plan
    method.initialization = initialization
    method.original_revision = 'original SeqLinear and old native named-seed Q; fixed new FP16/GPUAdamW condition'
    return method


def fold(model, method):
    import torch
    assert set(method.rank_plan) == expected_modules() and len(method.layers) == 64
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.get_float32_matmul_precision() == 'highest'
    model.eval()
    records = []
    for name, rank in method.rank_plan.items():
        layer = model.get_submodule(name)
        assert rank == 8 and layer.scale == 4
        device = layer.weight.device
        assert device.type == 'cuda' and layer.weight.dtype == torch.float16
        assert layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert layer.A.weight.device == layer.B.weight.device == device
        with torch.cuda.device(device), torch.no_grad(), torch.autocast('cuda', enabled=False):
            weight = (layer.weight.detach().float() + layer.scale *
                      (layer.B.weight.detach().float() @ layer.A.weight.detach().float())).half()
            assert torch.isfinite(weight).all()
            linear = torch.nn.Linear(layer.weight.shape[1], layer.weight.shape[0],
                bias=layer.bias is not None, dtype=torch.float16, device=device)
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
    assert not any(name.endswith(('.A.weight', '.B.weight')) for name in model.state_dict())
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())
    return records
