"""WIP: explicit native-response modulation; no Torch work on import.

Only this module's24 FP32 scalars are new weights. Each original gate/up/down
module and its state-dict path are retained, without an original-MLP stash or a
second down matrix. The sign vector is a derived FP16 ordinary buffer.
The caller owns base freezing, common Q/V adaptation, optimizer and checkpoints.
"""
import math

LAYERS = 24
HIDDEN = 896
INTERMEDIATE = 4864
EPSILON = 1e-8
BASE_COEFFICIENTS = 893675552
FINAL_COEFFICIENTS = 893675576
FINAL_WEIGHT_BYTES = 1787351200
SIGN_BUFFER_BYTES = 233472
STATIC_TRAINING_CALIBRATION_BYTES = 96
_RESPONSE_CLASS = None


def _scales(values):
    """Decode only public calibration scalars, never model tensors."""
    if isinstance(values, dict) and 'layers' in values:
        values = values['layers']
    if isinstance(values, dict):
        values = [values[i] if i in values else values[str(i)] for i in range(LAYERS)]
    assert isinstance(values, (list, tuple)) and len(values) == LAYERS
    result = []
    for i, value in enumerate(values):
        if isinstance(value, dict):
            assert value['layer'] == i
            value = value['RMS']
        assert isinstance(value, (int, float)) and not isinstance(value, bool)
        value = float(value)
        assert math.isfinite(value) and value > 0
        result.append(value)
    return result


def _type():
    global _RESPONSE_CLASS
    if _RESPONSE_CLASS is not None:
        return _RESPONSE_CLASS
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResponseMLP(nn.Module):
        def __init__(self, original, arm, scale, final):
            super().__init__()
            assert arm in ('dynamic', 'static')
            assert isinstance(original.gate_proj, nn.Linear)
            assert isinstance(original.up_proj, nn.Linear)
            assert isinstance(original.down_proj, nn.Linear)
            assert original.gate_proj.weight.shape == (INTERMEDIATE, HIDDEN)
            assert original.up_proj.weight.shape == (INTERMEDIATE, HIDDEN)
            assert original.down_proj.weight.shape == (HIDDEN, INTERMEDIATE)
            assert original.down_proj.bias is None
            device = original.gate_proj.weight.device
            assert device.type in ('cuda', 'meta')
            # A meta backbone can still have its constructor-default dtype;
            # strict assign supplies the saved mixed-dtype state afterward.
            assert all(p.device == device and (device.type == 'meta' or p.dtype == torch.float16)
                       for p in original.parameters())
            # Retain the exact original modules, parameter objects and names.
            # Do not freeze them here: common Q/V training policy is external.
            self.gate_proj = original.gate_proj
            self.up_proj = original.up_proj
            self.down_proj = original.down_proj
            self.act_fn = original.act_fn
            self.beta = nn.Parameter(torch.zeros((), dtype=torch.float32, device=device),
                                     requires_grad=not final)
            sign = torch.ones(INTERMEDIATE, dtype=torch.float16, device=device)
            sign[1::2] = -1
            self.register_buffer('response_sign', sign, persistent=True)
            self.arm = arm
            self.final_format = bool(final)
            if arm == 'static' and not final:
                assert scale is not None
                self.register_buffer('calibration_scale',
                                     torch.tensor(scale, dtype=torch.float32, device=device),
                                     persistent=True)
            else:
                assert scale is None

        def forward(self, x):
            # Original Qwen2MLP h, without any cast or modification of its path.
            gate = self.gate_proj(x)
            phi = self.act_fn(gate) * self.up_proj(x)
            h = self.down_proj(phi)
            # Functional linear reuses D. It does not store D*diag(sign), clone
            # D, or recursively invoke a hooked down module.
            contrast = F.linear(phi * self.response_sign.to(dtype=phi.dtype),
                                self.down_proj.weight, self.down_proj.bias)
            if self.arm == 'dynamic':
                # Normal autograd connects d and contrast to the original x.
                # FP32 here is the fixed score arithmetic, not base precision.
                g32 = gate.float()
                even = g32[..., 0::2].square().mean(dim=-1)
                odd = g32[..., 1::2].square().mean(dim=-1)
                score = (even - odd) / (even + odd + EPSILON)
            elif self.final_format:
                score = self.beta.new_ones(())
            else:
                score = self.calibration_scale
            assert self.beta.dtype == torch.float32
            assert torch.isfinite(self.beta).all() and torch.isfinite(score).all()
            assert torch.isfinite(h).all() and torch.isfinite(contrast).all()
            coefficient = self.beta * score
            # No beta==0 shortcut: the first beta derivative must exist.
            # FP32 scalar/reduction path avoids casting the learned scalar to
            # FP16 before its gradient. The model-visible result retains h.dtype.
            result = (h.float() + coefficient.unsqueeze(-1) * contrast.float()).to(h.dtype)
            assert torch.isfinite(result).all()
            return result

    _RESPONSE_CLASS = ResponseMLP
    return ResponseMLP


def _modules(model):
    layers = model.model.layers
    assert len(layers) == LAYERS
    modules = [layer.mlp for layer in layers]
    cls = _type()
    assert all(isinstance(module, cls) for module in modules)
    return modules


def install(model, arm, scales=None, device=None, *, final=False):
    """Install24 responses, preserving all original matrix state-dict keys.

    Training: static requires24 diagnostic RMS values; dynamic ignores an
    optional same diagnostic argument and stores no calibration. New beta=0.
    Final reload: pass final=True, scales=None. Static beta then denotes the
    effective gamma=trained_beta*c with implicit scale1; dynamic beta keeps its
    training meaning. Strict-assign complete state and every saved named buffer
    before using a meta-constructed graph. No data-dependent work runs on meta.

    Install AFTER a common adapter initializer that freezes all parameters, or
    explicitly restore the24 beta training flags afterward. This function does
    not alter existing base/adapter requires_grad flags.
    """
    import torch
    assert arm in ('dynamic', 'static') and isinstance(final, bool)
    if final:
        assert scales is None, 'Final packages have no calibration payload'
    values = _scales(scales) if arm == 'static' and not final else None
    layers = model.model.layers
    assert len(layers) == LAYERS
    actual_device = layers[0].mlp.gate_proj.weight.device
    if device is not None:
        requested = torch.device(device)
        if requested.type == 'cuda' and requested.index is None:
            requested = torch.device('cuda', torch.cuda.current_device())
        assert actual_device == requested
    assert actual_device.type in ('cuda', 'meta')
    cls = _type()
    assert all(not isinstance(layer.mlp, cls) for layer in layers), 'Already installed'
    for i, layer in enumerate(layers):
        assert layer.mlp.gate_proj.weight.device == actual_device
        layer.mlp = cls(layer.mlp, arm, None if values is None else values[i], final)
    return metadata(model)


def metadata(model):
    """Structural serialization contract; contains no calibration values."""
    modules = _modules(model)
    arms = {module.arm for module in modules}
    formats = {module.final_format for module in modules}
    assert len(arms) == len(formats) == 1
    arm, final = next(iter(arms)), next(iter(formats))
    for module in modules:
        assert module.beta.shape == () and str(module.beta.dtype) == 'torch.float32'
        assert module.response_sign.shape == (INTERMEDIATE,)
        assert str(module.response_sign.dtype) == 'torch.float16'
        assert hasattr(module, 'calibration_scale') == (arm == 'static' and not final)
    return dict(arm=arm, final=final,
                format=('static-effective-gamma-v1' if arm == 'static' and final
                        else 'static-beta-calibration-v1' if arm == 'static'
                        else 'dynamic-beta-v1'),
                layers=LAYERS, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE,
                beta_parameters=LAYERS, beta_dtype='torch.float32',
                beta_semantics='effective_gamma' if arm == 'static' and final else 'beta',
                dynamic_epsilon=EPSILON, gate_score_arithmetic='GPU FP32',
                sign_buffer_name='response_sign', sign_dtype='torch.float16',
                sign_definition='even channels +1, odd channels -1',
                derived_sign_buffer_bytes=SIGN_BUFFER_BYTES,
                calibration_buffer_name='calibration_scale' if arm == 'static' and not final else None,
                calibration_buffer_bytes=STATIC_TRAINING_CALIBRATION_BYTES if arm == 'static' and not final else 0,
                final_coefficients=FINAL_COEFFICIENTS, final_weight_bytes=FINAL_WEIGHT_BYTES,
                both_arms_explicit_branch=True, down_matrix_copied=False,
                original_gate_up_down_state_paths_preserved=True)


def compile_final(model):
    """Finalize scalar representation only; never fold the branch into dense D.

    Static: beta.copy_(beta*c) in GPU FP32, remove24 calibration buffers and
    thereafter execute scale1. Dynamic: freeze beta unchanged. The explicit
    signed-phi Down branch remains in both models. This one-way conversion must
    precede export. Use metadata(...).format to reconstruct the matching state
    and buffer keys; a final static loader must not create calibration_scale.
    """
    import torch
    modules = _modules(model)
    assert all(not module.final_format for module in modules), 'Already final'
    arm = modules[0].arm
    assert all(module.arm == arm for module in modules)
    device = modules[0].beta.device
    assert device.type == 'cuda'
    for module in modules:
        assert module.beta.device == device and module.beta.dtype == torch.float32
        assert torch.isfinite(module.beta).all()
        if arm == 'static':
            scale = module.calibration_scale
            assert scale.device == device and scale.dtype == torch.float32
            assert torch.isfinite(scale).all() and scale > 0
            with torch.no_grad(), torch.autocast('cuda', enabled=False):
                effective = module.beta.detach().float() * scale.float()
                assert torch.isfinite(effective).all()
                module.beta.copy_(effective)
            del module.calibration_scale
        module.beta.grad = None
        module.beta.requires_grad_(False)
        module.final_format = True
    result = metadata(model)
    assert result['calibration_buffer_bytes'] == 0
    return dict(result, compiled_on_device=str(device), compile_arithmetic='GPU FP32 scalar only',
                no_dense_D_fold=True, final_saved_named_buffers_include_sign=True,
                static_calibration_absorbed=arm == 'static')
