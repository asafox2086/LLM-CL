"""Frozen 8x8 spatial readout with original OneVision crop/AnyRes semantics.

No Torch import, model construction or computation occurs on module import.
Packing follows Transformers4.46.3 LlavaOnevision.pack_image_features, Apache2:
https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llava_onevision/modeling_llava_onevision.py
Only the feature lattice is8x8. Image selection/crop size remains384, not512.
The official Idefics3 scale4 shuffle is a rearrangement before one compiled
12288->896 affine. No native960 readout or fitted960->896 factor is retained.
"""
from types import MethodType, SimpleNamespace
import math

CROP_SIZE = 384
STUDENT_SIZE = 512
PATCH_SIZE = 16
PATCH_GRID = 32
SCALE_FACTOR = 4
OUTPUT_GRID = 8
INPUT_WIDTH = 12288
OUTPUT_WIDTH = 896
STUDENT_COEFFICIENTS = 86433024
COMPILED_COEFFICIENTS = 11010944
COMPLETE_COEFFICIENTS = 591534976
_CLASSES = None


def metadata():
    return dict(original_crop_size=CROP_SIZE, student_size=STUDENT_SIZE,
                student_patch_size=PATCH_SIZE, student_patch_grid=PATCH_GRID,
                scale_factor=SCALE_FACTOR, output_grid=OUTPUT_GRID,
                compiled_input_width=INPUT_WIDTH, compiled_output_width=OUTPUT_WIDTH,
                compiled_bias=True, student_coefficients=STUDENT_COEFFICIENTS,
                compiled_coefficients=COMPILED_COEFFICIENTS,
                complete_design_coefficients=COMPLETE_COEFFICIENTS,
                original_normalization_and_crop_count_preserved=True,
                original_crop_to_student='FP32 bicubic; align_corners=False; antialias=False; then FP16',
                native_shuffle_order='official Idefics3 scale4 reshape/permute order',
                feature_packing='8x8 lattice; original384 AnyRes grid/unpad/max9/newline',
                no_original_tower_projector_or_unfolded_readout_retained=True)


def build_student(config, device):
    """Allocate an uninitialized GPU or meta tower for strict asset loading.

    Suppress discarded random initialization. Native deterministic nonpersistent
    position-index buffers are constructed normally on the requested GPU.
    Caller must strict-load the bound full student state before any forward.
    A meta caller must also assign every nonpersistent buffer before execution.
    """
    import torch
    from transformers.modeling_utils import no_init_weights
    from transformers.models.idefics3.configuration_idefics3 import Idefics3VisionConfig
    from transformers.models.idefics3.modeling_idefics3 import Idefics3VisionTransformer
    config = config.to_dict() if hasattr(config, 'to_dict') else dict(config)
    cfg = Idefics3VisionConfig(**config)
    assert (cfg.image_size, cfg.patch_size, cfg.hidden_size) == (512, 16, 768)
    cfg._attn_implementation = 'eager'
    device = torch.device(device)
    assert device.type in ('cuda', 'meta'), 'No real CPU model construction'
    with torch.device(device), no_init_weights():
        tower = Idefics3VisionTransformer(cfg)
    # Allocate the final dtype without converting uninitialized float values.
    for parameter in tower.parameters():
        parameter.data = torch.empty_like(parameter, dtype=torch.float16, device=device)
    assert all(p.device.type == device.type and p.dtype == torch.float16 for p in tower.parameters())
    tower.requires_grad_(False).eval()
    return tower


def patch_features(tower, pixels):
    """Return official shuffled64x12288 features for each unchanged384 crop.

    The original normalized crop is resized exactly as the old small-tower
    condition. No interpolation of the resulting32x32 feature lattice occurs.
    """
    import torch
    from torch.nn import functional as F
    assert pixels.ndim == 4 and tuple(pixels.shape[1:]) == (3, CROP_SIZE, CROP_SIZE)
    assert pixels.is_cuda and next(tower.parameters()).is_cuda
    assert pixels.device == next(tower.parameters()).device
    assert next(tower.parameters()).dtype == torch.float16
    resized = F.interpolate(pixels.float(), size=(STUDENT_SIZE, STUDENT_SIZE),
                            mode='bicubic', align_corners=False, antialias=False).to(torch.float16)
    feature = tower(resized, output_hidden_states=False, return_dict=True).last_hidden_state
    assert tuple(feature.shape[1:]) == (1024, 768)
    # Exact Idefics3Connector.pixel_shuffle sequence, including its final
    # transpose. A different local patch/channel ordering changes W's function.
    batch, _, channels = feature.shape
    feature = feature.view(batch, PATCH_GRID, PATCH_GRID, channels)
    feature = feature.view(batch, PATCH_GRID, OUTPUT_GRID, channels * SCALE_FACTOR)
    feature = feature.permute(0, 2, 1, 3)
    feature = feature.reshape(batch, OUTPUT_GRID, OUTPUT_GRID, channels * SCALE_FACTOR ** 2)
    feature = feature.permute(0, 2, 1, 3)
    return feature.reshape(batch, OUTPUT_GRID ** 2, INPUT_WIDTH).contiguous()


def _pack_image_features(core, image_features, image_sizes, image_newline=None,
                         vision_aspect_ratio='anyres_max_9'):
    """Instance-bound native signature; original crop grid plus8x8 readout.

    Uses the installed original resolution-selection and unpad helpers. The
    original config remains384/14; no false patch-size metadata is introduced.
    """
    import torch
    from torch.nn import functional as F
    from transformers.models.llava_onevision.modeling_llava_onevision import (
        get_anyres_image_grid_shape, unpad_image)
    assert core.config.vision_config.image_size == CROP_SIZE
    assert core.config.vision_config.patch_size == 14
    assert vision_aspect_ratio == 'anyres_max_9'
    assert len(image_features) == len(image_sizes) and len(image_features) > 0
    packed, lengths = [], []
    for image_index, crops in enumerate(image_features):
        assert crops.ndim == 3 and tuple(crops.shape[1:]) == (OUTPUT_GRID ** 2, OUTPUT_WIDTH)
        if crops.shape[0] > 1:
            base, tiled = crops[0], crops[1:]
            grid_h, grid_w = get_anyres_image_grid_shape(
                image_sizes[image_index], core.config.image_grid_pinpoints, CROP_SIZE)
            assert len(tiled) == grid_h * grid_w
            tiled = tiled.view(grid_h, grid_w, OUTPUT_GRID, OUTPUT_GRID, OUTPUT_WIDTH)
            tiled = tiled.permute(4, 0, 2, 1, 3).contiguous()
            tiled = tiled.flatten(1, 2).flatten(2, 3)
            tiled = unpad_image(tiled, image_sizes[image_index])
            channels, current_h, current_w = tiled.shape
            ratio = math.sqrt(current_h * current_w / (9 * OUTPUT_GRID ** 2))
            if ratio > 1.1:
                tiled = F.interpolate(tiled[None], [int(current_h // ratio), int(current_w // ratio)],
                                      mode='bilinear')[0]
            if image_newline is not None:
                assert image_newline.shape == (OUTPUT_WIDTH,)
                newline = image_newline[:, None, None].expand(channels, tiled.shape[1], 1).to(tiled)
                tiled = torch.cat((tiled, newline), dim=-1)
            sequence = torch.cat((base, tiled.flatten(1, 2).transpose(0, 1)), dim=0)
        else:
            assert len(crops) == 1
            sequence = crops[0]
            if image_newline is not None:
                sequence = torch.cat((sequence, image_newline[None].to(sequence)), dim=0)
        packed.append(sequence)
        lengths.append(sequence.shape[0])
    result = torch.cat(packed, dim=0)
    return result, torch.tensor(lengths, dtype=torch.long, device=result.device)


def _types():
    global _CLASSES
    if _CLASSES is not None:
        return _CLASSES
    import torch
    from torch import nn

    class SpatialReadout(nn.Module):
        def __init__(self, tower, weight, bias):
            super().__init__()
            assert tuple(weight.shape) == (OUTPUT_WIDTH, INPUT_WIDTH)
            assert tuple(bias.shape) == (OUTPUT_WIDTH,)
            assert weight.device.type in ('cuda', 'meta') and bias.device == weight.device
            assert weight.dtype == bias.dtype == torch.float16
            assert next(tower.parameters()).device == weight.device
            if weight.device.type == 'cuda':
                assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
            self.student = tower
            # Meta allocation performs no discarded random CUDA calculation.
            self.compiled = nn.Linear(INPUT_WIDTH, OUTPUT_WIDTH, bias=True,
                                      device='meta', dtype=torch.float16)
            self.compiled.weight = nn.Parameter(weight.detach().clone(), requires_grad=False)
            self.compiled.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
            self.requires_grad_(False)

        def forward(self, pixel_values, output_hidden_states=True, **kwargs):
            value = self.compiled(patch_features(self.student, pixel_values))
            return SimpleNamespace(last_hidden_state=value, hidden_states=(value,))

    class PackedSpatial(nn.Module):
        def __init__(self, core):
            super().__init__()
            self.tower = core.vision_tower
            self.projector = core.multi_modal_projector
            self.image_newline = core.image_newline
            object.__setattr__(self, 'core', core)

        def get_dtype(self):
            return next(self.tower.parameters()).dtype

        def forward(self, pixels, grid_thw=None):
            if pixels.ndim == 2 and pixels.shape[-1] == OUTPUT_WIDTH:
                return pixels
            assert grid_thw is not None
            with torch.autocast(pixels.device.type, dtype=self.get_dtype()):
                crop_features = self.core.get_image_features(
                    pixels.to(self.get_dtype()), grid_thw,
                    self.core.config.vision_feature_layer,
                    self.core.config.vision_feature_select_strategy)
                packed, _ = self.core.pack_image_features(
                    crop_features, grid_thw, self.image_newline,
                    self.core.config.vision_aspect_ratio)
            return packed

    _CLASSES = SpatialReadout, PackedSpatial
    return _CLASSES


def compile_interface(tower, weight, bias):
    """Own only tower plus already-GPU-composed FP16 affine, no factors."""
    SpatialReadout, _ = _types()
    return SpatialReadout(tower, weight, bias).eval()


def install(bridge, interface):
    """Rebind core and bridge together, before frozen_vision.install.

    Both core.forward/generate and the bridge's prepacked path call the same
    instance pack method. Existing pack_native_batch rebuilds image placeholders
    from actual returned lengths while preserving all non-image text tokens.
    """
    from torch import nn
    _, PackedSpatial = _types()
    assert not hasattr(bridge, '_llmcl_visual_original'), 'Install before frozen_vision.install'
    core = bridge.core
    assert core.config.vision_config.image_size == 384 and core.config.vision_config.patch_size == 14
    assert core.config.vision_feature_layer == -1
    assert core.config.vision_feature_select_strategy == 'full'
    assert core.config.vision_aspect_ratio == 'anyres_max_9'
    assert core.config.text_config.hidden_size == OUTPUT_WIDTH
    assert isinstance(interface, _types()[0])
    core.vision_tower = interface
    core.multi_modal_projector = nn.Identity()
    core.pack_image_features = MethodType(_pack_image_features, core)
    bridge.visual = PackedSpatial(core)
    assert bridge.visual.tower is core.vision_tower
    assert bridge.visual.projector is core.multi_modal_projector
    assert bridge.visual.image_newline is core.image_newline
    bridge.requires_grad_(False).eval()
    return metadata()
