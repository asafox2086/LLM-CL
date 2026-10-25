"""One paired GPU alignment of fixed spatial readout coordinates.

The caller supplies the same shuffled small-tower features and teacher tokens
once per QA. This module neither loads models nor opens QA/images. Projection,
pooling, moments, ridge solve, composition and diagnostics are CUDA operations;
CPU is used only for permutation metadata, scalar receipts and optional hashes.
"""
import hashlib
import json
import random
import struct
import time

ARMS = ('native', 'permuted')
INPUT_WIDTH = 12288
PROJECTED_WIDTH = 960
OUTPUT_WIDTH = 896
FIT_QA = 293
HELD_QA = 32


def configure_gpu_math():
    import torch
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    return dict(statistics='CUDA torch.float64', pooling='CUDA torch.float32 then torch.float16',
        fixed_projection='CUDA torch.float16', compose='CUDA torch.float32 then torch.float16',
        matmul_allow_tf32=False, autocast_disabled_for_pool_statistics_compose=True)


def make_projections(native_weight, storage=None):
    """W and W[:,perm], with one stdlib seed42 permutation of all input columns."""
    import torch
    assert native_weight.is_cuda and tuple(native_weight.shape) == (PROJECTED_WIDTH, INPUT_WIDTH)
    assert native_weight.is_floating_point()
    original_dtype = str(native_weight.dtype)
    weight = native_weight.detach().to(dtype=torch.float16).contiguous().clone()
    assert torch.isfinite(weight).all()
    permutation = list(range(INPUT_WIDTH))
    random.Random(42).shuffle(permutation)
    assert len(set(permutation)) == INPUT_WIDTH
    encoded = struct.pack('<' + 'I' * INPUT_WIDTH, *permutation)
    indices = torch.tensor(permutation, dtype=torch.long, device=weight.device)
    alternate = weight.index_select(1, indices).contiguous()
    assert alternate.device == weight.device and alternate.dtype == torch.float16
    bank = dict(native=weight, permuted=alternate)
    receipt = dict(seed=42, permutation_algorithm='stdlib random.Random(42).shuffle(range(12288))',
        permutation_SHA256=hashlib.sha256(encoded).hexdigest(),
        permutation_encoding='12288 unsigned32 integers, little endian',
        projection_shape=[PROJECTED_WIDTH, INPUT_WIDTH], original_dtype=original_dtype,
        fixed_projection_dtype='torch.float16', bias=False, nonlinear_activation=False,
        complete_input_column_bijection=True,
        singular_spectrum_and_row_norms_preserved_by_permutation=True,
        control_changes='correspondence of fixed readout columns to patch-channel input coordinates',
        no_pure_spatial_or_universal_pretraining_necessity_claim=True)
    if storage is not None:
        receipt['weight_SHA256'] = {arm: storage.digest({'weight': value}) for arm, value in bank.items()}
    return bank, receipt


def teacher_target(teacher):
    """Same27x27 projected teacher tokens -> adaptive-average8x8, not interpolation."""
    import torch
    import torch.nn.functional as F
    assert teacher.is_cuda and teacher.dtype == torch.float16
    assert teacher.ndim == 3 and tuple(teacher.shape[1:]) == (729, OUTPUT_WIDTH)
    assert teacher.shape[0] > 0
    with torch.no_grad(), torch.cuda.device(teacher.device), torch.autocast('cuda', enabled=False):
        grid = teacher.detach().transpose(1, 2).reshape(-1, OUTPUT_WIDTH, 27, 27).float()
        target = F.adaptive_avg_pool2d(grid, (8, 8)).flatten(2).transpose(1, 2)
        target = target.contiguous().to(dtype=torch.float16)
    assert tuple(target.shape) == (teacher.shape[0], 64, OUTPUT_WIDTH)
    assert torch.isfinite(target).all()
    return target


def _check_pair(raw, teacher):
    import torch
    assert raw.is_cuda and teacher.device == raw.device
    assert raw.dtype == teacher.dtype == torch.float16
    assert raw.ndim == 3 and tuple(raw.shape[1:]) == (64, INPUT_WIDTH)
    assert raw.shape[0] == teacher.shape[0] and raw.shape[0] > 0
    assert torch.isfinite(raw).all() and torch.isfinite(teacher).all()


def _metadata(meta):
    assert isinstance(meta, dict) and isinstance(meta['id'], str) and isinstance(meta['group_id'], str)
    if 'task' in meta:
        assert meta['task'] == 1
    if 'domain' in meta:
        assert meta['domain'] == 'natural'
    json.dumps(meta, ensure_ascii=False, allow_nan=False)
    return dict(meta)


class Moments:
    """Each QA contributes one averaged second moment, independent of tile count."""
    def __init__(self, device):
        import torch
        self.device = torch.device(device)
        assert self.device.type == 'cuda'
        self.count = 0
        self.sx = torch.zeros(PROJECTED_WIDTH, device=self.device, dtype=torch.float64)
        self.sy = torch.zeros(OUTPUT_WIDTH, device=self.device, dtype=torch.float64)
        self.xx = torch.zeros((PROJECTED_WIDTH, PROJECTED_WIDTH), device=self.device, dtype=torch.float64)
        self.xy = torch.zeros((PROJECTED_WIDTH, OUTPUT_WIDTH), device=self.device, dtype=torch.float64)

    def add(self, projected, target):
        import torch
        assert projected.device == target.device == self.device
        assert projected.shape[:-1] == target.shape[:-1]
        with torch.no_grad(), torch.cuda.device(self.device), torch.autocast('cuda', enabled=False):
            x = projected.detach().reshape(-1, PROJECTED_WIDTH).double()
            y = target.detach().reshape(-1, OUTPUT_WIDTH).double()
            assert len(x) > 0 and torch.isfinite(x).all() and torch.isfinite(y).all()
            self.sx.add_(x.mean(0))
            self.sy.add_(y.mean(0))
            self.xx.add_((x.T @ x) / len(x))
            self.xy.add_((x.T @ y) / len(x))
            self.count += 1

    def solve_and_compile(self, projection, storage=None):
        import torch
        assert self.count == FIT_QA and projection.device == self.device
        assert tuple(projection.shape) == (PROJECTED_WIDTH, INPUT_WIDTH)
        assert projection.dtype == torch.float16
        assert torch.backends.cuda.matmul.allow_tf32 is False
        with torch.no_grad(), torch.cuda.device(self.device), torch.autocast('cuda', enabled=False):
            mx, my = self.sx / self.count, self.sy / self.count
            cxx = self.xx / self.count - mx[:, None] * mx[None, :]
            cxy = self.xy / self.count - mx[:, None] * my[None, :]
            cxx = (cxx + cxx.T) / 2
            ridge = 1e-3 * torch.trace(cxx) / PROJECTED_WIDTH
            assert ridge.isfinite() and ridge.item() > 0
            coefficient = torch.linalg.solve(cxx + ridge * torch.eye(
                PROJECTED_WIDTH, dtype=torch.float64, device=self.device), cxy)
            bias = my - mx @ coefficient
            assert torch.isfinite(coefficient).all() and torch.isfinite(bias).all()
            # The deployed graph directly executes this one compiled FP16 map.
            weight = (coefficient.T.float() @ projection.detach().float()).half().contiguous()
            bias = bias.float().half().contiguous()
            assert torch.isfinite(weight).all() and torch.isfinite(bias).all()
        payload = dict(weight=weight, bias=bias)
        receipt = dict(FIT_QA=self.count, rows_equal_within_QA=True, QA_equal_weight=True,
            input_width=PROJECTED_WIDTH, output_width=OUTPUT_WIDTH,
            statistics_device=str(self.device), statistics_dtype='torch.float64',
            covariance_centered=True, intercept_penalty=False, ridge=float(ridge.item()),
            ridge_rule='1e-3*trace(centered_Cxx)/960',
            compiled_weight_shape=list(weight.shape), compiled_bias_shape=list(bias.shape),
            compiled_dtype='torch.float16', compose_dtype='torch.float32',
            compose_device=str(self.device), compose_autocast=False, compose_TF32=False,
            compiled_coefficients=weight.numel() + bias.numel(),
            final_payload_keys=['weight', 'bias'], no_projection_or_factor_payload=True,
            intermediate_FP16_projection_vs_compiled_rounding_not_assumed_equal=True)
        if storage is not None:
            receipt['compiled_SHA256'] = storage.digest(payload)
        return payload, receipt


def fit_streams(native_weight, fit_stream, held_stream, storage=None, on_record=None):
    """Streams yield(raw[T,64,12288], teacher[T,729,896], metadata{id,group_id}).

    Consume293fit and32held known S1 groups once each. The same raw pair feeds
    both fixed arms. Returned compiled[arm] contains only CUDA FP16 weight/bias;
    receipt contains scalar metadata, never factors, source W or feature caches.
    """
    import torch
    import torch.nn.functional as F
    math_receipt = configure_gpu_math()
    device = native_weight.device
    torch.cuda.synchronize(device)
    tick = time.monotonic()
    projections, permutation = make_projections(native_weight, storage)
    moments = {arm: Moments(device) for arm in ARMS}
    ids, groups, fit_records = set(), set(), []
    for raw, teacher, meta in fit_stream:
        meta = _metadata(meta)
        assert meta['id'] not in ids and meta['group_id'] not in groups
        ids.add(meta['id']); groups.add(meta['group_id'])
        assert len(ids) <= FIT_QA
        _check_pair(raw, teacher)
        assert raw.device == device
        target = teacher_target(teacher)
        with torch.no_grad(), torch.cuda.device(device), torch.autocast('cuda', dtype=torch.float16):
            for arm in ARMS:
                projected = F.linear(raw, projections[arm])
                moments[arm].add(projected, target)
        record = dict(phase='fit', **meta, tiles=raw.shape[0], rows=raw.shape[0] * 64,
                      arms_same_raw_teacher_pair=True, QA_weight=1)
        fit_records.append(record)
        if on_record is not None:
            on_record(record)
        del raw, teacher, target, projected
    assert len(ids) == len(groups) == FIT_QA
    compiled, fit_receipts = {}, {}
    for arm in ARMS:
        compiled[arm], fit_receipts[arm] = moments[arm].solve_and_compile(projections[arm], storage)
    del projections, moments
    held_ids, held_groups, held_records = set(), set(), []
    held_totals = {arm: dict(MSE=0., target_energy=0.) for arm in ARMS}
    for raw, teacher, meta in held_stream:
        meta = _metadata(meta)
        assert meta['id'] not in ids | held_ids and meta['group_id'] not in groups | held_groups
        held_ids.add(meta['id']); held_groups.add(meta['group_id'])
        assert len(held_ids) <= HELD_QA
        _check_pair(raw, teacher)
        assert raw.device == device
        target = teacher_target(teacher)
        diagnostics = {}
        with torch.no_grad(), torch.cuda.device(device):
            for arm in ARMS:
                with torch.autocast('cuda', dtype=torch.float16):
                    actual = F.linear(raw, compiled[arm]['weight'], compiled[arm]['bias'])
                with torch.autocast('cuda', enabled=False):
                    mse = (actual.float() - target.float()).square().mean()
                    energy = target.float().square().mean()
                assert mse.isfinite() and energy.isfinite()
                diagnostics[arm] = dict(MSE=mse.item(), target_energy=energy.item(),
                    ratio=(mse / energy).item() if energy.item() > 0 else None)
                held_totals[arm]['MSE'] += mse.item() / HELD_QA
                held_totals[arm]['target_energy'] += energy.item() / HELD_QA
        record = dict(phase='held', **meta, tiles=raw.shape[0], rows=raw.shape[0] * 64,
                      compiled_only=True, diagnostics=diagnostics)
        held_records.append(record)
        if on_record is not None:
            on_record(record)
        del raw, teacher, target, actual
    assert len(held_ids) == len(held_groups) == HELD_QA
    torch.cuda.synchronize(device)
    return compiled, dict(math=math_receipt, permutation=permutation, fit=fit_receipts,
        FIT_QA=FIT_QA, held_QA=HELD_QA, fit_records=fit_records, held_records=held_records,
        held_QA_equal_weight=held_totals,
        teacher_target='GPU FP32 adaptive_avg_pool2d27to8 then FP16, channel width896',
        no_decoder_or_answer_supervision=True, full_feature_cache=False,
        final_projection_factors_not_returned=True, GPU_device=str(device),
        synchronized_seconds=time.monotonic() - tick)
