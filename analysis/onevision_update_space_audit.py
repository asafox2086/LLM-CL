"""Read-only TRAIN-update geometry at one sealed native LoRA checkpoint.

No model forward, QA prediction, optimizer, GPU, or queued-source mutation.
Coordinate counts are distinguished from local weight-space dimensions. These
are known matrix facts and observations, not new theory or F1 causality.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'summary/onevision_diagnostics_20261016'
REPORT = BASE / 'update_space_20261018'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def checks(torch):
    generator = torch.Generator().manual_seed(42)
    rows = []
    for m, n in [(5, 5), (4, 7), (7, 4)]:
        rank = 2
        u, _, vh = torch.linalg.svd(torch.randn(m, n, generator=generator, dtype=torch.float64),
                                    full_matrices=False)
        k = min(m, n) - rank
        ul, vl = u[:, k:], vh[k:, :]
        uh, vhigh = u[:, :k], vh[:k, :]
        left = torch.eye(m, dtype=torch.float64) - uh @ uh.T
        right = torch.eye(n, dtype=torch.float64) - vhigh.T @ vhigh
        free, protected = [], []
        for index in range(m * rank):
            du = torch.zeros(m, rank, dtype=torch.float64)
            du.flatten()[index] = 1
            free.append((du @ vl).flatten())
            protected.append((left @ du @ vl).flatten())
        for index in range(rank * n):
            dv = torch.zeros(rank, n, dtype=torch.float64)
            dv.flatten()[index] = 1
            free.append((ul @ dv).flatten())
            protected.append((ul @ dv @ right).flatten())
        fd = int(torch.linalg.matrix_rank(torch.stack(free, dim=1), tol=1e-8))
        pd = int(torch.linalg.matrix_rank(torch.stack(protected, dim=1), tol=1e-8))
        assert fd == rank * (m + n - rank)
        assert pd == (m - k) * (n - k)
        sample = torch.randn(m, n, generator=generator, dtype=torch.float64)
        block = left @ sample @ right
        # Dense block projection and its residual are orthogonal.
        assert abs(float((block * (sample - block)).sum())) < 1e-10
        assert torch.linalg.vector_norm(block) <= torch.linalg.vector_norm(sample) + 1e-10
        rows.append(dict(shape=[m, n], free_jacobian_rank=fd, protected_jacobian_rank=pd,
                         full_coordinate_count=rank * (m + n)))
    return rows


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    assert not torch.cuda.is_initialized()
    synthetic = checks(torch)
    sources = {}

    def checked(path, expected=None):
        h = sha(path)
        assert expected is None or h == expected, str(path)
        sources[str(path.relative_to(ROOT))] = h
        return path

    checked(Path(__file__))
    plan = json.loads(checked(BASE / 'two_module_prototype/CPU_PILOT_PLAN.json').read_text())['spec']
    start = torch.load(checked(ROOT / plan['past_natural_checkpoint'], plan['past_natural_sha256']),
                       map_location='cpu', weights_only=True)['adapter']
    cases = {}
    for task in (2, 11, 12):
        proof = json.loads(checked(BASE / f'two_module_prototype/T{task:02d}_control_complete.json').read_text())
        assert proof['complete'] and proof['valid_training'] and proof['spec'] == plan
        run = ROOT / proof['run']
        final = torch.load(checked(run / 'final_adapter.pt', proof['artifact_sha256']['final_adapter.pt']),
                           map_location='cpu', weights_only=True)
        raw = torch.load(checked(run / 'raw_gradient.pt', proof['artifact_sha256']['raw_gradient.pt']),
                         map_location='cpu', weights_only=True)
        assert set(raw['TRAIN_ids']) == set(plan['TRAIN_ids'][str(task)])
        assert len(start) == len(final['adapter']) == 48 and start.keys() == final['adapter'].keys()
        assert final['alpha'] / final['rank'] == 4
        cases[task] = dict(final=final['adapter'], TRAIN_QA=len(raw['TRAIN_ids']), layers=[])
    model_path = ROOT / 'model/LLaVA-OneVision-Qwen2-0.5B-OV/model.safetensors'
    checked(model_path)
    model_stat = (model_path.stat().st_size, model_path.stat().st_mtime_ns)
    layers = []
    with safe_open(model_path, framework='pt', device='cpu') as reader:
        for name in sorted(start):
            weight_name = name + '.weight'
            original = reader.get_tensor(weight_name)
            a, b = start[name]['A'].double(), start[name]['B'].double()
            # Algebraic linear coefficient, NOT a claim of bit-exact native
            # forward equivalence across separate LoRA and dense matmuls.
            current = (original.double() + 4 * b @ a).float()
            u, singular, vh = torch.linalg.svd(current, full_matrices=False)
            assert float(singular.min()) > 0
            m, n = current.shape
            rank = 8
            k = min(m, n) - rank
            assert a.shape == (rank, n) and b.shape == (m, rank)
            uh, vhigh = u[:, :k].double(), vh[:k, :].double()
            ul, vl = u[:, k:].double(), vh[k:, :].double()
            reconstruction = (u * singular) @ vh
            relative_error = float((reconstruction - current).norm() / current.norm())
            assert relative_error < 2e-5
            dimensions = dict(name=name, shape=[m, n], trainable_coordinates=rank * (m + n),
                              free_local_weight_dimension=rank * (m + n - rank),
                              protected_local_weight_dimension=(m - k) * (n - k),
                              decomposition_relative_error=relative_error,
                              smallest_singular_value=float(singular.min()),
                              source_tensor_sha256=hashlib.sha256(original.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
            layers.append(dimensions)
            for task, case in cases.items():
                after = case['final'][name]
                da, db = after['A'].double() - a, after['B'].double() - b
                # Exact finite BA displacement, not raw factor gradient.
                delta = 4 * (db @ a + b @ da + db @ da)
                protected = delta - uh @ (uh.T @ delta)
                protected = protected - (protected @ vhigh.T) @ vhigh
                free = ul @ (ul.T @ delta) + (delta @ vl.T) @ vl - ul @ ((ul.T @ delta) @ vl.T) @ vl
                energy = float(delta.square().sum())
                pe, fe = float(protected.square().sum()), float(free.square().sum())
                assert energy > 0 and 0 <= pe <= energy * (1 + 2e-5) and 0 <= fe <= energy * (1 + 2e-5)
                case['layers'].append(dict(name=name, displacement_energy=energy,
                    protected_block_energy=pe, free_tangent_energy=fe,
                    protected_fraction=pe / energy, free_fraction=fe / energy,
                    isotropic_mean_protected_fraction=dimensions['protected_local_weight_dimension'] / (m * n),
                    isotropic_mean_free_fraction=dimensions['free_local_weight_dimension'] / (m * n)))
    totals = {key: sum(x[key] for x in layers) for key in
              ['trainable_coordinates', 'free_local_weight_dimension', 'protected_local_weight_dimension']}
    assert totals['trainable_coordinates'] == 540672 and len(layers) == 48
    summaries = []
    for task, case in cases.items():
        energy = sum(x['displacement_energy'] for x in case['layers'])
        pe = sum(x['protected_block_energy'] for x in case['layers'])
        fe = sum(x['free_tangent_energy'] for x in case['layers'])
        pi = sum(x['displacement_energy'] * x['isotropic_mean_protected_fraction'] for x in case['layers']) / energy
        fi = sum(x['displacement_energy'] * x['isotropic_mean_free_fraction'] for x in case['layers']) / energy
        summaries.append(dict(task=task, TRAIN_QA=case['TRAIN_QA'], finite_BA_displacement_norm=math.sqrt(energy),
                              protected_retained_energy_fraction=pe / energy,
                              free_tangent_retained_energy_fraction=fe / energy,
                              isotropic_mean_protected_fraction=pi, isotropic_mean_free_fraction=fi,
                              protected_observed_over_isotropic_mean=(pe / energy) / pi,
                              free_observed_over_isotropic_mean=(fe / energy) / fi,
                              layers=case['layers']))
    assert not torch.cuda.is_initialized()
    assert model_stat == (model_path.stat().st_size, model_path.stat().st_mtime_ns)
    # Recheck small immutable inputs. The full model was hashed above; it is
    # read-only here and no trainer/source or queued input is written.
    for path, digest in sources.items():
        if path != str(model_path.relative_to(ROOT)):
            assert sha(ROOT / path) == digest
    limits = [
        'Known local rank-manifold and orthogonal-projection algebra, not novel theory.',
        'One original native LoRA start and three sealed one-step TRAIN updates only; no population claim.',
        'Projected finite LoRA displacement is a representability diagnostic, NOT the SVD optimizer step or loss descent.',
        'Protected/free SVD GPU starts and trajectories have NOT been measured by this script.',
        'Native effective coefficient uses original W plus scaled BA; separate BF16 matmuls need not equal one merged forward.',
        'Low retained energy does not prove bad directions, poor F1, or that removing protection improves performance.',
        'Small projected energy is expected in a low-dimensional space; an analytic isotropic mean is reported, not a stochastic significance test.',
        'Local differential dimension is not statistical capacity, memory, trainability, or independent semantic knowledge.',
        'No TEST predictions/answers/scores, model forward, GPU, optimizer, route or hyperparameter selection.',
        'No edits to frozen queued sources/protocol/settings; both predeclared SVD conditions and full confirmations retained.'
    ]
    result = dict(CPU_only=True, TEST_read=False, model_forward=False, new_optimizer=False,
                  synthetic_Jacobian_controls=synthetic, dimensions=totals, actual48_layers=layers,
                  actual_TRAIN_update_cases=summaries, source_sha256=sources, limits=limits)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'EVIDENCE.json').write_text(json.dumps(result, indent=2) + '\n')
    lines = ['# 相同训练参数数量，不等于相同可更新空间', '',
             '只读原native自然域后同一起点和T02/T11/T12三条已封存TRAIN单步更新；不读TEST、不执行模型前向或新优化器。', '',
             '| 量 | 全48个q/v之和 |', '| --- | ---: |']
    for key, value in totals.items():
        lines.append(f'| {key} | {value} |')
    lines += ['', 'free局部维数 r(m+n-r)；protected高空间秩k=min(m,n)-r后，允许块维数(m-k)(n-k)。矩形v存在额外零空间，不能把全部矩阵误算为r²。合成三形状以实际Jacobian秩独立核验。', '',
              '| TRAIN任务 | QA | 实际BA位移范数 | protected能量% | 各向同性均值% | free能量% | 各向同性均值% |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for x in summaries:
        lines.append(f"| T{x['task']:02d} | {x['TRAIN_QA']} | {x['finite_BA_displacement_norm']:.6f} | {100*x['protected_retained_energy_fraction']:.4f} | {100*x['isotropic_mean_protected_fraction']:.4f} | {100*x['free_tangent_retained_energy_fraction']:.4f} | {100*x['isotropic_mean_free_fraction']:.4f} |")
    lines += ['', '各向同性参考是每层维数/(m×n)的精确方向均值，再按原实际位移能量加权；不是随机显著性检验。小子空间本来就只能覆盖少量任意方向，不能只凭低能量比例判定特殊不匹配或有害。', '',
              '这提供一个需要GPU对照检验的风险：缩小SVD余量同时加保护会限制可更新方向。不能据此认定该限制已造成准确率损害，也不能把原LoRA方向当作最优方向。当前两组GPU对照保持原样，等待实际习得／最终F1与资源证据。', '', '局限：']
    lines += ['- ' + item for item in limits]
    (REPORT / 'RESEARCH_FINDINGS.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(dict(dimensions=totals, cases=[{k:v for k,v in x.items() if k!='layers'} for x in summaries],
                          CPU_only=True, TEST_read=False)))


if __name__ == '__main__':
    main()
