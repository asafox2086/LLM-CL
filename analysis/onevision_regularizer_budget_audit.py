"""CPU-only, source-bound PASs-MoE activation audit; no model performance claims.

Uses the current V2 adapter's unchanged, SHA-checked author learner on a tiny
CPU module. Does not load a backbone, read TEST, or alter any queued source.
"""
import argparse
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import tempfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source(path, first, last):
    p = ROOT / path
    lines = p.read_text().splitlines()
    return {'path': path, 'sha256': sha(p), 'first_line': first,
            'last_line': last, 'text': '\n'.join(lines[first-1:last])}


def build():
    import torch
    from torch import nn
    from types import SimpleNamespace
    sys.path.insert(0, str(ROOT / 'exp/two_backbone_native_cl_20261017_v2'))
    from pass_adapter import components

    manifest_path = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    run = ROOT / 'exp/two_backbone_native_cl_20261017_v2/run.py'
    text = run.read_text()
    # Fail closed if the schedule/call contract changes; no inferred migration.
    for needle in ['updates = math.ceil(len(order)/16)',
                   'enumerate(range(0, len(order), 16), 1)',
                   'method.post_backward(scaler.get_scale(), start/len(order))']:
        assert needle in text, needle
    rows = []
    for task in manifest['tasks']:
        entry = task['files']['train']
        path = ROOT / entry['path']
        assert sha(path) == entry['sha256']
        n = sum(bool(line.strip()) for line in path.read_text().splitlines())
        assert n == entry['count'] and n > 0
        progress = [start/n for start in range(0, n, 16)]
        rows.append({'task': task['task_id'], 'domain': task['domain'],
                     'TRAIN_QA': n, 'updates': math.ceil(n/16),
                     'progress_per_update': progress,
                     'warmup_factor_per_update': [min(1., p/.3) for p in progress],
                     'all_updates_zero_warmup_coefficient': all(p == 0 for p in progress),
                     'TRAIN_sha256': entry['sha256']})
    assert sum(x['TRAIN_QA'] for x in rows) == 1313
    assert sum(x['updates'] for x in rows) == 93
    one = [x['task'] for x in rows if x['all_updates_zero_warmup_coefficient']]
    assert one == ['T02','T05','T06','T07','T08','T09','T10','T11','T12']

    author, revision = components()  # Checks the recorded author file hashes.
    torch.manual_seed(42)
    layer = author['CoINMOELoraLinear'](
        'default', 4, 4, r=4, lora_alpha=8, lora_dropout=0., expert_num=2,
        gate_rank=2, gate_tau=1.5, gate_router='a', gate_tie='A_T')
    model = nn.Sequential(layer)
    with tempfile.TemporaryDirectory() as temp:
        learner = author['moeloraLearner'](SimpleNamespace(
            moelora_orth_lambda=1e-4, moelora_B_reg_scale=.1,
            moelora_importance_top_p=.4, output_dir=temp), 0, model)
        A = [e.mlp.weight for e in layer.lora_A['default'].loraA]
        B = [e.mlp.weight for e in layer.lora_B['default'].loraB]
        # Deliberately nonzero distances: a zero gradient cannot be explained
        # by theta == reference, empty importance, or a missing .grad tensor.
        learner._I_agg = {'0': torch.ones(layer._current_E(), A[0].shape[0])}
        learner._A_ref = {'0': torch.stack([p.detach().clone()-.25 for p in A])}
        learner._B_ref = {'0': torch.stack([p.detach().clone()-.25 for p in B])}
        before = [p.detach().clone() for p in A+B]
        controls = []
        for ratio, progress in [(.3, 0.), (.3, .15), (.3, .3), (0., 0.)]:
            for p in A+B:
                p.grad = torch.zeros_like(p)
            learner.reg_warmup_ratio = ratio
            learner.post_backward(model, cl_task_progress=progress)
            norm = sum(float(p.grad.square().sum()) for p in A+B)**.5
            controls.append({'warmup_ratio': ratio, 'progress': progress,
                             'added_regularization_gradient_norm': norm})
        assert controls[0]['added_regularization_gradient_norm'] == 0
        assert controls[1]['added_regularization_gradient_norm'] > 0
        assert math.isclose(controls[2]['added_regularization_gradient_norm'],
                            2*controls[1]['added_regularization_gradient_norm'], rel_tol=1e-6)
        assert controls[3]['added_regularization_gradient_norm'] == controls[2]['added_regularization_gradient_norm']
        assert all(torch.equal(a, b) for a, b in zip(before, A+B))

    import transformers
    from transformers import Trainer
    inner = inspect.getsource(Trainer._inner_training_loop)
    hf_file = Path(inspect.getsourcefile(Trainer))
    assert inner.index('self.state.epoch = 0') < inner.index('self.training_step(')
    assert inner.index('self.training_step(') < inner.index('self.state.epoch = epoch +')
    upstream_trainer = ROOT/'code/PASs_MoE/source/llava/train/llava_trainer.py'
    upstream = upstream_trainer.read_text()
    assert 'cur_epoch / max(num_epochs, 1e-6)' in upstream
    assert 'self.cl_learner.post_backward(raw_model, cl_task_progress=task_progress)' in upstream
    result = {
        'kind': 'CPU operator control + deterministic source schedule, NOT GPU performance',
        'author_revision': revision, 'manifest_sha256': sha(manifest_path),
        'audit_source_sha256': sha(Path(__file__)), 'seed': 42,
        'torch_version': torch.__version__, 'CUDA_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
        'GPU_backbone_loads': 0, 'TEST_QA_read': 0, 'optimizer_updates': 0,
        'schedule': rows, 'one_update_zero_coefficient_tasks': one,
        'total_TRAIN_QA': 1313, 'total_optimizer_updates_in_protocol': 93,
        'tiny_CPU_operator_controls': controls, 'tiny_parameters_unchanged': True,
        'installed_HF_epoch_order': {'version': transformers.__version__,
                                   'trainer_source_sha256': sha(hf_file),
                                   'scope': 'installed Trainer source order only; author full Trainer not executed'},
        'sources': [source('exp/two_backbone_native_cl_20261017_v2/run.py', 243, 298),
                    source('exp/two_backbone_native_cl_20261017_v2/pass_adapter.py', 76, 100),
                    source('code/PASs_MoE/source/cl_learner/moelora.py', 165, 216),
                    source('code/PASs_MoE/source/cl_learner/moelora.py', 283, 314),
                    source('code/PASs_MoE/source/llava/train/llava_trainer.py', 478, 490)],
        'limits': ['No actual OneVision regularizer-gradient telemetry was added or inferred.',
                   'Zero penalty coefficient is not zero task LR, zero task gradient, or zero learning.',
                   'No regularization-disable or warmup-change GPU ablation; no F1 cause/effect established.',
                   'Other PASs-MoE components continue; a method can improve with this term inactive.',
                   'Full upstream trainer accumulation/runtime equivalence is not established.',
                   'No active/queued trainer, data, parameters, or schedule modified.']}
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    result = build()
    target = Path(args.out)
    if target.exists():
        raise FileExistsError('Evidence already exists; use an explicitly new evidence path.')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(target), 'one_update_tasks': result['one_update_zero_coefficient_tasks'],
                      'CPU_controls': result['tiny_CPU_operator_controls'], 'GPU_results': False}))
