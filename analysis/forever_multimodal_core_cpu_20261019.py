"""Exercise the new FOREVER loss/drift/selector bridge without a model or GPU."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

from forever_multimodal_core_20261019 import (
    ROOT, REVISION, ForeverStage, intensity_beta, kernels, select_history,
)

REPORT = ROOT / 'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v1'


def main():
    destination = REPORT / 'CORE_CPU_CHECKS.json'
    assert not destination.exists(), 'Preserve prior evidence; use a new version'
    assert not torch.cuda.is_initialized()
    checks = []
    original_env = dict(os.environ)
    assert importlib.import_module('forever_multimodal_core_20261019')
    assert dict(os.environ) == original_env and not torch.cuda.is_initialized()
    for device in ['0', '1']:
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': device}
        result = subprocess.run([
            sys.executable, '-c',
            'import os,torch; before=dict(os.environ); '
            'import forever_multimodal_core_20261019; '
            'assert dict(os.environ)==before; assert not torch.cuda.is_initialized()'
        ], env=env, cwd=ROOT/'analysis', capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    checks.append('imports preserve CUDA_VISIBLE_DEVICES 0/1 and do not initialize CUDA')

    policy = json.loads((ROOT/'exp/two_backbone_native_cl_20261017_v2/execution_policy.json').read_text())
    path = ROOT/policy['conditions']['pct01']['manifest']
    manifest = json.loads(path.read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == policy['conditions']['pct01']['manifest_sha256']
    train = []
    data_sha = {}
    for task in manifest['tasks'][:2]:
        info = task['files']['train']
        raw = (ROOT/info['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == info['sha256']
        train.append([json.loads(line) for line in raw.splitlines()])
        data_sha[info['path']] = info['sha256']
    assert [len(rows) for rows in train] == [347, 11]
    selected, selection = select_history(train, 2)
    old = json.loads((ROOT/'summary/two_backbone_pct01_20261017/forever_preflight_20261018/EVIDENCE.json').read_text())
    expected = old['profile'][1]
    assert len(selected) == expected['author_memory_QA'] == 303
    assert selection['selected_ids_sha256'] == expected['selected_TRAIN_IDs_sha256']
    assert not set(selection['selected_ids']) & {row['id'] for row in train[1]}
    checks.append('new selector bridge reproduces all 303 pinned historical TRAIN IDs')

    # Small CPU tensors validate the new integration, not model performance.
    with tempfile.TemporaryDirectory(prefix='llmcl-forever-core-') as temp:
        model = torch.nn.Linear(3, 2, bias=False).double()
        with torch.no_grad():
            model.weight.copy_(torch.arange(6, dtype=torch.float64).reshape(2, 3)/10)
        stage = ForeverStage(model, train, 2, Path(temp)/'stage02')
        theta = model.weight.detach().clone()
        with torch.no_grad():
            model.weight.add_(0.125)
        stage.main_update_complete()
        info = stage.end_current()
        assert info['main_updates'] == 1 and info['dynamic_replay_triggers'] == 0
        assert info['memory_QA'] == 303 and info['final_replay_required']
        assert info['tau'] == float(torch.linalg.vector_norm(
            model.weight.detach().float()-theta.float()))
        checks.append('one-step calibration records actual displacement and retains final replay')

        model.zero_grad(set_to_none=True)
        base = (model.weight**2).sum()
        total, terms = stage.loss(base)
        expected_total = base + 0.001*stage.beta*((model.weight-theta)**2).sum()
        assert torch.equal(total, expected_total)
        actual_grad = torch.autograd.grad(total, model.weight, retain_graph=True)[0]
        expected_grad = torch.autograd.grad(expected_total, model.weight)[0]
        assert torch.equal(actual_grad, expected_grad) and torch.isfinite(actual_grad).all()
        assert terms['unscaled_regularizer'] > 0 and terms['regularizer_coefficient'] > 0
        checks.append('unchanged author regularized total loss and gradient are exact')

        prev_tau = stage.drift.tau
        prev_step = stage.drift.step
        with torch.no_grad():
            model.weight.sub_(0.01*actual_grad)
        stage.replay_complete()
        assert stage.drift.tau == prev_tau and stage.drift.step == prev_step
        assert torch.equal(stage.drift.prev_vec, model.weight.detach().float().reshape(-1).cpu())
        checks.append('replay updates parameter reference without advancing author model time')

        first = ForeverStage(model, train, 1, Path(temp)/'stage01')
        assert first.memory == [] and first.selection is None
        assert first.replay_loss is None
        checks.append('first stage does not fabricate historical samples or replay')
        try:
            ForeverStage(model, [train[0]*2], 1, Path(temp)/'unsupported')
        except AssertionError as error:
            assert 'General dynamic scheduler required' in str(error)
        else:
            raise AssertionError('Larger update budget was silently accepted')
        checks.append('larger budgets fail closed instead of omitting dynamic replay')

    assert intensity_beta(SimpleNamespace(mu=4., mu0=1.)) == 3.
    assert intensity_beta(SimpleNamespace(mu=0., mu0=1.)) == .5
    checks.append('author intensity scaling retains original 0.5 and 3 bounds')
    _, author_sha = kernels()
    files = ['analysis/forever_multimodal_core_20261019.py',
             'analysis/forever_multimodal_core_cpu_20261019.py']
    report = {
        'status': 'CORE_BRIDGE_ONLY_NOT_GPU_ADMITTED',
        'checks_passed': len(checks), 'checks': checks,
        'upstream_revision': REVISION, 'author_training_source_sha256': author_sha,
        'source_sha256': {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                          for name in files},
        'TRAIN_files_sha256': data_sha,
        'model_loaded': False, 'GPU_forward': 0, 'TEST_predictions_or_answers_read': 0,
        'actual_forever_optimizer_updates': 0,
        'CPU_toy_tensors_only': True,
        'ready_to_launch_formal_run': False,
        'pending': ['common multimodal runner integration',
                    'actual HF replay scheduler and optimizer instrumentation',
                    'both-backbone finite-gradient and checkpoint-reload gates',
                    'independent formal publication verifier'],
    }
    REPORT.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'checks_passed': len(checks), 'GPU_admitted': False,
                      'report': str(destination)}))


if __name__ == '__main__':
    main()
