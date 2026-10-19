"""Admission checks for a published-optimizer continual-tuning adaptation."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/lora_rite_continuous_20261019/v1'


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(4)
    sys.path.insert(0, str(ROOT / 'analysis'))
    import onevision_lora_rite_trial_20261019 as worker
    from onevision_lora_optimizer_counterfactual_20261019 import author_class, tensor_bytes
    assert not torch.cuda.is_initialized()
    checks = []
    for device in ('0', '1'):
        script = "import os,sys;sys.path.insert(0,'analysis');import onevision_lora_rite_trial_20261019 as w;import torch;assert os.environ['CUDA_VISIBLE_DEVICES']==sys.argv[1];assert not torch.cuda.is_initialized();w.author_class();assert not torch.cuda.is_initialized();print('PASS')"
        env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = device
        result = subprocess.run([sys.executable, '-c', script, device], env=env,
                                check=True, text=True, capture_output=True)
        assert result.stdout.strip() == 'PASS'
    checks.append('actual complete worker imports preserve physical0/1 and do not initialize CUDA')
    generator = torch.Generator().manual_seed(42)
    pairs = {}
    for m in (128, 896):
        a = torch.randn(8, 896, generator=generator) / 30.
        b = torch.zeros(m, 8)
        pairs[str(m)] = dict(A=a, B=b)
    parameters = []
    for name, item in pairs.items():
        a = torch.nn.Parameter(item['A'].clone()); b = torch.nn.Parameter(item['B'].clone())
        g = torch.randn(b.shape[0], a.shape[1], generator=generator) * .001
        a.grad = b.detach().T @ g; b.grad = g @ a.detach().T
        parameters.extend((a, b))
    optimizer = author_class()(parameters, lr=1e-4, betas=(.9, .999), clip_unmagnified_grad=1.)
    with torch.no_grad():
        optimizer.step()
    end = {name: dict(A=parameters[2*i].detach(), B=parameters[2*i+1].detach())
           for i, name in enumerate(pairs)}
    assert all(torch.equal(end[n]['A'], pairs[n]['A']) and torch.count_nonzero(end[n]['B']) for n in pairs)
    assert all(torch.isfinite(v).all() for item in end.values() for v in item.values())
    checks.append('author optimizer finite on exact zero-B cold start; first A stays fixed and B changes')
    explicit = sum(float((4 * (end[n]['B'].double() @ end[n]['A'].double())).square().sum()) for n in pairs) ** .5
    actual = worker.displacement(pairs, end)
    assert abs(actual - explicit) < 1e-12
    checks.append('effective-deltaW low-rank Gram calculation equals actual dense result')
    state_bytes = tensor_bytes(optimizer.state)
    counts = sum(p.numel() for p in parameters)
    assert state_bytes == (2 * counts + 4 * 8 * 8) * 4
    checks.append('persistent author optimizer tensor count includes both bases, moments and two covariance matrices per layer')
    audit = json.loads((ROOT / 'summary/onevision_diagnostics_20261016/optimizer_geometry_20261019/v1/EVIDENCE.json').read_text())
    assert audit['status'] == 'CPU_ACTUAL_TRAIN_GRADIENT_COUNTERFACTUAL_COMPLETE'
    assert all(r['variants'][0]['actual_deltaW_norm'] > 0 for r in audit['results'])
    checks.append('three sealed real-TRAIN gradient cases already completed; evidence reused without rerunning')
    spec, values = worker.plan()
    assert spec['train_QA'] == 1313 and spec['stages'] == 19 and spec['expected_updates'] == 93
    assert spec['full_QA'] == 21454 and spec['expert_count'] == 1
    assert spec['optimizer_settings']['clip_unmagnified_grad'] == 1. and not spec['external_factor_gradient_clipping']
    manifest, data, domains, tests, full, original, proof = values
    assert sum(len(data[t, 'train']) for t in range(1, 20)) == 1313
    assert sum(len(rr) for rr in full.values()) == 21454
    assert sum(len(rr) for rr in tests.values()) == 500
    assert sum(len(rr) for rr in spec['generation_ids'].values()) == 155
    checks.append('actual frozen counts and one-expert constant capacity; full confirmation required independently of diagnostics')
    assert all((ROOT / rel).is_file() and worker.digest(ROOT / rel) == h for rel, h in spec['source_sha256'].items())
    checks.append('all inherited/scientific/pinned author sources bind exact SHA')
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / 'CPU_PLAN.json').write_text(json.dumps(dict(spec=spec, checks=checks,
        cold_numerical_control_optimizer_tensor_bytes=state_bytes, GPU_model_NOT_run=True), indent=2) + '\n')
    assert not torch.cuda.is_initialized()
    print(json.dumps(dict(checks_passed=len(checks), checks=checks, GPU_model_NOT_run=True)))


if __name__ == '__main__':
    main()
