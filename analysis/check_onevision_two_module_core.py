"""CPU mechanism checks on real adapter factors and explicit finite-step controls.

No target QA generated, no model scored, no improvement/novelty claim.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
from pathlib import Path

import torch
from onevision_two_module_core import (TrainLoss, accept_displacement, choose_reuse,
                                      clone, reuse_function, interpolate)

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016/two_module_prototype'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    checks = []
    reference_path = ROOT / 'summary/onevision_diagnostics_20261016/research_current_attempt.json'
    reference = json.loads(reference_path.read_text())
    path = ROOT / reference['old_control_run'] / 'adapters/isolated_natural.pt'
    assert sha(path) == reference['completed_control_sha256'][str(path.relative_to(ROOT))]
    actual = torch.load(path, map_location='cpu', weights_only=True)['adapter']
    assert len(actual) == 48
    unchanged = clone(actual)
    for strength in (0., .5, 1.):
        candidate = reuse_function(actual, strength)
        assert all(torch.equal(candidate[n]['A'], actual[n]['A']) for n in actual)
        assert all(torch.equal(candidate[n]['B'], actual[n]['B'] * strength) for n in actual)
        # The operation is checked against actual saved BA functions, not norms.
        for name in list(actual)[:2]:
            before = actual[name]['B'].double() @ actual[name]['A'].double()
            after = candidate[name]['B'].double() @ candidate[name]['A'].double()
            assert torch.equal(after, strength * before)
        assert all(torch.equal(actual[n][f], unchanged[n][f]) for n in actual for f in ('A', 'B'))
    checks.append('all48 real adapter A bases retained; B/function scaling exact; source unmodified')
    copied = interpolate(actual, unchanged, 1.)
    assert all(torch.equal(copied[n][f], unchanged[n][f]) for n in actual for f in ('A', 'B'))
    checks.append('full-step interpolation is an exact independent copy, without subtraction-rounding drift')

    def state(a, b):
        return {'layer': {'A': torch.tensor([[a]], dtype=torch.float64),
                          'B': torch.tensor([[b]], dtype=torch.float64)}}

    def function(s):
        return float((s['layer']['B'] @ s['layer']['A'])[0, 0])

    current_ids, memory_ids = ('TRAIN:current',), {'old': ('TRAIN:old',)}

    def evidence(s):
        y = function(s)
        return TrainLoss(y*y, current_ids), {'old': TrainLoss((y-1)**2, memory_ids['old'])}

    old = state(1, 1)
    unguarded, a = choose_reuse(old, evidence, current_ids, memory_ids, {'old': 0.}, False)
    guarded, ab = choose_reuse(old, evidence, current_ids, memory_ids, {'old': 0.}, True)
    assert a['chosen_strength'] == 0 and ab['chosen_strength'] == 1
    checks.append('current-fit reuse candidate can erase old function; TRAIN guard rejects it in explicit conflicting case')
    accepted, b = accept_displacement(guarded, state(1, 0), evidence,
                                      current_ids, memory_ids, {'old': 0.})
    assert b['accepted_fraction'] == .125
    cur, memory = evidence(accepted)
    assert cur.value < evidence(guarded)[0].value and memory['old'].value <= .02
    checks.append('finite-step guard accepts nonzero improvement with old quadratic budget respected')

    # Both A and B change: guard must measure the actual BA, including cross term.
    def bilinear_evidence(s):
        y = function(s)
        return TrainLoss((y-1)**2, current_ids), {'old': TrainLoss(y*y, memory_ids['old'])}

    candidate, bilinear = accept_displacement(state(1, 0), state(2, 1), bilinear_evidence,
                                             current_ids, memory_ids, {'old': 0.})
    assert bilinear['accepted_fraction'] == .125
    assert function(candidate) == .125*(1+.125)
    assert bilinear_evidence(candidate)[1]['old'].value <= .02
    checks.append('actual bilinear finite update checked; not a first-order factor projection')

    def impossible(s):
        y = function(s)
        return TrainLoss(y*y, current_ids), {'old': TrainLoss((y-1)**2, memory_ids['old'])}

    restored, rejected = accept_displacement(old, state(1, 0), impossible,
                                            current_ids, memory_ids, {'old': 0.}, tolerance=0.)
    assert rejected['accepted_fraction'] == 0
    assert rejected['caller_must_restore_optimizer_and_RNG']
    assert all(torch.equal(restored[n][f], old[n][f]) for n in old for f in ('A', 'B'))
    checks.append('all-infeasible proposals return unchanged parameters; caller rollback contract explicit')
    for invalid in (TrainLoss(0., ('TEST:forbidden',), 'test'),
                    TrainLoss(0., ('TEST:forbidden',), 'train'),
                    TrainLoss(float('nan'), current_ids)):
        try:
            invalid.validate(current_ids)
        except AssertionError:
            pass
        else:
            raise AssertionError('Invalid loss evidence accepted')
    checks.append('TEST split, foreign IDs and nonfinite losses rejected')

    # Original Adam geometry warning remains a mathematical/unit control only.
    theta = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))
    new_gradient = torch.tensor([2., 1.], dtype=torch.float64)
    old_gradient = torch.tensor([1., -2.], dtype=torch.float64)
    optimizer = torch.optim.AdamW([theta], lr=1e-4, weight_decay=0.)
    theta.grad = new_gradient.clone()
    optimizer.step()
    assert float(torch.dot(new_gradient, old_gradient)) == 0
    linear = float(torch.dot(theta.detach(), old_gradient))
    assert linear > 0
    checks.append('actual CPU Adam can change old first-order loss despite zero raw gradient dot; no GPU claim')
    assert not torch.cuda.is_initialized()
    result = {'passed': True, 'CPU_only': True, 'GPU_runtime_validated': False,
              'GPU_worker_prepared': (ROOT/'analysis/onevision_two_module_pilot.py').exists(),
              'GPU_method_or_accuracy_results': False, 'checks': checks,
              'actual_adapter_layers': len(actual), 'actual_adapter_sha256': sha(path),
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
                                (Path(__file__), ROOT/'analysis/onevision_two_module_core.py', reference_path, path)},
              'toy_current_fit_choice': a, 'toy_guarded_choice': ab,
              'toy_finite_step': b, 'toy_bilinear_step': bilinear,
              'Adam_unit_old_linear_change': linear,
              'limits': ['CPU primitives only; do not prove model quality or novelty.',
                         'Toy losses are explicit quadratic unit controls, not measured model NLL.',
                         'Optimizer/RNG rollback, routed-pool integration, deterministic GPU controls and all activations remain to implement/verify.',
                         'TRAIN finite-loss checks only constrain the fixed memory/probes, not TEST or all historical data.',
                         'An entirely rejected native task must fail validity admission rather than be reported as successful CL.']}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'real_adapter_layers': len(actual),
                      'GPU_runtime_validated': False, 'quality_gain_measured': False}))


if __name__ == '__main__':
    main()
