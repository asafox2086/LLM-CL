"""CPU exact trainer/RNG/rollback tests; no OneVision performance claim."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from transformers import get_cosine_schedule_with_warmup

from onevision_two_module_core import TrainLoss
from onevision_two_module_engine import epoch, rng_state, restore_rng

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016/two_module_fullstream'


def seed():
    random.seed(42);np.random.seed(42);torch.manual_seed(42)


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.A = torch.nn.Parameter(torch.randn(2, 3)*.2)
        self.B = torch.nn.Parameter(torch.randn(2, 2)*.2)
        self.dropout = torch.nn.Dropout(.1)

    def forward(self, x):
        return self.B @ self.A @ self.dropout(x)

    def snapshot(self):
        return {'layer': {'A': self.A.detach().clone(), 'B': self.B.detach().clone()}}

    def restore(self, state):
        with torch.no_grad():
            self.A.copy_(state['layer']['A']);self.B.copy_(state['layer']['B'])


def main():
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    seed();model = Tiny();initial = model.snapshot()
    generator = torch.Generator().manual_seed(831)
    rows = [{'id': f'TRAIN:{i}', 'x': torch.randn(3, generator=generator),
             'y': torch.randn(2, generator=generator)} for i in range(33)]

    def loss(row):
        value = (model(row['x'])-row['y']).square().mean()
        return value, {'target_tokens': 2}

    # Equivalent to original Experiment.train's update operations, including the
    # incomplete final batch, gradient clipping and pre-step scheduler semantics.
    model.restore(initial);model.train();seed()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0)
    schedule = get_cosine_schedule_with_warmup(optimizer, 0, 3)
    reference_events = []
    for start in range(0, 33, 16):
        optimizer.zero_grad(set_to_none=True);batch = rows[start:start+16];values = []
        for row in batch:
            value, _ = loss(row);(value/len(batch)).backward();values.append(float(value.detach()))
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        used = optimizer.param_groups[0]['lr'];optimizer.step();schedule.step()
        reference_events.append((sum(values)/len(values), float(norm), used))
    reference = model.snapshot();del optimizer, schedule
    checks = []

    def run(use_B=False, evaluate=None, callback=None, count=33):
        model.restore(initial);model.train();seed()
        return epoch(list(model.named_parameters()), rows[:count], loss, model.snapshot, model.restore,
                     evaluate=evaluate, current_ids=('TRAIN:probe',),
                     memory_ids={'old': ('TRAIN:old',)}, references={'old': 0.},
                     use_B=use_B, on_update=callback)

    plain = run()
    assert all(torch.equal(model.snapshot()['layer'][f], reference['layer'][f]) for f in ('A', 'B'))
    assert [(e['loss'], e['gradient_l2_preclip'], e['actual_learning_rate_used']) for e in plain['events']] == reference_events
    assert plain['effective_updates'] == 3
    checks.append('exact all-parameter/loss/norm/LR equivalence to reference epoch33QA,including partial batch')

    def random_logger(event, raw, payload):
        assert raw.numel() == sum(p.numel() for p in model.parameters())
        assert all(torch.isfinite(v).all() for v in payload['proposed']['layer'].values())
        torch.rand(67);random.random();np.random.random(39)

    run(callback=random_logger)
    assert all(torch.equal(model.snapshot()['layer'][f], reference['layer'][f]) for f in ('A', 'B'))
    checks.append('random-consuming logging cannot perturb optimizer/dropout trajectory')

    def sham(state):
        torch.rand(23);random.random();np.random.random(13)
        return TrainLoss(0., ('TRAIN:probe',)), {'old': TrainLoss(0., ('TRAIN:old',))}

    guarded = run(True, sham, random_logger)
    assert all(e['decision']['accepted_fraction'] == 1 for e in guarded['events'])
    assert all(torch.equal(model.snapshot()['layer'][f], reference['layer'][f]) for f in ('A', 'B'))
    checks.append('sham accepted protection exactly equals control despite RNG-consuming callbacks')

    def reject(state):
        torch.rand(19);random.random();np.random.random(5)
        return TrainLoss(0., ('TRAIN:probe',)), {'old': TrainLoss(1., ('TRAIN:old',))}

    rejected = run(True, reject, random_logger)
    assert rejected['effective_updates'] == 0 and not rejected['optimizer_state']['state']
    assert all(torch.equal(model.snapshot()['layer'][f], initial['layer'][f]) for f in ('A', 'B'))
    checks.append('all-infeasible steps restore exact parameters and empty original optimizer moments')
    seed();expected_rng = rng_state()
    assert torch.equal(rejected['RNG_after']['torch'], expected_rng['torch'])
    assert rejected['RNG_after']['python'] == expected_rng['python']
    assert np.array_equal(rejected['RNG_after']['numpy'][1], expected_rng['numpy'][1])
    checks.append('rejected proposals and random callbacks preserve original Python/NumPy/Torch RNG')

    def finite(state):
        distance = sum(float((state['layer'][f]-initial['layer'][f]).double().square().sum()) for f in ('A', 'B'))
        return TrainLoss(0., ('TRAIN:probe',)), {'old': TrainLoss(distance*1e6, ('TRAIN:old',))}

    shortened = run(True, finite, count=10)
    fraction = shortened['events'][0]['decision']['accepted_fraction']
    assert 0 < fraction < 1 and shortened['effective_updates'] == 1
    assert finite(model.snapshot())[1]['old'].value <= .02
    checks.append('nonzero shortened actual finite step satisfies fixed old-loss budget')
    assert all(e['actual_learning_rate_used'] > 0 for e in plain['events'])
    checks.append('all three real pre-step learning rates positive; post-final scheduler zero not mislabeled')
    assert not torch.cuda.is_initialized()
    result = {'passed': True, 'CPU_only': True, 'checks': checks, 'original_GPU_trainer_equivalence_pending': True,
              'full19_GPU_states_pending': True, 'quality_gain_measured': False,
              'shortened_fraction_toy': fraction, 'reference_steps': reference_events,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  (Path(__file__), ROOT/'analysis/onevision_two_module_engine.py', ROOT/'analysis/onevision_two_module_core.py')}}
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT/'CPU_ENGINE_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'passed': True, 'checks': len(checks), 'shortened_fraction': fraction,
                      'GPU_used': False, 'full_CL_results': False}))


if __name__ == '__main__':
    main()
