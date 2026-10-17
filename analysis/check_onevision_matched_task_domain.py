"""Actual original six-control trainers on a CPU toy; not GPU proof."""
import contextlib, copy, io, json, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
from onevision_task_domain_intervention import Mechanism, METHODS
from onevision_research_worker import digest
import onevision_matched_task_domain as protocol


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.A = torch.nn.Linear(3, 2, bias=False); self.B = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.B.weight); self.dropout = torch.nn.Dropout(.1)
    def forward(self, value): return self.B(self.A(self.dropout(value)))


class Tiny(Mechanism):
    def __init__(self, model, directory):
        self.model = model; self.run = directory; self.monitor = SimpleNamespace(snapshot=lambda *a, **k: None)
    def loss(self, row):
        v = row['index']; inputs = torch.tensor([1., v / 10., v * v / 100.])
        return (self.model(inputs) - row['target']).square().mean(), {'target_tokens': 1}


def main():
    torch.set_num_threads(4); torch.manual_seed(42); torch.use_deterministic_algorithms(True)
    initial = Model(); rows = [{'id': f'train:{i}', 'index': i, 'target': (i % 2) / 2} for i in range(8)]
    other = [{'id': f'next:{i}', 'index': i + 8, 'target': (i % 3) / 3} for i in range(8)]
    stages, controls = {}, {}
    with tempfile.TemporaryDirectory(prefix='matched-control-cpu-') as tmp:
        for method in METHODS:
            x = Tiny(copy.deepcopy(initial), Path(tmp) / method)
            with contextlib.redirect_stdout(io.StringIO()): x.step_task('toy', 1, rows, method, [], rows, [])
            stages[method] = {k: v.detach().clone() for k, v in x.model.state_dict().items()}
            fishers = [x.fisher(rows)] if method.startswith('EWC_') else []
            with contextlib.redirect_stdout(io.StringIO()): x.step_task('toy', 2, other, method, rows, rows, fishers)
            saved = json.loads((x.run / 'training/toy/stage_02.json').read_text()); events = saved['events']
            assert len(events) == 4 and all(r['lr_used'] == 1e-4 for r in events)
            assert all(torch.isfinite(p).all() for p in x.model.parameters())
            assert saved['new_plus_replay_exposures'] == (16 if method == 'ER_LoRA' else 8)
            if method.startswith('EWC_'): assert events[0]['penalty'] == 0
            if method.startswith('anchor_gradient_'): assert sum(r['reference_extra_QA'] for r in events) == 32
            controls[method] = {'positive_lr_updates': 4, 'exposures': saved['new_plus_replay_exposures']}
        reference = stages['SeqFT_LoRA']
        assert all(torch.equal(v, stages[method][k]) for method in METHODS for k, v in reference.items())
    *_, spec = protocol.plan(); assert spec == json.loads(json.dumps(spec))
    assert len(protocol.CASES) == 48 and len({tuple(c) for c in protocol.CASES}) == 48
    assert not torch.cuda.is_initialized()
    result = {'CPU_toy_only': True, 'source_sha256': digest(Path(protocol.__file__)),
              'check_source_sha256': digest(Path(__file__)), 'all_six_stage1_weights_exact': True,
              'original_step_task_reused_without_edit': True, 'stage2_controls': controls,
              'spec_JSON_roundtrip_exact': True, 'frozen_48_cases_unique': True, 'GPU_used': False,
              'actual_OneVision_stage1_gate': 'pending'}
    (protocol.REPORT / 'CPU_TRAINER_CHECK.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
