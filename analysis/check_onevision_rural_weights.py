"""CPU toy checks for the actual weighted trainer; not GPU results."""
import contextlib, copy, io, json, tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
import onevision_rural_weight_probe as probe


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.A = torch.nn.Linear(3, 2, bias=False); self.B = torch.nn.Linear(2, 1, bias=False)
        torch.nn.init.zeros_(self.B.weight)


class Tiny:
    def __init__(self, model, directory):
        self.model = model; self.directory = directory; self.monitor = SimpleNamespace(snapshot=lambda *a, **k: None)
    def configure(self, start, finetune_shared):
        with torch.no_grad():
            self.model.A.weight.copy_(start['toy']['A']); self.model.B.weight.copy_(start['toy']['B'])
    def snapshot(self):
        return {'toy': {'A': self.model.A.weight.detach().clone(), 'B': self.model.B.weight.detach().clone()}}
    def loss(self, row):
        v = int(row['id'][-1]); features = torch.tensor([1., v / 10., v * v / 100.])
        prediction = self.model.B(self.model.A(features)); target = 1. if row['answer'] == 'urban' else -1.
        return (prediction - target).square().mean(), {'target_tokens': 1}


def main():
    rows = [{'id': f'toy:{i}', 'answer': 'urban' if i < 4 else 'rural'} for i in range(6)]
    maps = {name: probe.weights(rows, name) for name in probe.CASES}
    assert all(abs(sum(v.values()) - 1) < 1e-12 for v in maps.values())
    assert sorted(maps['class_balanced'].values()) == sorted(maps['urban_high_negative'].values())
    assert sum(maps['class_balanced'][r['id']] for r in rows if r['answer'] == 'rural') == .5
    assert sum(maps['urban_high_negative'][r['id']] for r in rows if r['answer'] == 'rural') == .25
    torch.manual_seed(42); model = Model()
    with tempfile.TemporaryDirectory(prefix='rural-weights-cpu-') as tmp:
        directory = Path(tmp); old = Tiny(copy.deepcopy(model), directory / 'old'); start = old.snapshot()
        with contextlib.redirect_stdout(io.StringIO()): probe.train_trajectory(old, rows, start, old.directory)
        outcomes = {}
        for name in probe.CASES:
            trial = Tiny(copy.deepcopy(model), directory / name)
            probe.weighted_train(trial, rows, start, trial.directory, maps[name]); outcomes[name] = trial.snapshot()
        error = probe.weight_error(old.snapshot(), outcomes['uniform']); assert error < 1e-6
        p = directory / 'class_balanced/raw_gradients/update_1.pt'; v = torch.load(p, weights_only=True)['gradient_preclip']
        trial = Tiny(copy.deepcopy(model), directory / 'manual'); trial.configure(start, True); trial.model.zero_grad()
        for row in sorted(rows, key=lambda r: probe.hashed(r, 'training')):
            loss, _ = trial.loss(row); (loss * maps['class_balanced'][row['id']]).backward()
        expected = torch.cat([p.grad.flatten() for p in trial.model.parameters()]); assert torch.equal(v, expected)
        assert probe.weight_error(outcomes['class_balanced'], outcomes['urban_high_negative']) > 0
    assert not torch.cuda.is_initialized()
    print(json.dumps({'synthetic_toy_only': True, 'source_sha256': probe.digest(Path(probe.__file__)),
                      'uniform_four_update_weight_error': error, 'weighted_first_gradient_exact': True,
                      'matched_coefficient_multiset': True, 'negative_changes_training': True, 'GPU_used': False}))


if __name__ == '__main__': main()
