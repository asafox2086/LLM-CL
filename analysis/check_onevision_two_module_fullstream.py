"""CPU data/proof/recovery gates; no OneVision model loading or GPU work."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import json
import tempfile
from pathlib import Path

import torch

import onevision_two_module_fullstream as stream
from onevision_research_worker import atomic_torch, digest
from run import write


def rejected(function):
    try:
        function()
    except (AssertionError, ValueError):
        return True
    return False


def main():
    checks = []
    manifest, data, domains, tests, generation, current, memories, spec = stream.plan()
    assert len(manifest['tasks']) == 19 and spec['target_TRAIN_QA'] == 1313
    assert sum(len(data[t, 'test']) for t in range(1, 20)) == 21454
    assert sum(map(len, tests.values())) == 500 and sum(map(len, generation.values())) == 155
    for stage in range(1, 20):
        allowed = set(spec['TRAIN_ids'][str(stage)])
        assert set(spec['current_probe_ids'][str(stage)]) <= allowed
        assert set(spec['old_memory_ids'][str(stage)]) <= allowed
        assert set(spec['training_order_ids'][str(stage)]) == allowed
    checks.append('Frozen19 stages/1313TRAIN/500NLL/155generation/21454TEST, all probe IDs TRAIN-only')
    with tempfile.TemporaryDirectory(prefix='llmcl-fullstream-cpu-') as temp:
        directory = Path(temp);path = directory/'snapshot.pt'
        atomic_torch(path, {'adapter': {'test': {'A': torch.ones(1), 'B': torch.zeros(1)}}, 'stage': 1})
        assert stream.recoverable_checkpoint(path) and stream.checkpoint(path)['stage'] == 1
        original = path.read_bytes();path.write_bytes(original+b'bad')
        assert rejected(lambda: stream.recoverable_checkpoint(path))
        path.write_bytes(original);path.with_suffix('.json').unlink()
        assert not stream.recoverable_checkpoint(path)
        checks.append('SHA-valid checkpoint loads; corrupted stops; missing atomic sidecar is unproven')

        class Model:
            def eval(self):
                pass

        class Fake:
            def __init__(self):
                self.run = directory;self.model = Model();self.calls = 0

            def evaluate(self, label, selected, generated):
                self.calls += 1
                folder = self.run/'evaluations'/label
                for domain, expected in selected.items():
                    output = folder/f'{domain}_image.jsonl'
                    existing = [json.loads(s) for s in output.read_text().splitlines()] if output.exists() else []
                    for row in expected[len(existing):]:
                        value = {'id': row['id'], 'references': row['answers'], 'nll': 1.,
                                 'prediction': row['answers'][0], 'scores': {'token_f1': 100.}}
                        existing.append(value)
                    output.write_text(''.join(json.dumps(v)+'\n' for v in existing))
                write(folder/'durable_complete.json', {'complete': True,
                      'files': {f'{d}_image.jsonl': digest(folder/f'{d}_image.jsonl') for d in selected}})

        fake = Fake();selected = {'test': [{'id': 'test:a', 'answers': ['x']}, {'id': 'test:b', 'answers': ['y']}]}
        small_spec = {'source_sha256': {'fake': 'fixed'}}
        stream.evaluate(fake, 'resume', selected, selected, 'same-state', small_spec)
        stream.evaluate(fake, 'resume', selected, selected, 'same-state', small_spec)
        assert fake.calls == 1
        assert rejected(lambda: stream.evaluate(fake, 'resume', selected, selected, 'changed-state', small_spec))
        checks.append('Complete same-state inference skipped; changed-state prediction splice rejected')
        folder = directory/'evaluations/resume';output = folder/'test_image.jsonl'
        values = output.read_text().splitlines();output.write_text(values[0]+'\n{"interrupted":')
        (folder/'durable_complete.json').unlink()
        stream.evaluate(fake, 'resume', selected, selected, 'same-state', small_spec)
        assert fake.calls == 2 and list((folder/'interrupted_tail').iterdir())
        checks.append('Only incomplete last JSONL line quarantined; same-weight original ID prefix resumed')
        changed = dict(selected);changed['test'] = list(reversed(selected['test']))
        assert rejected(lambda: stream.evaluate(fake, 'resume', changed, changed, 'same-state', small_spec))
        rows = [json.loads(s) for s in output.read_text().splitlines()];rows[0]['references'] = ['wrong']
        output.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        write(folder/'durable_complete.json', {'files': {'test_image.jsonl': digest(output)}})
        assert rejected(lambda: stream.check_evaluation(directory, 'resume', selected, selected))
        checks.append('Changed ID order and changed gold references rejected even with internally valid file SHA')
        rows[0]['references'] = ['x'];rows[0]['scores']['token_f1'] = 99.
        output.write_text(''.join(json.dumps(v)+'\n' for v in rows))
        write(folder/'durable_complete.json', {'files': {'test_image.jsonl': digest(output)}})
        assert rejected(lambda: stream.check_evaluation(directory, 'resume', selected, selected))
        checks.append('Stored primary score must exactly equal existing AnswerF1 recomputation')
    assert not torch.cuda.is_initialized()
    report = {'passed': True, 'CPU_only': True, 'checks': checks,
              'actual19stage_GPU_equivalence_pending': True, 'full21454_GPU_confirmation_pending': True,
              'sources': {str(p.relative_to(stream.ROOT)): digest(p) for p in [Path(__file__), Path(stream.__file__)]}}
    write(stream.REPORT/'CPU_WORKER_VALIDATION.json', report)
    print(json.dumps({'passed': True, 'checks': len(checks), 'GPU_used': False}))


if __name__ == '__main__':
    main()
