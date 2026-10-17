"""Synthetic collector integrity check; no real experiment outcomes."""
import contextlib, io, json, tempfile
from pathlib import Path
import torch
import onevision_rural_weight_evidence as audit
from run import score


def main():
    source = Path(audit.__file__)
    with tempfile.TemporaryDirectory(prefix='rural-weight-collector-') as tmp:
        root = Path(tmp); report = root / 'report'; (report / 'rural_weights').mkdir(parents=True)
        def put(p, value):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value) + '\n')
        def lines(p, rows):
            p.parent.mkdir(parents=True, exist_ok=True); p.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        def state(step): return {'adapter': {'toy': {'A': torch.tensor([[1., 2.]]), 'B': torch.tensor([[step / 4.]])}}}
        groups = {part: [{'id': f'{part}:{i}', 'answer': 'urban' if i < urban else 'rural',
                          'answers': ['urban' if i < urban else 'rural'], 'images': [f'{part}-{i}.png']}
                         for i in range(total)] for part, total, urban in [('TRAIN', 6, 4), ('TEST', 100, 44)]}
        audit.ROOT = root; audit.REPORT = report; audit.__file__ = str(root / 'collector.py')
        Path(audit.__file__).write_bytes(source.read_bytes())
        audit.freeze = lambda: (None, {(12, 'train'): groups['TRAIN'], (12, 'test'): groups['TEST']}, None, None, None, None, {'source_manifest_sha256': 'synthetic'})
        spec = {'cases': audit.CASES, 'updates': 4, 'manifest_sha256': 'synthetic', 'source_sha256': {},
                'train_ids': [r['id'] for r in groups['TRAIN']], 'test_ids': [r['id'] for r in groups['TEST']], 'basis_case': 'basis'}
        def prediction(r): return {'id': r['id'], 'references': r['answers'], 'prediction': 'urban', 'nll': 1., 'scores': score('urban', r['answers'])}
        basis = root / 'basis'; basis.mkdir()
        for step in [1, 4]:
            torch.save(state(step), basis / f'update_{step}.pt')
            lines(basis / 'evaluations' / f'endpoint_{step}' / 'native_image.jsonl', [prediction(r) for r in groups['TEST']])
        markers = []
        for name in audit.CASES:
            run = root / name; run.mkdir(); put(run / 'protocol.json', spec)
            put(run / 'memory_summary.json', {'scope': 'synthetic training only'})
            put(run / 'training.json', {'records': [{'update': i, 'gradient_l2_preclip': 5., 'learning_rate_used': 1e-4, 'cumulative_QA_exposures': i * 6} for i in range(1, 5)]})
            for step in range(1, 5):
                p = run / 'raw_gradients' / f'update_{step}.pt'; p.parent.mkdir(exist_ok=True)
                torch.save({'gradient_preclip': torch.tensor([3., 4.]), 'layout': [('toy', [2])]}, p)
            torch.save(state(0), run / 'start.pt')
            for step in [1, 4]:
                torch.save(state(step), run / f'update_{step}.pt')
                for group, rows in groups.items(): lines(run / 'evaluations' / f'endpoint_{step}' / (group + '_image.jsonl'), [prediction(r) for r in rows])
            paths = [*run.rglob('*.pt'), *run.rglob('*.jsonl'), run / 'training.json', run / 'memory_summary.json', run / 'protocol.json']
            marker = report / 'rural_weights' / (name + '_complete.json'); markers.append(marker)
            put(marker, {'complete': True, 'uniform_exact_gate': True, 'full21454': False, 'run': name, 'spec': spec,
                         'artifact_sha256': {str(p.relative_to(run)): audit.digest(p) for p in paths}})
        with contextlib.redirect_stdout(io.StringIO()): audit.main()
        result = audit.read(report / 'RURAL_WEIGHT_EVIDENCE.json')
        assert len(result['summaries']) == 36 and len(result['paired_differences']) == 12 and len(result['raw_gradient_audit']) == 12
        assert all(r['EM_correct'] == 44 for r in result['summaries'] if r['split'] == 'TEST' and r['gold_class'] == 'all')
        damaged = root / 'uniform/training.json'; damaged.write_text('{}\n')
        try: audit.main()
        except AssertionError: pass
        else: raise AssertionError('Artifact tampering accepted')
        for p in markers: p.unlink()
        (report / 'RURAL_WEIGHT_EVIDENCE.json').unlink()
        with contextlib.redirect_stdout(io.StringIO()): audit.main()
        assert not (report / 'RURAL_WEIGHT_EVIDENCE.json').exists() and not torch.cuda.is_initialized()
    print(json.dumps({'synthetic_only': True, 'collector_sha256': audit.digest(source), 'all_three_cases_pass': True,
                      'summary_cells': 36, 'paired_cells': 12, 'tampering_rejected': True, 'missing_completion_no_scores': True, 'GPU_used': False}))


if __name__ == '__main__': main()
