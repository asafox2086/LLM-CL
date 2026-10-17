"""Audit durable TRAIN updates without publishing incomplete clinical cases."""
import hashlib, json, math, sys
from pathlib import Path
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_research_worker import digest
from geometry import inner
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'


def load_verified(path):
    meta = json.loads(path.with_suffix('.json').read_text())
    assert digest(path) == meta['sha256'] and path.stat().st_size == meta['bytes']
    return torch.load(path, map_location='cpu', weights_only=True), meta['sha256']


def main():
    torch.set_num_threads(4); output = []; sources = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    for pointer in sorted((REPORT / 'knowledge_fit').glob('*_attempt.json')):
        info = json.loads(pointer.read_text()); spec = info['spec']; run = ROOT / info['run']
        for p, h in spec['source_sha256'].items(): assert digest(ROOT / p) == h
        available = [step for step in [8, 64] if (run / f'update_{step:02d}.json').exists()]
        if not available: continue
        endpoint = max(available); start, sha = load_verified(run / 'update_00.pt')
        sources[str((run / 'update_00.pt').relative_to(ROOT))] = sha
        actual, sha = load_verified(run / f'update_{endpoint:02d}.pt')
        sources[str((run / f'update_{endpoint:02d}.pt').relative_to(ROOT))] = sha
        assert actual['updates'] == endpoint and start['updates'] == 0
        a, b = start['adapter'], actual['adapter']; assert a.keys() == b.keys()
        events = [json.loads(s) for s in (run / 'training.jsonl').read_text().splitlines()[:endpoint]]
        assert [r['step'] for r in events] == list(range(1, endpoint + 1))
        expected_ids = ['external-KG:' + k for k in spec['fact_ids']]
        gradients = []
        for record in events:
            step = record['step']; path = run / 'raw_gradients' / f'update_{step:02d}.pt'
            raw, sha = load_verified(path); sources[str(path.relative_to(ROOT))] = sha
            epoch = (step - 1) // 2; offset = ((step - 1) % 2) * 8
            order = sorted(expected_ids, key=lambda k: hashlib.sha256(f'42:{epoch}:{k}'.encode()).digest())
            assert raw['TRAIN_fact_ids'] == order[offset:offset + 8]
            vector = raw['preclip_gradient'].double(); assert torch.isfinite(vector).all()
            count = sum(math.prod(shape) for _, shape in raw['layout']); assert count == vector.numel()
            norm = float(torch.linalg.vector_norm(vector)); error = abs(norm - record['gradient_L2_preclip']) / max(norm, 1e-12)
            assert error < 1e-5 and record['lr_used'] == 1e-4 and record['cumulative_auxiliary_QA'] == step * 8
            position = 0; norms = {'A': 0., 'B': 0.}
            for name, shape in raw['layout']:
                size = math.prod(shape); factor = name.rsplit('.', 2)[-2]; assert factor in norms
                norms[factor] += float(vector[position:position + size].square().sum()); position += size
            gradients.append({'step': step, 'parameters': count, 'L2': norm, 'relative_error': error,
                              'A_gradient_L2': math.sqrt(norms['A']), 'B_gradient_L2': math.sqrt(norms['B'])})
        displacement = {f: math.sqrt(sum(float((b[k][f].double() - a[k][f].double()).square().sum()) for k in a)) for f in ['A', 'B']}
        squared = sum(inner(b[k], b[k]) + inner(a[k], a[k]) - 2 * inner(a[k], b[k]) for k in a)
        assert squared >= -1e-8
        if run.parts[-2].startswith('rank1'):
            assert gradients[0]['A_gradient_L2'] == 0 and gradients[0]['B_gradient_L2'] > 0
        output.append({'case': run.parts[-2], 'run': info['run'], 'durable_updates': endpoint, 'auxiliary_exposures': endpoint * 8,
                       'parameter_displacement_L2': displacement, 'actual_deltaW_displacement_Frobenius': 4 * math.sqrt(max(0, squared)),
                       'gradients': gradients, 'clinical_case_complete': (REPORT / 'knowledge_fit' / (run.parts[-2] + '_complete.json')).exists()})
    assert not torch.cuda.is_initialized()
    value = {'sources_sha256': sources, 'TRAIN_only': True, 'cases': output,
             'scope': 'durable checkpoint prefix, not completion of a64-update case; no clinical or recall scores',
             'limitation': 'positive LR, parameter change and nonzero deltaW do not prove successful fact encoding or downstream benefit'}
    (REPORT / 'KNOWLEDGE_TRAINING_AUDIT.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'audited_prefixes': [{k: r[k] for k in ['case', 'durable_updates', 'parameter_displacement_L2', 'actual_deltaW_displacement_Frobenius']} for r in output], 'GPU_used': False}))


if __name__ == '__main__': main()
