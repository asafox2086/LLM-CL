"""Read-only audit of durable knowledge-fit cases; never select rank/budget.

An unfinished case is excluded, even if an endpoint file exists. Raw gradients
are measured on CPU; no model loads onto CUDA and no active runner is edited.
"""
import json, math, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_knowledge_pair_audit import paired_cluster
from onevision_knowledge_encoding_probe import fact_rows
from onevision_research_worker import digest
from run import freeze, score
from geometry import inner

REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
CASES = ['rank1_correct', 'rank8_correct', 'rank8_permuted', 'rank1_permuted']


def read(p):
    return json.loads(p.read_text())


def records(p):
    return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def auxiliary(row, references):
    text = row['prediction'].strip()
    if text.lower().startswith('the answer is '):
        text = text[len('the answer is '):].strip()
    return score(text, references)['exact_match']


def main():
    proofs = {name: REPORT / 'knowledge_fit' / (name + '_complete.json') for name in CASES}
    available = {name: p for name, p in proofs.items() if p.exists()}
    if not available:
        print('Pending: no durable completed knowledge-fit case; no scores published.')
        return
    torch.set_num_threads(4)
    _, data, _, tests, _, _, _ = freeze()
    clinical = {'medical': tests['medical'], 'medical_KG': data[11, 'test']}
    pool_path = ROOT / 'data/OneVisionMechanisms/20261016/knowledge16.json'
    pool = read(pool_path)
    sources = {str(pool_path.relative_to(ROOT)): digest(pool_path),
               str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    summaries, audits, cached = [], [], {}
    for name, path in available.items():
        proof = read(path)
        assert proof['complete'] and not proof['formal_full_test']
        spec = proof['spec']; run = ROOT / proof['run']
        assert spec['protocol_version'] == 3 and spec['pool_sha256'] == digest(pool_path)
        assert spec['auxiliary_exposures_at64'] == spec['updates'] * spec['batch_QA'] == 512
        assert [c[0] for c in spec['cases']] == CASES
        sources[str(path.relative_to(ROOT))] = digest(path)
        for p, expected in spec['source_sha256'].items():
            assert digest(ROOT / p) == expected
        for p, expected in proof['artifact_sha256'].items():
            assert digest(run / p) == expected
        required = [*run.rglob('*.pt'), *run.rglob('*.jsonl'), run / 'protocol.json',
                    run / 'memory_summary.json', run / 'initial_function_gate.json']
        assert all(str(p.relative_to(run)) in proof['artifact_sha256'] for p in required)
        assert read(run / 'initial_function_gate.json')['passed']
        assert read(run / 'protocol.json') == spec
        states = {}
        for step in [0, 8, 64]:
            state = torch.load(run / f'update_{step:02d}.pt', map_location='cpu', weights_only=True)
            assert state['updates'] == step
            states[step] = state['adapter']
        displacement = {}
        initial = states[0]
        for step in [8, 64]:
            actual = states[step]; assert actual.keys() == initial.keys()
            squared = sum(inner(actual[k], actual[k]) + inner(initial[k], initial[k])
                          - 2 * inner(actual[k], initial[k]) for k in initial)
            assert squared >= -1e-8
            displacement[str(step)] = 4 * math.sqrt(max(0, squared))
        training = records(run / 'training.jsonl')
        assert [r['step'] for r in training] == list(range(1, 65))
        norm_errors = []
        for row in training:
            assert row['lr_used'] == 1e-4 and row['cumulative_auxiliary_QA'] == row['step'] * 8
            assert math.isfinite(row['pre_update_mean_loss'])
            raw = torch.load(run / 'raw_gradients' / f"update_{row['step']:02d}.pt", map_location='cpu', weights_only=True)
            v = raw['preclip_gradient']; assert torch.isfinite(v).all()
            assert v.numel() == sum(math.prod(shape) for _, shape in raw['layout'])
            norm = float(torch.linalg.vector_norm(v.double()))
            error = abs(norm - row['gradient_L2_preclip']) / max(norm, 1e-12)
            assert error < 1e-5
            norm_errors.append(error)
        for step in [0, 8, 64]:
            folder = run / 'evaluations' / f'update_{step:02d}'
            cohorts = {}
            for cohort in ['relevant_true', 'own_target']:
                kind = 'permuted' if name.endswith('permuted') and cohort == 'own_target' else 'correct'
                expected = fact_rows(pool[kind], cohort)
                rows = records(folder / (cohort + '_text.jsonl')); cohorts[cohort] = rows
                assert [r['id'] for r in rows] == [r['id'] for r in expected]
                assert [r['reference'] for r in rows] == [r['answer'] for r in expected]
                for template in ['original', 'paraphrase']:
                    group = [r for r in rows if r['template'] == template]; assert len(group) == 16
                    summaries.append({'case': name, 'step': step, 'group': cohort + ':' + template,
                                      'QA': 16, 'EM_correct': sum(r['scores']['exact_match'] == 100 for r in group),
                                      'prefix_aux_correct': sum(auxiliary(r, [r['reference']]) == 100 for r in group),
                                      'mean_NLL': float(np.mean([r['nll'] for r in group]))})
            assert [r['prediction'] for r in cohorts['relevant_true']] == [r['prediction'] for r in cohorts['own_target']], 'Changing reference alone changed generation'
            for group, expected in clinical.items():
                rows = records(run / 'evaluations' / f'clinical_update_{step:02d}' / (group + '_image.jsonl'))
                assert [r['id'] for r in rows] == spec['clinical_ids'][group] == [r['id'] for r in expected]
                assert all(r['references'] == e['answers'] for r, e in zip(rows, expected))
                cached[name, step, group] = rows
                summaries.append({'case': name, 'step': step, 'group': group, 'QA': len(rows),
                                  'EM_correct': sum(r['scores']['exact_match'] == 100 for r in rows),
                                  'prefix_aux_correct': sum(auxiliary(r, e['answers']) == 100 for r, e in zip(rows, expected)),
                                  'mean_NLL': float(np.mean([r['nll'] for r in rows]))})
        resources = records(run / 'resources.jsonl')
        windows = {}
        for window in [1, 2]:
            phases = [f'fit_window_{window}_before', f'fit_window_{window}_after']
            points = [next(r for r in resources if r['phase'] == phase) for phase in phases]
            assert points[0]['pid'] == points[1]['pid'] and points[0]['unix'] < points[1]['unix']
            windows[str(window)] = dict(zip(['before', 'after'], points))
        audits.append({'case': name, 'raw_gradient_checks': 64, 'max_norm_relative_error': max(norm_errors),
                       'actual_deltaW_displacement_Frobenius': displacement,
                       'training_windows_bytes': windows, 'overall_memory': read(run / 'memory_summary.json')})
    pairs = []
    for rank in [1, 8]:
        correct, wrong = f'rank{rank}_correct', f'rank{rank}_permuted'
        if correct not in available or wrong not in available:
            continue
        for step in [0, 8, 64]:
            for group, expected in clinical.items():
                a, b = cached[correct, step, group], cached[wrong, step, group]
                pairs.append({'rank': rank, 'step': step, 'group': group, 'difference': 'correct minus permuted',
                              'EM_pp': paired_cluster([x['scores']['exact_match'] - y['scores']['exact_match'] for x, y in zip(a, b)], [tuple(r['images']) for r in expected]),
                              'NLL': paired_cluster([x['nll'] - y['nll'] for x, y in zip(a, b)], [tuple(r['images']) for r in expected])})
    result = {'completed_cases': list(available), 'pending_cases': [n for n in CASES if n not in available],
              'sources_sha256': sources, 'summaries': summaries, 'audits': audits, 'paired_content_controls': pairs,
              'limitations': ['medical100 and KG267 overlap12, never pool as367 independent QA',
                              '16 facts and their paraphrases are not32 independent facts',
                              'synthetic wrong-target recall measures memorization, not factual accuracy',
                              'fixed-prefix sensitivity is auxiliary; primary scores unchanged',
                              'single seed and frozen small pool; no rank/budget choice from test scores',
                              '512 auxiliary exposures are additional to target CL; not full21454 confirmation']}
    assert not torch.cuda.is_initialized()
    (REPORT / 'KNOWLEDGE_FIT_EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed_cases': list(available), 'summary_cells': len(summaries), 'GPU_used': False}))


if __name__ == '__main__':
    main()
