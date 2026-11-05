"""Compare four native F1 arms after x2 completion; no model or rescoring."""
from pathlib import Path
import hashlib
import json
import math
import statistics

R = Path('/data2/liyapeng_grp/program/LLMCL')
S = R / 'summary/routing_strength_f1_20261104'
O = R / 'exp/routing_strength_f1_20261104/runs/strength2_v1'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def pair(a, b):
    assert a.keys() == b.keys()
    diffs = [a[k]['F1'] - b[k]['F1'] for k in a]
    return dict(delta_F1=statistics.mean(diffs), better=sum(x > 0 for x in diffs),
                worse=sum(x < 0 for x in diffs), tie=sum(x == 0 for x in diffs),
                identical_answers=sum(a[k]['prediction'] == b[k]['prediction'] for k in a))


def main():
    dest = S / 'VERIFIED_FOUR_ARM_ANALYSIS.json'
    assert not dest.exists(), 'Do not replay completed consumer'
    ex = read(S / 'EXIT.json')
    assert ex['returncode'] == 0 and not Path('/proc', str(ex['pid'])).exists()
    assert not (O / 'FAILURE.json').exists()
    assert sha(O / 'RESULT.json') == read(O / 'COMPLETE.json')['result_sha256']
    source_sha = sha(R / 'analysis/routing_strength_f1_20261104.py')
    assert source_sha == read(O / 'STARTED.json')['source_sha256'] == read(S / 'ADMISSION.json')['source_sha256']
    assert read(O / 'STARTED.json')['pid'] == ex['pid']
    for n in ['CONTROLS_COMPLETE.json', 'SCALE_ONE_CONTROLS_COMPLETE.json']:
        assert len(read(O / n)) == 5
    old = read(S / 'ORIGINAL_X1_SELECTED.json')
    assert sha(O / 'SELECTION.json') == old['selection_sha256']
    new = read(O / 'RESULT.json')
    values = {}
    for result in [old, new]:
        for row in result['records']:
            a = values.setdefault(row['arm'], {})
            assert row['id'] not in a and math.isfinite(row['F1']) and 0 <= row['F1'] <= 100
            a[row['id']] = row
        for arm, macro in result['macro_F1'].items():
            assert len(values[arm]) == 100
            assert abs(statistics.mean(x['F1'] for x in values[arm].values()) - macro) < 1e-10
    assert set(values) == {'dynamic_x1', 'fixed_x1', 'dynamic_x2', 'fixed_x2'}
    ids = set(read(O / 'SELECTION.json')['ids'])
    domains = sorted({x['domain'] for x in values['fixed_x1'].values()})
    assert len(domains) == 5
    for arm, rows in values.items():
        assert rows.keys() == ids and all(sum(x['domain'] == d for x in rows.values()) == 20 for d in domains)
        for q, row in rows.items():
            base = values['fixed_x1'][q]
            assert (row['domain'], row['task']) == (base['domain'], base['task'])
    proofs = []
    for arm in ['dynamic_x2', 'fixed_x2']:
        coverage = read(O / (arm + '_GATE_COVERAGE.json'))
        assert coverage['effective_mode'] == arm and coverage['coefficient_multiplier'] == 2
        report = coverage['report']
        assert report['fixed_main_coverage_complete'] and report['AST_proof']['roundtrip_original_AST_equal']
        assert len(report['fixed_per_layer']) == 168
        paths = list((O / arm).rglob('*_test_predictions.jsonl'))
        assert len(paths) == 100
        seen = set()
        for p in paths:
            v = read(p); q = v['id']; assert q not in seen; seen.add(q)
            row = values[arm][q]
            assert v['prediction'] == row['prediction'] and v['scores']['token_f1'] == row['F1']
            assert v['references'] == values['fixed_x1'][q]['references']
            scores = read(p.with_suffix('.scores.json'))
            assert scores['count'] == 1 and scores['token_f1'] == row['F1']
            proofs.append(dict(path=str(p), sha256=sha(p)))
        assert seen == ids
    comparisons = {name: pair(values[a], values[b]) for name, a, b in [
        ('dynamic_minus_fixed_x1', 'dynamic_x1', 'fixed_x1'),
        ('dynamic_minus_fixed_x2', 'dynamic_x2', 'fixed_x2'),
        ('dynamic_x2_minus_x1', 'dynamic_x2', 'dynamic_x1'),
        ('fixed_x2_minus_x1', 'fixed_x2', 'fixed_x1')]}
    domain_scores = {a: {d: statistics.mean(v['F1'] for v in rows.values() if v['domain'] == d) for d in domains} for a, rows in values.items()}
    output = dict(macro_F1={a: statistics.mean(v.values()) for a, v in domain_scores.items()}, domain_F1=domain_scores,
                  comparisons=comparisons, difference_in_differences= comparisons['dynamic_minus_fixed_x2']['delta_F1'] - comparisons['dynamic_minus_fixed_x1']['delta_F1'],
                  result_sha256=sha(O / 'RESULT.json'), original_x1_sha256=sha(S / 'ORIGINAL_X1_SELECTED.json'),
                  source_sha256=source_sha, exit_sha256=sha(S / 'EXIT.json'), source_proofs=proofs,
                  rescored=False, scope='same 100 prespecified TEST QA; all-expert coefficient multiplier 1 versus 2',
                  limits=['single multiplier, not an optimal-scale search', 'not newest-expert weighting', 'no data-size causal intervention'])
    with dest.open('x') as f:
        json.dump(output, f, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in output.items() if k != 'source_proofs'}))


if __name__ == '__main__':
    main()
