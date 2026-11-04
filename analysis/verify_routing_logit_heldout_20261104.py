"""Verify completed paired TEST diagnostic; no model, generation, or rescoring."""
from pathlib import Path
import hashlib
import json
import math
import statistics

R = Path('/data2/liyapeng_grp/program/LLMCL')
O = R / 'exp/routing_logit_heldout_20261104/runs/confirmation_v1'
S = R / 'summary/routing_logit_heldout_20261104'
SOURCE_SHA = 'edb40f493502e2116182976696e3790ff72b11ea11cad86202e9fe80e50fc493'


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def aggregate(rows):
    den = math.fsum(x['fixed_minus_off_centered_logit_squared'] for x in rows)
    num = math.fsum(x['dynamic_minus_fixed_centered_logit_squared'] for x in rows)
    assert den > 0 and num >= 0
    return dict(
        QA=len(rows), positions=sum(len(x['selected_target_positions']) for x in rows),
        centered_logit_energy_ratio=num / den, relative_centered_logit_L2=math.sqrt(num / den),
        fixed_off_top1_changes=sum(x['fixed_off_top1_changes'] for x in rows),
        dynamic_fixed_top1_changes=sum(x['dynamic_fixed_top1_changes'] for x in rows),
        gold_top1_counts={a: sum(x['gold_top1_counts'][a] for x in rows) for a in ['off', 'fixed', 'dynamic']},
        fixed_off_gold_margin_abs_QA_mean=statistics.mean(x['fixed_off_gold_margin_abs_mean'] for x in rows),
        dynamic_fixed_gold_margin_abs_QA_mean=statistics.mean(x['dynamic_fixed_gold_margin_abs_mean'] for x in rows),
        QA_with_dynamic_fixed_top1_change=sum(x['dynamic_fixed_top1_changes'] > 0 for x in rows),
        QA_with_fixed_off_top1_change=sum(x['fixed_off_top1_changes'] > 0 for x in rows),
    )


def main():
    dest = S / 'VERIFIED_ANALYSIS.json'
    assert not dest.exists(), 'Do not replay completed consumer'
    ex = read(S / 'EXIT.json')
    assert ex['returncode'] == 0 and not Path('/proc', str(ex['pid'])).exists()
    assert not (O / 'FAILURE.json').exists()
    assert sha(O / 'RESULT.json') == read(O / 'COMPLETE.json')['result_sha256']
    assert sha(R / 'analysis/routing_logit_heldout_20261104.py') == SOURCE_SHA
    started = read(O / 'STARTED.json')
    assert started['source_sha256'] == SOURCE_SHA and started['pid'] == ex['pid']
    assert len(read(O / 'CONTROLS_COMPLETE.json')) == 5
    loaded = read(O / 'MODEL_LOADED.json')
    assert loaded['QA'] == 100 and loaded['arms'] == ['off', 'fixed', 'dynamic']
    assert loaded['checkpoint_sha256'] == '8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4'
    selection = read(O / 'SELECTION.json')
    full_binding = read(R / 'summary/cl_fixed_unlimited_20261103/BINDING.json')
    assert selection['fixed_source'] == full_binding['old_TF_c_path']
    assert sha(Path(selection['fixed_source'])) == selection['fixed_source_sha256']
    result = read(O / 'RESULT.json')
    assert result['TRAIN_only'] is False and result['teacher_forced'] is True
    rows = result['records']
    assert len(rows) == 100 and len({x['id'] for x in rows}) == 100
    assert selection['ids'] == [x['id'] for x in rows]
    domains = sorted({x['domain'] for x in rows})
    assert len(domains) == 5 and all(sum(x['domain'] == d for x in rows) == 20 for d in domains)
    train = read(R / 'exp/routing_logit_mechanism_20261104/runs/diagnostic_v1/RESULT.json')
    assert not ({x['id'] for x in rows} & {x['id'] for x in train['records']})
    for i, x in enumerate(rows):
        assert read(O / ('QA_%02d.json' % i)) == x
        n = x['target_count']
        expected = sorted(set(round(j * (n - 1) / max(1, min(32, n) - 1)) for j in range(min(32, n))))
        assert x['selected_target_positions'] == expected
        for a in ['off', 'fixed', 'dynamic']:
            assert len(x['gold_margin'][a]) == len(expected)
            assert all(math.isfinite(v) for v in x['gold_margin'][a])
            assert 0 <= x['gold_top1_counts'][a] <= len(expected)
        for k in ['fixed_off_top1_changes', 'dynamic_fixed_top1_changes']:
            assert 0 <= x[k] <= len(expected)
    pooled = aggregate(rows)
    for key in ['centered_logit_energy_ratio', 'relative_centered_logit_L2', 'fixed_off_top1_changes', 'dynamic_fixed_top1_changes']:
        assert math.isclose(pooled[key], result[key], rel_tol=1e-12, abs_tol=1e-12)
    assert pooled['positions'] == result['sampled_answer_positions']
    report = dict(
        result_sha256=sha(O / 'RESULT.json'), exit_sha256=sha(S / 'EXIT.json'),
        source_sha256=SOURCE_SHA, selection_sha256=sha(O / 'SELECTION.json'),
        pooled=pooled,
        domains={d: aggregate([x for x in rows if x['domain'] == d]) for d in domains},
        scope='100 TEST QA; same fixed coefficients as complete TEST; paired gold-prefix diagnostic',
        limits=['teacher-forced sampled positions, not free-generation F1',
                'does not identify training cause or establish full-TEST mediation',
                'sample from existing TEST, not a new external test set'],
    )
    with dest.open('x') as f:
        json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
