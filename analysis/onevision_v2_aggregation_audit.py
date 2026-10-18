"""Recompute four fixed aggregation rules on completed V2 final predictions.

CPU only. Original Answer F1 and primary AP19 remain unchanged. Additional
equal-task-within-domain bridge isolates aggregation weights, not a new metric.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(completion_path):
    proof = json.loads(completion_path.read_text())
    assert proof['CPU_only'] and proof['raw_Answer_F1'] and not proof['TEST_used_to_select_method']
    assert [r['method'] for r in proof['results']] == ['SeqFT_LoRA', 'ER_LoRA']
    sources = {str(completion_path): sha(completion_path),
               'analysis/onevision_v2_aggregation_audit.py': sha(Path(__file__))}
    def checked(relative, expected=None):
        p = ROOT/relative
        h = sha(p)
        assert expected is None or h == expected, relative
        sources[str(relative)] = h
        return p
    for rel, expected in proof['source_sha256'].items():
        checked(rel, expected)
    metric_lock = json.loads(checked('summary/primary_metric_20261017/metric_lock.json').read_text())
    metric_path = checked('analysis/answer_f1.py', metric_lock['metric_sha256'])
    assert metric_lock['option_mapping'] is False and metric_lock['additional_prefix_or_answer_extraction'] is False
    spec = importlib.util.spec_from_file_location('locked_f1_aggregation', metric_path)
    metric = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metric)
    manifest_path = checked('data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    domains = list(dict.fromkeys(t['domain'] for t in manifest['tasks']))
    assert len(manifest['tasks']) == 19 and len(domains) == 5
    gold = {}
    for task in manifest['tasks']:
        entry = task['files']['test']
        gold[task['task']] = [json.loads(s) for s in checked(entry['path'], entry['sha256']).read_text().splitlines()]
        assert len(gold[task['task']]) == entry['count']
    assert sum(map(len, gold.values())) == 21454
    results = []
    for record in proof['results']:
        run = ROOT/record['run']
        assert run.resolve().is_relative_to(ROOT/'exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01')
        assert json.loads(checked(str(run.relative_to(ROOT)/'run_status.json')).read_text())['status'] == 'completed'
        assert json.loads(checked(str(run.relative_to(ROOT)/'stages/stage_19.json')).read_text())['complete']
        checked(str(run.relative_to(ROOT)/'config.json'), record['config_sha256'])
        tasks = []
        for task in manifest['tasks']:
            relative = f"evaluations/stage_19/T{task['task']:02d}_test_predictions.jsonl"
            p = checked(str(run.relative_to(ROOT)/relative), record['prediction_sha256'][relative])
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            expected_rows = gold[task['task']]
            assert len(rows) == len(expected_rows)
            values = []
            for row, expected in zip(rows, expected_rows):
                assert row['id'] == expected['id'] and row['references'] == expected['answers']
                value = metric.answer_f1(row['prediction'], expected['answers'])
                assert value == row['scores']['token_f1'] and math.isfinite(value)
                values.append(value)
            mean = math.fsum(values)/len(values)
            assert abs(mean-record['final_native19'][task['task']-1]) < 1e-10
            tasks.append({'task': task['task_id'], 'domain': task['domain'],
                          'native_label': task['native_label'], 'QA': len(values), 'Answer_F1': mean})
        dvalues = {}
        for domain in domains:
            tt = [t for t in tasks if t['domain'] == domain]
            n = sum(t['QA'] for t in tt)
            qm = math.fsum(t['QA']*t['Answer_F1'] for t in tt)/n
            tm = math.fsum(t['Answer_F1'] for t in tt)/len(tt)
            assert abs(qm-record['domains'][domain]['F1']) < 1e-10
            dvalues[domain] = {'tasks': len(tt), 'QA': n, 'QA_mean_F1': qm,
                               'task_mean_F1': tm, 'AP19_domain_weight': len(tt)/19,
                               'domain_macro_weight': .2, 'QA_micro_domain_weight': n/21454}
        aggregates = {
            'AP19_primary': math.fsum(t['Answer_F1'] for t in tasks)/19,
            'domain_task_macro5_bridge': math.fsum(v['task_mean_F1'] for v in dvalues.values())/5,
            'domain_QA_macro5_preexisting': math.fsum(v['QA_mean_F1'] for v in dvalues.values())/5,
            'QA_micro_descriptive': math.fsum(v['QA']*v['QA_mean_F1'] for v in dvalues.values())/21454}
        assert abs(aggregates['AP19_primary']-record['aggregate']['AP']) < 1e-10
        # This domain macro was already implemented; do not choose a new rule
        # because it ranks one of the observed fits more favorably.
        preexisting = json.loads(checked(str(run.relative_to(ROOT)/'metrics_by_score.json'), record['metrics_sha256']).read_text())['token_f1']
        assert abs(preexisting['domain_macro_final']-aggregates['domain_QA_macro5_preexisting']) < 1e-10
        results.append({'method': record['method'], 'run': record['run'], 'tasks': tasks,
                        'domains': dvalues, 'aggregates': aggregates})
    a, b = results
    delta = {k: b['aggregates'][k]-a['aggregates'][k] for k in a['aggregates']}
    contributions = []
    for domain in domains:
        av, bv = a['domains'][domain], b['domains'][domain]
        dt, dq = bv['task_mean_F1']-av['task_mean_F1'], bv['QA_mean_F1']-av['QA_mean_F1']
        contributions.append({'domain': domain, 'delta_domain_task_mean': dt, 'delta_domain_QA_mean': dq,
                              'between_domain_weights_contribution': (av['tasks']/19-.2)*dt,
                              'within_domain_weights_contribution': .2*(dt-dq)})
    between = math.fsum(v['between_domain_weights_contribution'] for v in contributions)
    within = math.fsum(v['within_domain_weights_contribution'] for v in contributions)
    assert abs(between-(delta['AP19_primary']-delta['domain_task_macro5_bridge'])) < 1e-12
    assert abs(within-(delta['domain_task_macro5_bridge']-delta['domain_QA_macro5_preexisting'])) < 1e-12
    for rel, expected in sources.items():
        p = Path(rel) if Path(rel).is_absolute() else ROOT/rel
        assert sha(p) == expected
    return {'CPU_only': True, 'GPU_started': False, 'TRAIN_QA_read': 0,
            'new_optimizer_updates': 0, 'new_predictions_generated': 0,
            'rescored_final_QA_each': 21454, 'same_original_F1': True,
            'primary_AP19_unchanged': True, 'conditions_or_metrics_selected_by_TEST': False,
            'source_sha256': sources, 'results': results, 'ER_minus_SeqFT': delta,
            'decomposition': {'domain_contributions': contributions,
                              'between_domain_weight_effect': between,
                              'within_domain_weight_effect': within,
                              'AP19_minus_domain_QA_macro_delta': between+within},
            'limits': ['Fixed two fitted seed42 baselines; no new training or method improvement.',
                       'Known weighted-average dependence, not knowledge/gradient/routing causality.',
                       'Four aggregation definitions reported together; primary AP19 remains locked.',
                       'Domain-QA macro already present in original metrics_by_score; bridge is diagnostic only.',
                       'Point estimates only, no new uncertainty/significance or population claim.',
                       'Formal V2 common-image protocol; not nativeAnyRes research results.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.completion)
    if args.out.exists():
        raise FileExistsError('Preserve prior evidence; use a new path.')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'methods': [r['method'] for r in result['results']],
                      'aggregates': [r['aggregates'] for r in result['results']],
                      'delta': result['ER_minus_SeqFT'], 'decomposition': result['decomposition']}))
