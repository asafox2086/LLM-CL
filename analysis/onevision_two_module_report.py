"""CPU-only verified reporting for fixed two-module pilots, Answer F1 only."""
import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import time
import traceback
from collections import defaultdict
from pathlib import Path

from answer_f1 import answer_f1
import onevision_two_module_queue as queue

ROOT = queue.ROOT
REPORT = queue.PILOT


def paired(values, identities, original):
    clusters = defaultdict(list)
    for identity, value in zip(identities, values):
        clusters[tuple(original[identity]['images'])].append(value)
    groups = list(clusters.values());rng = random.Random(42);samples = []
    for _ in range(2000):
        chosen = [groups[rng.randrange(len(groups))] for _ in groups]
        samples.append(sum(map(sum, chosen))/sum(map(len, chosen)))
    samples.sort()
    return {'QA': len(values), 'delta_F1_pp': sum(values)/len(values),
            'recipient_image_cluster_95CI': [samples[49], samples[1949]], 'uncorrected': True}


def build():
    _, spec = queue.plans()
    metric = json.loads((ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert queue.prior.sha(ROOT/'analysis/answer_f1.py') == metric['metric_sha256']
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert queue.prior.sha(manifest_path) == spec['manifest_sha256']
    original = {}
    for task in json.loads(manifest_path.read_text())['tasks']:
        f = task['files']['test'];path = ROOT/f['path']
        assert queue.prior.sha(path) == f['sha256']
        original.update({r['id']: r for r in map(json.loads, path.read_text().splitlines())})
    sources = {}

    def rows(run, proof, label, group, ids):
        path = run/f'evaluations/{label}/{group}_image.jsonl'
        expected = proof['artifact_sha256'][str(path.relative_to(run))]
        assert queue.prior.sha(path) == expected
        values = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in values] == ids
        generated = []
        for row in values:
            assert row['references'] == original[row['id']]['answers']
            assert math.isfinite(row['nll'])
            if 'prediction' in row:
                assert answer_f1(row['prediction'], row['references']) == row['scores']['token_f1']
                generated.append(row)
        assert generated
        sources[str(path.relative_to(ROOT))] = expected
        return values, generated

    result = [];accepted = {};all_control_initial_old = None
    for task, variant in spec['cases']:
        if not queue.pilot_complete((task, variant), spec):
            continue
        path = REPORT/f'T{task:02d}_{variant}_complete.json'
        sources[str(path.relative_to(ROOT))] = queue.prior.sha(path)
        proof = json.loads(path.read_text());run = ROOT/proof['run']
        target, target_gen = rows(run, proof, 'final', 'target', spec['target_test_ids'][str(task)])
        old, old_gen = rows(run, proof, 'final', 'past_natural', spec['old_test_ids'])
        assert [r['id'] for r in old_gen] == [i for i in spec['old_test_ids'] if i in spec['old_generation_ids']]
        training = json.loads((run/'training.json').read_text())
        model = json.loads((run/'model_counts.json').read_text())
        assert model['trainable_parameters'] == 540672 and model['retained_LoRA_modules'] == 1
        if variant == 'control':
            _, initial_old = rows(run, proof, 'initial', 'past_natural', spec['old_test_ids'])
            values = [(r['id'], r['prediction'], r['scores']['token_f1']) for r in initial_old]
            if all_control_initial_old is not None:
                assert values == all_control_initial_old, 'Common initial old-function generation differs'
            all_control_initial_old = values
        valid = proof['valid_training']
        entry = {'task': task, 'variant': variant, 'valid_training': valid,
                 'target_QA': len(target_gen), 'old_generated_QA': len(old_gen), 'old_NLL_QA': len(old),
                 'target_mean_NLL': sum(r['nll'] for r in target)/len(target),
                 'old_mean_NLL': sum(r['nll'] for r in old)/len(old),
                 'target_Answer_F1': sum(r['scores']['token_f1'] for r in target_gen)/len(target_gen) if valid else None,
                 'old_Answer_F1': sum(r['scores']['token_f1'] for r in old_gen)/len(old_gen) if valid else None,
                 'invalid_case_diagnostic_scores_only': None if valid else {
                     'target_F1': sum(r['scores']['token_f1'] for r in target_gen)/len(target_gen),
                     'meaning': 'No accepted target update; excluded from method quality ranking.'},
                 'training': training, 'model_counts': model,
                 'memory_bytes': json.loads((run/'memory_summary.json').read_text()), 'run': proof['run']}
        result.append(entry);accepted[task, variant] = (entry, target_gen, old_gen)
    comparisons = []
    for task in (2, 11, 12):
        if (task, 'control') not in accepted:
            continue
        control, targets, old = accepted[task, 'control']
        for variant in ('A_only', 'B_only', 'A_B'):
            if (task, variant) not in accepted:
                continue
            item, after, after_old = accepted[task, variant]
            if not item['valid_training'] or not control['valid_training']:
                continue
            assert [r['id'] for r in targets] == [r['id'] for r in after]
            comparisons.append({'task': task, 'variant': variant,
                'target': paired([b['scores']['token_f1']-a['scores']['token_f1'] for a, b in zip(targets, after)],
                                 [a['id'] for a in targets], original),
                'old_natural': paired([b['scores']['token_f1']-a['scores']['token_f1'] for a, b in zip(old, after_old)],
                                      [a['id'] for a in old], original)})
    for name, digest in sources.items():
        assert queue.prior.sha(ROOT/name) == digest
    output = {'completed_cases': len(result), 'planned_cases': 12, 'case_results': result,
              'valid_training_cases': sum(v['valid_training'] for v in result), 'vs_control': comparisons,
              'metric': 'Answer F1', 'sources': sources, 'reporter_sha256': queue.prior.sha(Path(__file__)),
              'formal_CL_results': False, 'large_gain_demonstrated': False,
              'limits': ['Independent one-update pilot tasks, not a19-stage CL sequence; no AP/FWT/BWT inferred.',
                         'Only two old-natural TRAIN memory QA; constraints do not guarantee all old tasks/test quality.',
                         'Old generation10 QA is diagnostic; target full228/267/100 was previously studied, not untouched confirmation.',
                         'Final19-stage/1313TRAIN/21454TEST confirmation and module-count study remain required before broad improvement claims.',
                         'Fixed-weight single-seed exploratory intervals uncorrected; failed-training cases not ranked.']}
    queue.prior.write(REPORT/'VERIFIED_PILOT_RESULTS.json', output)
    lines = ['# 两模块原型：已核验的GPU结果', '',
             f"完成{len(result)}/12组，其中有效训练{output['valid_training_cases']}组。0组意味着尚待运行，不是0效果。", '',
             '| 原生任务 | 条件 | 目标Answer F1 | 旧自然Answer F1（10题） | 继承强度 | 接受步长比例 | 有效更新 |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for row in result:
        t = row['training'];target = f"{row['target_Answer_F1']:.3f}" if row['valid_training'] else '未通过正更新'
        old = f"{row['old_Answer_F1']:.3f}" if row['valid_training'] else '—'
        lines.append(f"| T{row['task']:02d} | {row['variant']} | {target} | {old} | {t['selection']['chosen_strength']:.3f} | {t['actual_step_fraction']:.3f} | {t['nonzero_effective_updates']} |")
    lines += ['', '训练前后实际NLL、全部候选/拒绝、原始梯度、参数/模型数量、CPU/GPU前后/峰值、配对差值见JSON与原运行。只用原Answer F1；这些独立任务不能合成持续学习AP/FWT/BWT，也不能称全1313/21454结果。', '']
    (REPORT/'VERIFIED_PILOT_RESULTS.md').write_text('\n'.join(lines).rstrip()+'\n')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--watch', action='store_true');args = parser.parse_args()
    try:
        if args.watch:
            observer_lock = (REPORT/'observer.lock').open('a')
            fcntl.flock(observer_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = None
        while True:
            signature = sorted((p.name, p.stat().st_mtime_ns) for p in REPORT.glob('T*_complete.json'))
            if signature != previous:
                output = build();previous = signature
                queue.prior.write(REPORT/'observer_status.json', {'pid': os.getpid(), 'CPU_only': True,
                    'phase': 'completed' if output['completed_cases'] == 12 else 'waiting_completed_GPU_cases',
                    'complete_cases': output['completed_cases']})
                if output['completed_cases'] == 12:
                    break
            if not args.watch:
                break
            time.sleep(45)
    except Exception:
        queue.prior.write(REPORT/'observer_needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc(),
                                                                 'active_GPU_source_not_changed': True})
        raise
