"""CPU-only proof-bound AnswerF1 tables; diagnostic and full cohorts separate."""
import argparse
import fcntl
import json
import math
import os
import time
import traceback
from collections import defaultdict

import onevision_two_module_fullqueue as queue
from answer_f1 import answer_f1

ROOT = queue.ROOT
REPORT = queue.FULL


def task_mean(rows):
    groups = defaultdict(list)
    for row in rows:
        if 'prediction' in row:
            groups[row['native_task']].append(row['scores']['token_f1'])
    return {task: sum(values)/len(values) for task, values in groups.items()}


def metrics(final, base, pre, acquired):
    ap = sum(final.values())/19 if set(final) == set(range(1, 20)) else None
    fwt = sum(pre[t]-base[t] for t in range(2, 20))/18 if all(t in pre and t in base for t in range(2, 20)) else None
    bwt = sum(final[t]-acquired[t] for t in range(1, 19))/18 if all(t in final and t in acquired for t in range(1, 19)) else None
    return {'AP': ap, 'FWT': fwt, 'BWT': bwt}


def rows(directory, label, domains, original):
    result = []
    for domain in domains:
        path = directory/'evaluations'/label/f'{domain}_image.jsonl'
        if not path.exists():
            continue
        for row in map(json.loads, path.read_text().splitlines()):
            assert row['id'] in original
            source = original[row['id']]
            assert row['references'] == source['answers'] and row['native_task'] == source['_task']
            if 'prediction' in row:
                assert row['scores']['token_f1'] == answer_f1(row['prediction'], row['references'])
            assert math.isfinite(row['nll'])
            result.append(row)
    return result


def build():
    _, _, training, confirmation = queue.plans()
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = json.loads(manifest_path.read_text());domains = list(confirmation['all_TEST_ids'])
    original = {}
    for task in manifest['tasks']:
        file = task['files']['test'];path = ROOT/file['path']
        assert queue.previous.prior.sha(path) == file['sha256']
        original.update({r['id']: dict(r, _task=task['task']) for r in map(json.loads, path.read_text().splitlines())})
    assert len(original) == 21454
    entries = [];sources = {}
    shared = {}
    if queue.confirmation_complete('control', 'initial', confirmation):
        path = queue.CONFIRM/'shared_initial_complete.json';proof = json.loads(path.read_text())
        sources[str(path.relative_to(ROOT))] = queue.previous.prior.sha(path)
        actual = rows(ROOT/proof['run'], 'base_full', domains, original)
        assert len(actual) == 21454 and all('prediction' in r for r in actual)
        shared = task_mean(actual)
    for variant in queue.VARIANTS:
        status = queue.full_training(variant, training)
        item = {'variant': variant, 'training_status': status, 'diagnostic155': None, 'full21454': None}
        if status != 'valid':
            entries.append(item);continue
        path = REPORT/(variant+'_complete.json');proof = json.loads(path.read_text());run = ROOT/proof['run']
        sources[str(path.relative_to(ROOT))] = queue.previous.prior.sha(path)
        final = rows(run, 'acquired_T19', domains, original)
        assert sum('prediction' in r for r in final) == 155
        base = task_mean(rows(run, 'base_initial', domains, original));pre = {};acquired = {}
        for task in manifest['tasks']:
            t, domain = task['task'], task['domain']
            pre.update(task_mean(rows(run, f'pre_T{t:02d}', [domain], original)))
            if t < 19:
                acquired.update(task_mean(rows(run, f'acquired_T{t:02d}', [domain], original)))
        item['diagnostic155'] = {'generated_QA': 155, **metrics(task_mean(final), base, pre, acquired)}
        training_rows = [json.loads((run/'stages'/f'T{t:02d}'/'training.json').read_text()) for t in range(1, 20)]
        item['training'] = {'target_QA': 1313, 'attempted_updates': sum(r['attempted_updates'] for r in training_rows),
             'effective_updates': sum(r['effective_updates'] for r in training_rows),
             'extra_TRAIN_probe_QA_forwards': sum(r['probe_QA_forwards'] for r in training_rows),
             'actual_used_lr_range': [min(e['actual_learning_rate_used'] for r in training_rows for e in r['events']),
                                      max(e['actual_learning_rate_used'] for r in training_rows for e in r['events'])],
             'per_stage_training_memory': json.loads((run/'memory_summary.json').read_text())['per_stage_training_windows'],
             'model_counts': json.loads((run/'model_counts.json').read_text())}
        if queue.confirmation_complete(variant, 'final', confirmation):
            path = queue.CONFIRM/(variant+'_final_complete.json');proof = json.loads(path.read_text());full_run = ROOT/proof['run']
            sources[str(path.relative_to(ROOT))] = queue.previous.prior.sha(path)
            full = rows(full_run, 'final_full', domains, original)
            assert len(full) == 21454 and all('prediction' in r for r in full)
            assert {r['id'] for r in full} == set(original)
            task_domain = {t['task']: t['domain'] for t in manifest['tasks']}
            grouped = {domain: [r for r in full if task_domain[r['native_task']] == domain] for domain in domains}
            domain_scores = {domain: sum(r['scores']['token_f1'] for r in grouped[domain])/len(grouped[domain]) for domain in domains}
            current = {};before = {}
            if queue.confirmation_complete(variant, 'sparse', confirmation):
                path = queue.CONFIRM/(variant+'_sparse_complete.json');sparse = json.loads(path.read_text())
                sources[str(path.relative_to(ROOT))] = queue.previous.prior.sha(path)
                for stage in range(1, 19):
                    values = task_mean(rows(ROOT/sparse['run'], f'state_T{stage:02d}_current_next', domains, original))
                    assert set(values) == {stage, stage+1}
                    current[stage], before[stage+1] = values[stage], values[stage+1]
            item['full21454'] = {'QA': 21454, 'domain_scores': domain_scores,
                  'domain_macro': sum(domain_scores.values())/5,
                  'QA_weighted': sum(r['scores']['token_f1'] for r in full)/21454,
                  'native_task_final': task_mean(full), **metrics(task_mean(full), shared, before, current),
                  'cohort_scores': json.loads((full_run/'cohort_scores.json').read_text()),
                  'full_FWT_BWT_available': bool(before) and bool(shared)}
        entries.append(item)
    control = next(r for r in entries if r['variant'] == 'control')
    for item in entries:
        if item['full21454'] and control['full21454']:
            delta = item['full21454']['AP']-control['full21454']['AP']
            item['full_AP_delta_vs_control_pp'] = delta
            item['predeclared_plus5pp_target_observed'] = delta >= 5
    result = {'primary': 'Existing raw AnswerF1 only', 'mode': training['mode'], 'entries': entries,
              'source_proofs': sources, 'unmeasured_values': None,
              'no_claim_of_gain_from_CPU_checks': True, 'target_plus5pp_is_hypothesis_not_guarantee': True}
    queue.previous.prior.write(REPORT/'VERIFIED_RESULTS.json', result)
    text = ['# 两模块四组消融结果', '', '唯一主指标：Answer F1。完整测试与155题诊断分开；未完成用“待测”，无有效训练的组不能参加排名。', '',
            '| 方法 | 状态 | '+' | '.join(domains)+' | AP | FWT | BWT |', '|---|---|'+'---:|'*8]
    fmt = lambda value: '待测' if value is None else f'{value:.3f}'
    for item in entries:
        full = item['full21454'];values = [None]*8 if full is None else [*[full['domain_scores'][d] for d in domains], full['AP'], full['FWT'], full['BWT']]
        text.append('| '+item['variant']+' | '+item['training_status']+' | '+' | '.join(fmt(v) for v in values)+' |')
    text += ['', '完整AP需每组全部21454题；完整FWT/BWT还需同一完整测试队列的初始、刚学完及下一任务训练前结果。阶段未测单元保持空缺，155题结果不能代填。', '',
             '| 方法 | 诊断155 AP | 诊断155 FWT | 诊断155 BWT |', '|---|---:|---:|---:|']
    for item in entries:
        diagnostic = item['diagnostic155']
        text.append('| '+item['variant']+' | '+' | '.join(fmt(None if diagnostic is None else diagnostic[k]) for k in ['AP', 'FWT', 'BWT'])+' |')
    text += ['', '全部四组预先固定，不按测试分数挑组。单rank8、540672个可训练参数；训练探针、候选前向、实际梯度、逐阶段内存前后值/峰值及checkpoint计入成本。原生AnyRes研究设置与正式V2缩放输入基线分别报告。', '']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(text))
    return result


def check():
    final = {t: t+4. for t in range(1, 20)};base = {t: float(t) for t in range(1, 20)}
    pre = {t: t+2. for t in range(2, 20)};acquired = {t: t+5. for t in range(1, 19)}
    assert metrics(final, base, pre, acquired) == {'AP': 14., 'FWT': 2., 'BWT': -1.}
    assert metrics(final, {}, pre, acquired)['FWT'] is None
    assert metrics(final, base, pre, {})['BWT'] is None
    assert metrics({1: 100.}, base, pre, acquired)['AP'] is None
    result = build()
    queue.previous.prior.write(REPORT/'CPU_REPORT_VALIDATION.json', {'passed': True, 'CPU_only': True, 'formula_checks': 4,
          'actual_GPU_cases_verified': sum(e['training_status'] == 'valid' for e in result['entries']),
          'source_sha256': queue.previous.prior.sha(__import__('pathlib').Path(__file__))})
    print(json.dumps({'passed': True, 'formula_checks': 4, 'GPU_used': False}))


def main(watch=False):
    lock = (REPORT/'observer.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signature = None
    while True:
        paths = [p for directory in [REPORT, queue.CONFIRM] for p in directory.glob('*_complete.json')]
        paths += list(REPORT.glob('*_invalid.json'))
        current = sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)
        if signature != current:
            result = build();signature = current
            queue.previous.prior.write(REPORT/'observer_status.json', {'pid': os.getpid(), 'CPU_only': True,
                 'valid19stage_variants': sum(r['training_status'] == 'valid' for r in result['entries']),
                 'full_final_variants': sum(r['full21454'] is not None for r in result['entries'])})
        if not watch:
            return
        time.sleep(30)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true');parser.add_argument('--watch', action='store_true');args = parser.parse_args()
    try:
        check() if args.check else main(args.watch)
    except Exception:
        queue.previous.prior.write(REPORT/'observer_needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
