"""Read-only, proof-triggered four-cell report; no model or optimizer loading."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import json
from pathlib import Path
import time
import traceback

import onevision_cold_quality_queue as queue
import onevision_functional_router_report as routing
import onevision_warm_start_report as initialization

ROOT, BASE, REPORT = queue.ROOT, queue.BASE, queue.STUDY
DOMAINS = initialization.DOMAINS


def warm_routing(spec):
    names = ['STUDY', 'FULL', 'functional_plan', 'complete']
    saved = {name: getattr(routing, name) for name in names}
    try:
        routing.STUDY, routing.FULL = REPORT, queue.FULL
        routing.functional_plan, routing.complete = lambda: spec, queue.complete
        return routing.build()
    finally:
        for name, value in saved.items():
            setattr(routing, name, value)


def route_row(result, scope, cell):
    found = [r for r in result['aggregate'] if r['scope'] == scope and r['view'] == 'quality_hard']
    assert len(found) <= 1
    if not found:
        return None
    a = found[0]
    domains = {r['domain']:r['Answer_F1'] for r in result['cells']
               if r['scope'] == scope and r['view'] == 'quality_hard'}
    assert set(domains) == set(DOMAINS)
    return {'cell': cell, 'generated': a['generated'], 'domains': domains,
            'AP': a['native19_AP'], 'domain_macro': a['domain_macro_F1'], 'FWT': None, 'BWT': None}


def build():
    auth = json.loads((queue.REPORT/'AUTHORIZATION.json').read_text())
    assert auth['source_sha256'] == queue.method_sources(), 'Frozen source changed'
    lock = json.loads((ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert queue.sha(ROOT/'analysis/answer_f1.py') == lock['metric_sha256']
    assert not lock['option_mapping'] and not lock['additional_prefix_or_answer_extraction']
    init = initialization.build()
    cold = routing.build()
    spec = queue.checked_plan()
    warm = warm_routing(spec) if spec else None
    rows = {'diagnostic155': [], 'full21454': []}
    for scope, phase in [('diagnostic155', 'train'), ('full21454', 'full')]:
        for variant, cell in [('learned_cosine_K3', '00'), (queue.prior.cold.previous.VARIANT, '10')]:
            matches = [r for r in init['results'] if r['variant'] == variant and r['phase'] == phase]
            assert len(matches) <= 1
            for r in matches:
                rows[scope].append({'cell': cell, 'generated': r['generated'], 'domains': r['domains'],
                    'AP': r['AP'], 'domain_macro': sum(r['domains'].values())/5, 'FWT': None, 'BWT': None})
        report_scope = 'pilot' if phase == 'train' else 'full'
        for r, cell in [(cold, '01'), (warm, '11')]:
            row = route_row(r, report_scope, cell) if r else None
            if row:
                rows[scope].append(row)
        rows[scope].sort(key=lambda r:r['cell'])
        assert all(r['generated'] == (155 if phase == 'train' else 21454) for r in rows[scope])
    resources = {'initialization_controls': init['resources'], 'quality_inference': {}}
    if spec:
        for name in ['pilot_complete.json', 'full_quality_hard_complete.json']:
            p = REPORT/name
            if p.exists():
                proof = json.loads(p.read_text())
                # warm_routing above already validates the full proof and files.
                resources['quality_inference'][name] = json.loads((ROOT/proof['run']/'memory_summary.json').read_text())
    interaction = {}
    for scope, values in rows.items():
        by_cell = {r['cell']:r for r in values}
        if len(by_cell) == 4:
            interaction[scope] = {key: by_cell['11'][key]-by_cell['10'][key]-by_cell['01'][key]+by_cell['00'][key]
                                  for key in ['AP', 'domain_macro']}
    return {'CPU_only': True, 'raw_Answer_F1': True, 'formal_V2': False,
            'TEST_selects_anything': False, 'rows': rows, 'interaction': interaction,
            'parent_warm19_complete': init['warm_train_complete'], 'warm_plan_ready': spec is not None,
            'complete_full_cells': len(rows['full21454']), 'required_full_cells': 4,
            'resources': resources, 'calibration': warm['calibration'] if warm else [],
            'completion_sha256': {'initialization': init['completion_sha256'],
                                  'cold_quality': cold['completion_sha256'],
                                  'warm_quality': warm['completion_sha256'] if warm else {}},
            'source_sha256': queue.method_sources(),
            'limits': ['Local inherited-control evidence does not prove all-method or knowledge-only causes.',
                       '155 diagnostic generated answers are not complete21454.',
                       'No fullFWT/BWT from final-only. Missing rows remain pending, not zero.',
                       'Both quality cells charge43 historical TRAIN,172NLL forwards and16parameter CPU fit.',
                       'Interaction is a descriptive four-cell contrast, not multi-seed significance.',
                       'No cold/warm gains are added together to invent combined performance.']}


def once():
    r = build()
    queue.write(REPORT/'FOUR_CELL_EVIDENCE.json', r)
    lines = ['# 冷启动继承与功能质量调用：四格结果', '',
             f"完整格：{r['complete_full_cells']}/4；温启动19阶段：{int(r['parent_warm19_complete'])}/1。缺失为待测，不是0分。", '']
    names = {'00':'冷启动＋原几何', '10':'完整继承＋原几何', '01':'冷启动＋质量硬调用', '11':'完整继承＋质量硬调用'}
    for scope in ['full21454', 'diagnostic155']:
        lines += ['## '+scope, '', '| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏平均 | AP | FWT | BWT |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        found = {row['cell']:row for row in r['rows'][scope]}
        for cell, name in names.items():
            row = found.get(cell)
            scores = [f"{row['domains'][d]:.3f}" for d in DOMAINS] + [f"{row['domain_macro']:.3f}", f"{row['AP']:.3f}"] if row else ['—']*7
            lines.append('| '+cell+' '+name+' | '+' | '.join(scores)+' | — | — |')
        lines += ['']
    lines += ['完整与诊断分表；全部原始ID/标注/Answer F1逐题核验后才填格，不拼接不同协议。', '',
              '[两问题、模块、判据及成本](PROBLEM_AND_METHOD.md) · [全部证明与资源](FOUR_CELL_EVIDENCE.json)', '']
    lines += ['- '+s for s in r['limits']]
    (REPORT/'FOUR_CELL_RESULTS.md').write_text('\n'.join(lines)+'\n')
    queue.write(REPORT/'observer_status.json', {'pid':os.getpid(), 'CPU_only':True,
        'complete_full_cells':r['complete_full_cells'], 'parent_warm19_complete':r['parent_warm19_complete'],
        'warm_plan_ready':r['warm_plan_ready']})
    return r


def stamp():
    keys = [BASE/'learned_reuse'/('learned_cosine_K3_'+p+'_complete.json') for p in ['train','full']]
    keys += [BASE/'warm_start_20261018'/('warm_inherit_K3_v1_'+p+'_complete.json') for p in ['train','full']]
    keys += [BASE/'functional_router'/name for name in ['pilot_complete.json','full_quality_hard_complete.json']]
    keys += [REPORT/name for name in ['CPU_PLAN_CHECK.json','pilot_complete.json','full_quality_hard_complete.json']]
    return tuple(queue.sha(p) if p.exists() else None for p in keys)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    try:
        previous = None
        while True:
            current = stamp()
            if current != previous:
                r = once()
                previous = current
                print(json.dumps({'CPU_only':True, 'complete_full_cells':r['complete_full_cells'],
                                  'parent_warm19_complete':r['parent_warm19_complete']}), flush=True)
            if not args.watch or r['complete_full_cells'] == 4:
                break
            time.sleep(45)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        queue.write(REPORT/'observer_needs_attention.json', {'pid':os.getpid(), 'traceback':traceback.format_exc()})
        raise
