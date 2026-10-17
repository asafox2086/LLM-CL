"""CPU proof-gated full21454 confirmation, never reads a moving TEST suffix.

Watch parent uses stdlib only. Immutable trained-state equality checks run in a
separate CPU Torch process. No training, GPU use, or TEST-dependent selection.
"""
import argparse
import fcntl
import json
import math
import os
import random
import subprocess
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

import onevision_causal_priority_queue as validator
from answer_f1 import answer_f1

ROOT = validator.ROOT
BASE = validator.BASE
REPORT = BASE/'full_confirmation_evidence'


def checked_json(path, sources):
    sources[str(path.relative_to(ROOT))] = validator.sha(path)
    return json.loads(path.read_text())


def route_diagnostic(proof, saved, sources):
    """Compare existing two500-QA NLL evaluations at the same frozen pool."""
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = checked_json(manifest_path, sources)
    assert validator.sha(manifest_path) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    original = {}
    for task in manifest['tasks']:
        file = task['files']['test'];path = ROOT/file['path']
        assert validator.sha(path) == file['sha256'];sources[str(path.relative_to(ROOT))] = file['sha256']
        original.update({r['id']: r for r in map(json.loads, path.read_text().splitlines())})
    directory = ROOT/proof['run'];result = []
    for domain in dict.fromkeys(t['domain'] for t in manifest['tasks']):
        versions = []
        for label in ['final_input_router', 'final_known_task_reference']:
            path = directory/'evaluations'/label/f'{domain}_image.jsonl'
            assert validator.sha(path) == proof['artifact_sha256'][str(path.relative_to(directory))]
            sources[str(path.relative_to(ROOT))] = validator.sha(path)
            rows = list(map(json.loads, path.read_text().splitlines()));assert len(rows) == 100
            for row in rows:
                assert row['references'] == original[row['id']]['answers'] and math.isfinite(row['nll'])
            versions.append(rows)
        input_rows, reference = versions
        assert [r['id'] for r in input_rows] == [r['id'] for r in reference]
        clusters = defaultdict(list)
        for a, b in zip(input_rows, reference):
            clusters[tuple(original[a['id']]['images'])].append(a['nll']-b['nll'])
        groups = list(clusters.values());samples = [];rng = random.Random(42)
        for _ in range(2000):
            selected = [groups[rng.randrange(len(groups))] for _ in groups]
            samples.append(sum(map(sum, selected))/sum(map(len, selected)))
        samples.sort()
        result.append({'domain': domain, 'QA': 100, 'image_clusters': len(groups),
            'input_router_NLL': sum(r['nll'] for r in input_rows)/100,
            'task_assignment_reference_NLL': sum(r['nll'] for r in reference)/100,
            'input_minus_reference_NLL': sum(map(sum, groups))/100,
            'image_cluster_95CI': [samples[49], samples[1949]],
            'exact_expert_agreement_QA': sum(a['route_expert'] == b['route_expert'] for a, b in zip(input_rows, reference)),
            'reference_expert_is_recorded_assignment': all(r['route_expert'] == saved['assignments'][r['native_task']] for r in reference),
            'uncorrected': True})
    return result


def compare_router_prototypes(proof, saved, sources):
    import torch
    control = checked_json(BASE/'capacity/cap1_complete.json', sources)
    path = ROOT/control['run']/'latest.pt';assert validator.sha(path) == control['final_state_sha256']
    sources[str(path.relative_to(ROOT))] = control['final_state_sha256']
    first = torch.load(path, map_location='cpu', weights_only=True)
    assert set(first['prototypes']) == set(saved['prototypes']) == set(range(1, 20))
    vector_checks = [{'task': task, 'exact_equal': torch.equal(first['prototypes'][task], saved['prototypes'][task]),
                      'max_abs_difference': float((first['prototypes'][task]-saved['prototypes'][task]).abs().max())}
                     for task in range(1, 20)]
    predictions = []
    for domain in ['natural', 'medical', 'geography', 'finance', 'traffic']:
        rows = []
        for item in [control, proof]:
            path = ROOT/item['run']/'evaluations/final_input_router'/f'{domain}_image.jsonl'
            assert validator.sha(path) == item['artifact_sha256'][str(path.relative_to(ROOT/item['run']))]
            sources[str(path.relative_to(ROOT))] = validator.sha(path)
            rows.append(list(map(json.loads, path.read_text().splitlines())))
        assert [r['id'] for r in rows[0]] == [r['id'] for r in rows[1]]
        assert [r['references'] for r in rows[0]] == [r['references'] for r in rows[1]]
        predictions.append({'domain': domain, 'QA': len(rows[0]),
             'same_selected_prototype_QA': sum(a['route_prototype_task'] == b['route_prototype_task'] for a, b in zip(*rows)),
             'selected_prototype_matches_native_task_QA': sum(r['route_prototype_task'] == r['native_task'] for r in rows[1])})
    return {'compared_caps': [1, 3], 'prototype_vectors': vector_checks,
            'predictions': predictions, 'prototype_scalar_count': sum(v.numel() for v in saved['prototypes'].values()),
            'interpretation': 'More LoRA experts need not alter the frozen-base input feature/centroid decisions. Centroids are still fitted on TRAIN; not a claim no routing learning ever occurs.',
            'warning': 'Native-task/prototype agreement is not expert routing accuracy: multiple tasks share an expert and may overwrite it.'}


def state_audit(cap):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    torch.set_num_threads(4)
    sources = {str(Path(__file__).relative_to(ROOT)): validator.sha(Path(__file__))}
    proof = checked_json(BASE/'capacity'/f'cap{cap}_complete.json', sources)
    assert proof['complete'] and proof['cap'] == cap
    path = ROOT/proof['run']/'latest.pt'
    assert validator.sha(path) == proof['final_state_sha256']
    sources[str(path.relative_to(ROOT))] = proof['final_state_sha256']
    saved = torch.load(path, map_location='cpu', weights_only=True)
    assert saved['stage'] == 19 and saved['cap'] == cap
    assert saved['source_sha256'] == proof['source_sha256']
    for relative, sha in proof['source_sha256'].items():
        assert validator.sha(ROOT/relative) == sha;sources[relative] = sha
    assert len(saved['pool']) == proof['retained_experts'] <= cap
    exact = []
    if cap == 1:
        native = checked_json(BASE/'deterministic_native/complete.json', sources)
        assert native['complete']
        for stage in range(1, 20):
            original = ROOT/native['run']/'adapters'/f'sequential_T{stage:02d}.pt'
            candidate = ROOT/proof['run']/'states'/f'cap1_T{stage:02d}.pt'
            assert validator.sha(original) == native['artifact_sha256'][str(original.relative_to(ROOT/native['run']))]
            assert validator.sha(candidate) == proof['artifact_sha256'][str(candidate.relative_to(ROOT/proof['run']))]
            sources[str(original.relative_to(ROOT))] = validator.sha(original)
            sources[str(candidate.relative_to(ROOT))] = validator.sha(candidate)
            expected = torch.load(original, map_location='cpu', weights_only=True)['adapter']
            state = torch.load(candidate, map_location='cpu', weights_only=True)
            assert len(state['pool']) == 1
            actual = next(iter(state['pool'].values()))
            assert actual.keys() == expected.keys() and len(actual) == 48
            assert all(torch.equal(actual[k][f], expected[k][f]) for k in actual for f in ['A', 'B'])
            exact.append(stage)
        assert all(torch.equal(next(iter(saved['pool'].values()))[k][f], expected[k][f]) for k in expected for f in ['A', 'B'])
    count = sum(t.numel() for state in saved['pool'].values() for factors in state.values() for t in factors.values())
    assert count == 540672*len(saved['pool'])
    assert not torch.cuda.is_initialized()
    routes = route_diagnostic(proof, saved, sources)
    if cap == 1:
        assert all(r['input_minus_reference_NLL'] == 0 and r['exact_expert_agreement_QA'] == 100 for r in routes)
    result = {'passed': True, 'CPU_only': True, 'cap': cap, 'retained_modules': len(saved['pool']),
              'retained_LoRA_parameters': count, 'actual_full19_states_exact_to_original_SeqFT': exact,
              'assignments': saved['assignments'], 'source_sha256': sources,
              'same_pool_routing_NLL500_auxiliary': routes,
              'optimal_self_selected_capacity_proven': False}
    if cap == 3:
        result['frozen_router_comparison'] = compare_router_prototypes(proof, saved, sources)
    validator.write(REPORT/f'cap{cap}_STATE_AUDIT.json', result)
    print(json.dumps({'CPU_only': True, 'cap': cap, 'modules': len(saved['pool']), 'exact_native_stages': len(exact)}))


def mean(rows):
    assert rows and all(math.isfinite(r['scores']['token_f1']) for r in rows)
    return sum(r['scores']['token_f1'] for r in rows)/len(rows)


def summarize(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['native_task']].append(row)
    values = {task: mean(part) for task, part in groups.items()}
    return {'QA': len(rows), 'QA_weighted_Answer_F1': mean(rows), 'native_task_F1': values,
            'native_task_AP': sum(values.values())/19 if set(values) == set(range(1, 20)) else None}


def audit(cap):
    assert validator.complete_full(cap), 'Full21454 durable proof required; never inspect live partials'
    sources = {str(Path(__file__).relative_to(ROOT)): validator.sha(Path(__file__))}
    lock = checked_json(ROOT/'summary/primary_metric_20261017/metric_lock.json', sources)
    assert lock['option_mapping'] is False and validator.sha(ROOT/'analysis/answer_f1.py') == lock['metric_sha256']
    sources['analysis/answer_f1.py'] = lock['metric_sha256']
    gate = checked_json(REPORT/f'cap{cap}_STATE_AUDIT.json', sources);assert gate['passed'] and gate['CPU_only']
    for relative, sha in gate['source_sha256'].items():
        assert validator.sha(ROOT/relative) == sha
    full_proof = checked_json(BASE/'full_input_repair'/f'cap{cap}_full_confirm_complete.json', sources)
    trained = checked_json(BASE/'capacity'/f'cap{cap}_complete.json', sources)
    assert full_proof['trained_state_sha256'] == trained['final_state_sha256']
    run, old = ROOT/full_proof['run'], ROOT/trained['run']
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = checked_json(manifest_path, sources)
    assert validator.sha(manifest_path) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    domains = list(dict.fromkeys(t['domain'] for t in manifest['tasks']))
    all_rows, diagnostic, per_domain, reference_nll = [], [], [], []
    for domain in domains:
        expected = []
        for task in manifest['tasks']:
            if task['domain'] != domain:
                continue
            file = task['files']['test'];path = ROOT/file['path']
            assert validator.sha(path) == file['sha256'];sources[str(path.relative_to(ROOT))] = file['sha256']
            expected += [dict(json.loads(line), _task=task['task']) for line in path.read_text().splitlines()]
        path = run/'evaluations/final_full_input_router'/f'{domain}_image.jsonl'
        assert validator.sha(path) == full_proof['artifact_sha256'][str(path.relative_to(run))]
        sources[str(path.relative_to(ROOT))] = validator.sha(path)
        rows = list(map(json.loads, path.read_text().splitlines()))
        assert [r['id'] for r in rows] == [r['id'] for r in expected]
        for row, original in zip(rows, expected):
            assert row['references'] == original['answers'] and row['native_task'] == original['_task']
            assert math.isfinite(row['nll']) and row['scores']['token_f1'] == answer_f1(row['prediction'], row['references'])
        all_rows += rows;current = {r['id']: r for r in rows}
        path = old/'evaluations/final_input_router'/f'{domain}_image.jsonl'
        assert validator.sha(path) == trained['artifact_sha256'][str(path.relative_to(old))]
        sources[str(path.relative_to(ROOT))] = validator.sha(path)
        previous = list(map(json.loads, path.read_text().splitlines()));assert len(previous) == 100
        for row in previous:
            now = current[row['id']]
            assert now['references'] == row['references'] and now['nll'] == row['nll']
            if 'prediction' in row:
                assert now['prediction'] == row['prediction'] and now['scores']['token_f1'] == row['scores']['token_f1']
                diagnostic.append(now)
        seen_ids = {r['id'] for r in previous}
        seen = [r for r in rows if r['id'] in seen_ids];others = [r for r in rows if r['id'] not in seen_ids]
        assignments = {int(k): v for k, v in gate['assignments'].items()}
        per_domain.append({'domain': domain, **summarize(rows), 'predeclared_diagnostic100': summarize(seen),
             'outside_predeclared_diagnostic100': summarize(others),
             'expert_matches_task_final_assignment_QA': sum(r['route_expert'] == assignments[r['native_task']] for r in rows),
             'dominant_raw_outputs': Counter(r['prediction'] for r in rows).most_common(5),
             'NLL_reference_truncated_QA': sum(r['target_truncated'] for r in rows)})
        # Same fixed pool, task-label reference is AUXILIARY, never actual routing.
        path = old/'evaluations/final_known_task_reference'/f'{domain}_image.jsonl'
        assert validator.sha(path) == trained['artifact_sha256'][str(path.relative_to(old))]
        sources[str(path.relative_to(ROOT))] = validator.sha(path)
        known = list(map(json.loads, path.read_text().splitlines()))
        assert [r['id'] for r in known] == [r['id'] for r in previous]
        reference_nll.append({'domain': domain, 'QA': 100,
             'input_router_NLL': sum(r['nll'] for r in previous)/100,
             'task_assignment_reference_NLL': sum(r['nll'] for r in known)/100,
             'input_minus_reference_NLL': sum(a['nll']-b['nll'] for a, b in zip(previous, known))/100,
             'scope': 'Task-label auxiliary over same fixed pool; not best-expert oracle, generated accuracy, or actual routing'})
    assert len(all_rows) == 21454 and len({r['id'] for r in all_rows}) == 21454 and len(diagnostic) == 155
    result = {'cap': cap, 'full21454_complete': True, 'metric': 'Existing raw Answer F1 only',
              'retained_modules': gate['retained_modules'], 'retained_LoRA_parameters': gate['retained_LoRA_parameters'],
              'full': summarize(all_rows), 'domains': per_domain,
              'five_domain_macro': sum(r['QA_weighted_Answer_F1'] for r in per_domain)/5,
              'same_model_diagnostic155': summarize(diagnostic), 'FWT': None, 'BWT': None,
              'overlap_checks': {'NLL_exact_QA': 500, 'prediction_and_F1_exact_QA': 155},
              'router_assignment_auxiliary': reference_nll, 'source_sha256': sources,
              'limits': ['Final-only full inference cannot produce full FWT/BWT or acquisition/forgetting causality.',
                         'Research nativeAnyRes protocol, not the shared resized-input formal V2 baseline.',
                         'Caps1/3 predeclared; all capacity runs hit their ceilings, not optimal self-selected module count.',
                         'Diagnostic500 and other20954 are reporting partitions, not a claim the other20954 were untouched; auxiliary studies and earlier closed-domain audits inspected further TEST examples.',
                         'Agreement with task-final expert assignment is not router accuracy or a proof which expert is semantically best; older experts may be overwritten.',
                         'Single seed/order/fitted checkpoints; cohort score differences are not training gains or domain-count causal effects.',
                         'AnswerF1 is lexical output-quality overlap, not complete semantic correctness.']}
    for name, sha in sources.items():
        assert validator.sha(ROOT/name) == sha
    validator.write(REPORT/f'cap{cap}_VERIFIED.json', result)
    print(json.dumps({'cap': cap, 'full_QA': 21454, 'AP': result['full']['native_task_AP'], 'domain_macro': result['five_domain_macro']}))


def render():
    results = [json.loads(path.read_text()) for path in sorted(REPORT.glob('cap*_VERIFIED.json'))]
    domains = ['natural', 'medical', 'geography', 'finance', 'traffic']
    text = ['# 原生AnyRes研究权重：完整测试确认', '',
            '唯一Answer F1；只接受全部21454题完成证明。cap1是原顺序LoRA权重的容量控制，cap3为旧容量机制对照，不是正式V2主表或已验证新算法。', '',
            '| 研究权重 | 模块数 | '+' | '.join(domains)+' | AP | FWT | BWT |', '|---|---:|'+'---:|'*8]
    for result in results:
        values = {r['domain']: r['QA_weighted_Answer_F1'] for r in result['domains']}
        text.append('| cap'+str(result['cap'])+' | '+str(result['retained_modules'])+' | '+' | '.join(f'{values[d]:.3f}' for d in domains)+f" | {result['full']['native_task_AP']:.3f} | 未测 | 未测 |")
    if not results:
        text += ['', '完整GPU确认尚未完成，当前不发布完整AP或读取正在增长的交通结果。']
    gates = [json.loads(p.read_text()) for p in sorted(REPORT.glob('cap*_STATE_AUDIT.json'))]
    if gates:
        text += ['', '## 既有固定500题路由机制辅助', '',
                 '相同最终专家池，只比较输入原型路由与原任务分配参考；后者使用任务标签，仅诊断，不是实际测试路线、最优专家oracle或生成准确率。两个任务可能共享/覆盖同一专家，分配匹配不能称路由正确率。', '',
                 '| cap | 域 | QA/图像数 | 输入路由NLL | 任务分配参考NLL | 差值及图像聚类95%区间 | 专家相同题数 |',
                 '|---|---|---:|---:|---:|---|---:|']
        for gate in gates:
            for row in gate['same_pool_routing_NLL500_auxiliary']:
                ci = row['image_cluster_95CI']
                text.append(f"| {gate['cap']} | {row['domain']} | 100/{row['image_clusters']} | {row['input_router_NLL']:.4f} | {row['task_assignment_reference_NLL']:.4f} | {row['input_minus_reference_NLL']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] | {row['exact_expert_agreement_QA']} |")
        text += ['', '区间固定权重、单seed、未校正多重比较。NLL差异可定位该固定池的路由影响，不能替代生成性能，不能据此改测试路线或按测试选择专家。cap1全部19阶段的实际参数逐元素等于原确定性顺序LoRA，cap3只确认实际3模块，不证明最优数量。']
        for gate in gates:
            if 'frozen_router_comparison' in gate:
                comparison = gate['frozen_router_comparison']
                exact = sum(r['exact_equal'] for r in comparison['prototype_vectors'])
                same = sum(r['same_selected_prototype_QA'] for r in comparison['predictions'])
                text += ['', f'cap1/cap3实际最终原型向量{exact}/19逐元素相同，500题中{same}/500选择同一原型任务。容量变大没有改变这些原型决策；这不是“路由从来没拟合”，也不是任务ID匹配率等于专家正确率。完整逐域选择与源SHA见cap3_STATE_AUDIT.json。']
    for result in results:
        text += ['', f"cap{result['cap']}：完整任务AP={result['full']['native_task_AP']:.3f}，同权重155题诊断AP={result['same_model_diagnostic155']['native_task_AP']:.3f}；是测试队列差异，不是训练提升。旧500题NLL与155条原生成逐题精确重合。五域等权均值={result['five_domain_macro']:.3f}，与19任务AP分开。", '',
                 '路由的任务标签参考仅报辅助NLL；实际测试路线不使用标签或金标。cap达到上限不能宣称自动找到最优数量。']
    text += ['', '完整FWT/BWT未补初始/习得/下一任务前状态，保持为空。500/其他20954只作报告分区，辅助研究已经查看过其他题，不能称未触及的确认集。单seed，不能由终点分数识别知识或多域的唯一原因。', '']
    (REPORT/'VERIFIED_RESULTS.md').write_text('\n'.join(text))


def check():
    sample = [{'native_task': t, 'scores': {'token_f1': float(t)}, 'id': f'q{t}'} for t in range(1, 20)]
    assert summarize(sample)['native_task_AP'] == 10.
    assert summarize(sample[:3])['native_task_AP'] is None
    repeated = sample+[sample[0]]*10
    assert summarize(repeated)['native_task_AP'] == 10. and summarize(repeated)['QA_weighted_Answer_F1'] != 10.
    assert not validator.complete_full(1) or (BASE/'full_input_repair/cap1_full_confirm_complete.json').is_file()
    validator.write(REPORT/'CPU_REPORT_CHECK.json', {'passed': True, 'CPU_only': True, 'checks': 4,
          'reporter_sha256': validator.sha(Path(__file__)), 'full_GPU_proofs_not_implied': True})
    render();print('CPU native-AP weighting/missing-cell/proof gates passed; no GPU started')


def main(watch=False):
    lock = (REPORT/'observer.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    validated = set()
    while True:
        for cap in [1, 3]:
            if not validator.complete_full(cap):
                continue
            complete = REPORT/f'cap{cap}_VERIFIED.json'
            if complete.exists():
                if cap not in validated:
                    result = json.loads(complete.read_text())
                    for relative, sha in result['source_sha256'].items():
                        assert validator.sha(ROOT/relative) == sha
                    validated.add(cap)
                continue
            gate = REPORT/f'cap{cap}_STATE_AUDIT.json'
            commands = ([['--state-cap', str(cap)]] if not gate.exists() else [])+[['--audit-cap', str(cap)]]
            for command in commands:
                with (REPORT/f'cap{cap}_CPU_audit.log').open('a') as log:
                    subprocess.run([str(ROOT/'.vision-env/bin/python'), '-u', str(Path(__file__)), *command], cwd=ROOT,
                        env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4'),
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
            render()
        validator.write(REPORT/'observer_status.json', {'pid': os.getpid(), 'CPU_only': True,
            'verified_caps': [cap for cap in [1, 3] if (REPORT/f'cap{cap}_VERIFIED.json').exists()]})
        if not watch:
            return
        time.sleep(60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--state-cap', type=int, choices=[1, 3])
    parser.add_argument('--audit-cap', type=int, choices=[1, 3]);parser.add_argument('--check', action='store_true')
    parser.add_argument('--watch', action='store_true');args = parser.parse_args();REPORT.mkdir(parents=True, exist_ok=True)
    try:
        if args.check:
            check()
        elif args.state_cap is not None:
            state_audit(args.state_cap)
        elif args.audit_cap is not None:
            audit(args.audit_cap)
        else:
            main(args.watch)
    except Exception:
        validator.write(REPORT/'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
