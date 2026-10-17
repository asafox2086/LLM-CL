"""Read-only chronological decomposition; not a randomized domain-count study."""
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return sum(values) / len(values)


def main():
    sources = {}

    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    proof = read(REPORT / 'deterministic_native/complete.json')
    assert proof['complete']
    run = ROOT / proof['run']
    for name, digest in proof['source_sha256'].items():
        assert sha(ROOT / name) == digest
        sources[name] = digest
    manifest_path = ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = read(manifest_path)
    assert sources[str(manifest_path.relative_to(ROOT))] == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    original = {}
    for task in manifest['tasks']:
        info = task['files']['test']
        path = ROOT / info['path']
        assert sha(path) == info['sha256']
        sources[str(path.relative_to(ROOT))] = info['sha256']
        original.update({row['id']: row for row in map(json.loads, path.read_text().splitlines())})
    ends = {task['domain']: max(t['task'] for t in manifest['tasks'] if t['domain'] == task['domain'])
            for task in manifest['tasks']}
    cache = {}

    def load(label, domain):
        key = label, domain
        if key in cache:
            return cache[key]
        path = run / 'evaluations' / label / (domain + '_image.jsonl')
        name = str(path.relative_to(run))
        assert name in proof['artifact_sha256'], name
        digest = sha(path)
        assert digest == proof['artifact_sha256'][name], name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == 100 and len({r['id'] for r in rows}) == 100
        for row in rows:
            assert row['references'] == original[row['id']]['answers']
            assert math.isfinite(row['nll'])
            if 'scores' in row:
                assert row['scores']['exact_match'] in [0, 100]
        assert sha(path) == digest
        sources[str(path.relative_to(ROOT))] = digest
        cache[key] = {row['id']: row for row in rows}
        return cache[key]

    def paired(left, right, ids):
        assert set(left) == set(right)
        assert all(left[i]['references'] == right[i]['references'] and
                   left[i]['target_tokens'] == right[i]['target_tokens'] for i in ids)
        values = [right[i]['nll'] - left[i]['nll'] for i in ids]
        clusters = defaultdict(list)
        for ident, value in zip(ids, values):
            clusters[tuple(original[ident]['images'])].append(value)
        groups = list(clusters.values())
        rng = random.Random(42)
        samples = []
        for _ in range(2000):
            picked = [groups[rng.randrange(len(groups))] for _ in groups]
            samples.append(sum(sum(g) for g in picked) / sum(map(len, picked)))
        samples.sort()
        domain = next(t['domain'] for t in manifest['tasks'] if t['task'] == left[ids[0]]['native_task'])
        fixed = load('sequential_T19', domain)
        generated = [i for i in ids if 'scores' in fixed[i]]
        available_left = [i for i in generated if 'scores' in left[i]]
        available_right = [i for i in generated if 'scores' in right[i]]
        transitions = None
        if available_left == generated and available_right == generated:
            transitions = dict(retained=0, lost=0, gained=0, always_wrong=0)
            for ident in generated:
                a, b = bool(left[ident]['scores']['exact_match']), bool(right[ident]['scores']['exact_match'])
                transitions['retained' if a and b else 'lost' if a else 'gained' if b else 'always_wrong'] += 1
        return {'QA':len(ids), 'image_clusters':len(groups), 'mean_NLL_before':mean([left[i]['nll'] for i in ids]),
                'mean_NLL_after':mean([right[i]['nll'] for i in ids]), 'delta_NLL_after_minus_before':mean(values),
                'paired_image_cluster_95CI': [samples[49], samples[1949]],
                'generated_QA':len(generated), 'generation_available_before':len(available_left),
                'generation_available_after':len(available_right),
                'EM_correct_before':sum(bool(left[i]['scores']['exact_match']) for i in generated) if available_left == generated else None,
                'EM_correct_after':sum(bool(right[i]['scores']['exact_match']) for i in generated) if available_right == generated else None,
                'transitions':transitions,
                'interpretation':'chronological paired observation; uncorrected intervals; not domain-count causal estimate'}

    tasks = []
    for task in manifest['tasks']:
        t, domain = task['task'], task['domain']
        acquired = load(f'sequential_T{t:02d}', domain)
        end = load(f'sequential_T{ends[domain]:02d}', domain)
        final = load('sequential_T19', domain)
        isolated = load(f'matched_native_T{t:02d}', domain)
        before = load(f'pre_T{t:02d}', domain)
        ids = [i for i, row in acquired.items() if row['native_task'] == t]
        assert ids
        result = {'task':t, 'domain':domain, 'train_QA':task['files']['train']['count'],
                  'diagnostic_QA':len(ids), 'domain_last_stage':ends[domain],
                  'current_training':paired(before, acquired, ids),
                  'isolated_to_history_acquired':paired(isolated, acquired, ids),
                  'later_same_domain':paired(acquired, end, ids),
                  'later_cross_domain':paired(end, final, ids),
                  'acquired_to_final':paired(acquired, final, ids)}
        if ends[domain] < 19:
            label = f'sequential_T{ends[domain]+1:02d}'
            name = f'evaluations/{label}/{domain}_image.jsonl'
            if name in proof['artifact_sha256']:
                result['first_cross_domain_stage'] = paired(end, load(label, domain), ids)
            else:
                result['first_cross_domain_stage'] = None
                result['first_cross_domain_missing_reason'] = 'Original run did not evaluate old domain at this stage; no inferred cell.'
        tasks.append(result)

    domains = []
    for domain, end_stage in ends.items():
        end, final = load(f'sequential_T{end_stage:02d}', domain), load('sequential_T19', domain)
        ids = list(end)
        acquired = {i:load(f"sequential_T{row['native_task']:02d}", domain)[i] for i, row in end.items()}
        domains.append({'domain':domain, 'domain_end_stage':end_stage,
                        'acquired_to_domain_end':paired(acquired, end, ids),
                        'domain_end_to_final':paired(end, final, ids),
                        'acquired_to_final':paired(acquired, final, ids),
                        'note':'Acquired values combine different task acquisition times; not one model snapshot.'})
    out = {'status':'verified', 'manifest_sha256':sha(manifest_path), 'tasks':tasks, 'domains':domains,
           'sources':sources, 'limitations':[
               'Retrospective already-inspected 500NLL/155generation QA, not full21454 or an untouched confirmatory cohort.',
               'Chronological within/cross changes differ in number of later stages, exposure, content and order; no isolated abstract domain-count causality.',
               'Earlier independent vs sequential models have different prior training costs and initial functions; history benefit is not free.',
               'Loss uses original gold tokens; low NLL need not imply decoded accuracy. Generated subset and primary scoring remain unchanged.',
               'One seed/backbone/native diagnostic image protocol; not formal shared V2 model ranking.',
               'Paired image-cluster bootstrap has 2000 draws seed42, uncorrected across all contrasts.',
               'First cross-domain old-task cells are unmeasured in the original run and remain null; do not reconstruct them from another weight state.',
               'Intermediate generation covers only current native task. Domain-end EM/transitions for the full predeclared cohort are null unless all fixed IDs were actually generated; do not select an intersection subset.',
               'Two-anchor six-mechanism deterministic GPU study remains the required matched control, not replaced by this audit.']}
    lines = ['# 19阶段：同域后续学习与跨域后续学习的逐题拆分', '',
             '读取已完成确定性运行，核验预测文件与完成证明SHA、原始ID/答案、相同目标token数。不使用在跑的全量预测。', '',
             '**这是已有轨迹的事后描述，不是匹配任务数/预算的领域因果实验。** 两锚六机制GPU复核仍在等待。500题NLL与155题生成不是完整测试。', '',
             '| 领域 | NLL题数 | 后续同域NLL变化 | 后续跨域NLL变化 | 生成题数 | 刚学完→最终正确数 | 总丢失/新增 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for d in domains:
        a,b,c=d['acquired_to_domain_end'],d['domain_end_to_final'],d['acquired_to_final']
        lines.append(f"| {d['domain']} | {a['QA']} | {a['delta_NLL_after_minus_before']:+.4f} | {b['delta_NLL_after_minus_before']:+.4f} | {c['generated_QA']} | {c['EM_correct_before']}→{c['EM_correct_after']} | {c['transitions']['lost']}/{c['transitions']['gained']} |")
    lines += ['', '“刚学会时”组合不同任务各自刚完成时的模型输出，不是单一快照；“域结束→最终”才是两固定权重快照。不能将同域与跨域的差额解释为纯领域效应。', '',
              'NLL为后减前，负值表示原答案概率改善。完整19任务的当前学习、历史起点差异、后续同域/跨域NLL及95%区间见JSON。中间阶段只生成当时任务，域结束点没有全部固定生成ID，因此对应EM/转移保持null；上表只报告完整可测的“刚学完→最终”转移。原运行也没有测“首个跨域阶段的旧域”，保持null。不选取可用交集来冒充原固定队列，不重新解析答案。', '',
              '## 19个原生任务', '',
              '| 任务 | TRAIN | 生成QA | 刚完成→域结束→最终正确数 | 同域NLL变化 | 跨域NLL变化 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for t in tasks:
        a,b=t['later_same_domain'],t['later_cross_domain']
        lines.append(f"| T{t['task']:02d} {t['domain']} | {t['train_QA']} | {a['generated_QA']} | {a['EM_correct_before']}→{a['EM_correct_after']}→{b['EM_correct_after']} | {a['delta_NLL_after_minus_before']:+.4f} | {b['delta_NLL_after_minus_before']:+.4f} |")
    lines += ['', '## 局限', ''] + ['- '+v for v in out['limitations']]
    for name, digest in sources.items():
        assert sha(ROOT/name) == digest, name
    out['audit_source_sha256'] = sha(Path(__file__))
    (REPORT/'DOMAIN_BOUNDARY_AUDIT.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    (REPORT/'DOMAIN_BOUNDARY_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'domains':domains,'tasks':len(tasks),'source_files_checked':len(sources)},ensure_ascii=False))


if __name__ == '__main__':
    main()
