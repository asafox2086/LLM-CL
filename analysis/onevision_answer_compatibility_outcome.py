"""All five sealed feedback conditions: realized routes, paired F1 and costs.

Never reads incomplete GPU suffixes. No model, new predictions or TEST tuning.
"""
from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import random

import onevision_answer_compatibility_report as verified
from onevision_state_cache_outcome import paired

ROOT, REPORT, SOURCE = verified.ROOT, verified.REPORT, verified.SOURCE
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']
VIEWS = ['seq', 'cache_geometry', 'cache_NLL', 'cache_AnswerF1', 'cache_AnswerF1_metric']


def read(path):
    return json.loads(path.read_text())


def main():
    proof_path = REPORT/'build_complete.json'
    if not proof_path.exists():
        print(json.dumps(dict(pending_complete_build=True, partial_predictions_read=False)))
        return
    verified.main()
    spec = read(REPORT/'CPU_PLAN_CHECK.json')['spec']
    assert verified.complete('build', None, spec)
    source_proof = read(SOURCE/'build_complete.json')
    source = verified.verify(source_proof, ROOT/'exp/onevision_state_cache_20261018/v1')
    proof = read(proof_path)
    run = verified.verify(proof, ROOT/'exp/onevision_answer_compatibility_20261018/v1')
    output = REPORT/'all_five_outcome'
    fingerprint = dict(build_proof_sha256=verified.sha(proof_path),
                       source_proof_sha256=verified.sha(SOURCE/'build_complete.json'),
                       analyzer_sha256=verified.sha(Path(__file__)))
    if (output/'EVIDENCE.json').exists():
        prior = read(output/'EVIDENCE.json')
        if all(prior.get(k) == v for k, v in fingerprint.items()):
            print(json.dumps(dict(unchanged_complete_proofs=True, no_repeated_analysis=True)))
            return
    score_report = read(REPORT/'VERIFIED_EVIDENCE.json')
    assert list(score_report['diagnostic_conditions']) == VIEWS
    fit = read(REPORT/'TRAIN_FIT_VERIFIED.json')
    for name, digest in fit['TRAIN_artifact_sha256'].items():
        assert verified.sha(run/name) == digest
    manifest = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert verified.sha(manifest) == spec['parent_spec']['manifest_sha256']
    images = {}
    for task in read(manifest)['tasks']:
        info = task['files']['test']
        path = ROOT/info['path']
        assert verified.sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            images[row['id']] = tuple(row['images'])
    rows, routes = {}, {}
    for view, label, directory, artifact_proof in [
            ('seq', 'seq_geometry', source, source_proof),
            ('cache_geometry', 'cache_geometry', source, source_proof),
            ('cache_NLL', 'cache_quality', source, source_proof),
            ('cache_AnswerF1', 'cache_AnswerF1', run, proof),
            ('cache_AnswerF1_metric', 'cache_AnswerF1_metric', run, proof)]:
        rows[view], routes[view] = {}, {}
        for domain in DOMAINS:
            path = directory/f'evaluations/{label}/{domain}_image.jsonl'
            assert verified.sha(path) == artifact_proof['artifact_sha256'][str(path.relative_to(directory))]
            original = [json.loads(line) for line in path.read_text().splitlines()]
            rows[view][domain] = {r['id']: r for r in original}
            assert len(original) == len(rows[view][domain]) == 100
            routes[view][domain] = dict(Counter(r['route_state'] for r in original))
            assert list(rows[view][domain]) == list(rows['seq'][domain])
    contrasts = []
    for reference, target in itertools.combinations(VIEWS, 2):
        domains, all_groups = {}, {}
        for domain in DOMAINS:
            old, new = rows[reference][domain], rows[target][domain]
            groups = defaultdict(list)
            up = down = same = changed = changed_generated = 0
            for ident, first in old.items():
                second = new[ident]
                different_route = first['route_state'] != second['route_state']
                changed += different_route
                if not different_route:
                    fields = ['nll', 'prediction', 'scores', 'references']
                    assert {k: first.get(k) for k in fields} == {k: second.get(k) for k in fields}, \
                        'Same selected frozen function produced incompatible semantic results'
                if first.get('prediction') is None:
                    assert second.get('prediction') is None
                    continue
                assert second.get('prediction') is not None
                delta = second['scores']['token_f1']-first['scores']['token_f1']
                groups[images[ident]].append(delta)
                up += delta > 0
                down += delta < 0
                same += delta == 0
                changed_generated += different_route
            blocks = list(groups.values())
            all_groups[domain] = blocks
            rng = random.Random(42)
            draws = sorted(paired(blocks, rng) for _ in range(10000))
            domains[domain] = dict(delta_F1=sum(map(sum, blocks))/sum(map(len, blocks)),
                image_95CI=[draws[249], draws[9749]], generated_QA=up+down+same,
                F1_up=up, F1_down=down, F1_same=same,
                route_changes_100=changed, route_changes_generated=changed_generated)
        rng = random.Random(42)
        draws = sorted(sum(paired(all_groups[d], rng) for d in DOMAINS)/5 for _ in range(10000))
        contrasts.append(dict(target=target, reference=reference, domains=domains,
            domain_macro_delta=sum(d['delta_F1'] for d in domains.values())/5,
            domain_macro_image_95CI=[draws[249], draws[9749]],
            all500_route_changes=sum(d['route_changes_100'] for d in domains.values()),
            all155_F1_up=sum(d['F1_up'] for d in domains.values()),
            all155_F1_down=sum(d['F1_down'] for d in domains.values()),
            all155_F1_same=sum(d['F1_same'] for d in domains.values())))
    resource = read(run/'memory_summary.json')
    snapshots = [json.loads(line) for line in (run/'resources.jsonl').read_text().splitlines()]
    metric_window = [r for r in snapshots if r['phase'] in ['CPU_router_TRAIN_before', 'CPU_router_TRAIN_after']]
    assert len(metric_window) == 2 and len({r['pid'] for r in metric_window}) == 1
    result = dict(CPU_only=True, **fingerprint, new_model_updates=0, new_predictions=0,
        conditions=score_report['diagnostic_conditions'], routes_500=routes, contrasts=contrasts,
        all10_pairwise_contrasts_reported=True, TRAIN_fit=fit,
        actual_process_memory=resource, CPU_metric_window=metric_window,
        resource_scope='Same process: control+60 TRAIN generations+CPU head+both diagnostics; not isolated model TRAIN peak.',
        limits=['500 NLL/155 generation diagnostic only; full21454 and formal V2 ranking remain separate.',
                'Uncorrected image bootstrap, fixed trained seed42. Domain-macro intervals are not AP19 intervals.',
                'Actual parameter change is not necessarily realized hard-route or F1 intervention.',
                'One process supplies both new views, so no independent per-view memory growth comparison.',
                'Both new full confirmations and all negative conditions retained; no TEST tuning or selection.'])
    output.mkdir(exist_ok=True)
    (output/'EVIDENCE.json').write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# TRAIN-F1反馈与学习距离：五格完整诊断', '',
        '- **探究的问题：** 直接用TRAIN输出F1监督匹配，能否修复NLL代理与实际生成质量不一致？学习距离是否真正改变调用和得分？',
        '- **做的实验：** 固定同一原连续1313TRAIN／19阶段／93更新、3个状态、20TRAIN锚，新增60条TRAIN生成及40次1792参数CPU更新。全部五格同500NLL／155生成；新原几何全量精确重现门槛通过，实际原评分／SHA／路由逐项核验。',
        '- **结果：** 以下全部条件及10组配对均保留，前三行明确引用原封存同协议，后两行为新封存；不选择赢家。', '',
        '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏 | AP19 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for view in VIEWS:
        s = result['conditions'][view]['scores']
        values = [s['domains'][d] for d in DOMAINS]+[s['domain_macro'], s['AP19']]
        lines.append('| '+view+' | '+' | '.join(f'{v:.3f}' for v in values)+' |')
    lines += ['', '| 对比 | 域宏差 | 图像95%区间 | 500题改变调用 |', '| --- | ---: | --- | ---: |']
    for c in contrasts:
        lo, hi = c['domain_macro_image_95CI']
        lines.append(f"| {c['target']} − {c['reference']} | {c['domain_macro_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {c['all500_route_changes']} |")
    metric = next(c for c in contrasts if c['target'] == 'cache_AnswerF1_metric' and c['reference'] == 'cache_AnswerF1')
    lines += ['', f"学习距离相对固定F1核，在500题中实际改变{metric['all500_route_changes']}次调用；155生成题提高／降低／持平为{metric['all155_F1_up']}／{metric['all155_F1_down']}／{metric['all155_F1_same']}。1792维及40步真实变化已经独立精确重放，但是否形成有效功能干预应以这些实际调用／输出为准。",
        '', '- **证明了什么：** 表中差异只约束这组固定池／特征／锚／核及诊断样本。若硬调用完全不变，只能认定本次学习没有实现新功能干预，不能反推任意可学习表示无效。即便点估计提高，仍需完整测试；NLL/F1目标更换也不自动保证收益。',
        '- **改进的方法：** 依据TRAIN拟合与真实调用区分反馈稀疏、邻域估计和硬决策未改变；暂不改变冻结参数、样本或方法，全部预定完整确认继续。',
        '- **改进的效果：** 当前仅上表诊断结果；完整21454、完整FWT/BWT及实际部署资源优势未由此证明。单进程总内存包含原控制／60TRAIN／两诊断，不能作两新方法各自纯TRAIN内存比较。',
        '', '[全部逐域配对／调用／参数／CPU-GPU原始内存](EVIDENCE.json) · [冻结协议](../PROTOCOL.md) · [TRAIN实际启用](../TRAIN_FIT_FINDINGS.md)']
    (output/'RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(conditions=result['conditions'], metric_intervention=metric,
                         contrasts=[{k: v for k, v in c.items() if k != 'domains'} for c in contrasts])))


if __name__ == '__main__':
    main()
