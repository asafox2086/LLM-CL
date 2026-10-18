"""Describe all sealed native cache views; no new inference or TEST selection."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import random

import onevision_state_cache_report as verified

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'summary/onevision_diagnostics_20261016'
REPORT = BASE/'functional_state_cache_20261018/v1'
DOMAINS = verified.DOMAINS


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(groups, rng):
    sample = [groups[rng.randrange(len(groups))] for _ in groups]
    return sum(map(sum, sample))/sum(map(len, sample))


def main():
    assert (REPORT/'build_complete.json').exists(), 'No partial-view analysis'
    verified.main()  # Actual source, all artifact, ID, gold, metric and update gates.
    proof = read(REPORT/'build_complete.json');run = ROOT/proof['run']
    spec = proof['spec'];views = spec['views']
    manifest = read(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    images = {}
    for task in manifest['tasks']:
        info = task['files']['test'];path = ROOT/info['path'];assert sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line);images[row['id']] = tuple(row['images'])
    rows, route_counts = {}, {}
    for view in views:
        rows[view], route_counts[view] = {}, {}
        for domain in DOMAINS:
            path = run/f'evaluations/{view}/{domain}_image.jsonl'
            assert sha(path) == proof['artifact_sha256'][str(path.relative_to(run))]
            data = [json.loads(line) for line in path.read_text().splitlines()]
            rows[view][domain] = {row['id']: row for row in data}
            route_counts[view][domain] = dict(Counter(row['route_state'] for row in data))
    reference = rows['seq_geometry']
    for domain in DOMAINS:
        for view in views:
            assert list(rows[view][domain]) == list(reference[domain])
        for row_id, old in reference[domain].items():
            twin = rows['seq_quality'][domain][row_id]
            fields = ['id', 'nll', 'references', 'prediction', 'scores']
            assert {k: old[k] for k in fields if k in old} == {k: twin[k] for k in fields if k in twin}
    contrasts = []
    for target, ref in [('cache_geometry', 'seq_geometry'), ('cache_quality', 'seq_geometry'),
                        ('cache_quality', 'cache_geometry')]:
        domain_results, all_groups = {}, {}
        for domain in DOMAINS:
            blocks = defaultdict(list);wins = losses = ties = changed_scored_routes = 0
            scored_nll_changes = [];lower_nll_lower_F1 = higher_nll_higher_F1 = 0
            old_rows, new_rows = rows[ref][domain], rows[target][domain]
            for row_id, old in old_rows.items():
                new = new_rows[row_id]
                if 'scores' not in old:
                    continue
                delta = new['scores']['token_f1']-old['scores']['token_f1']
                nll_delta = new['nll']-old['nll'];scored_nll_changes.append(nll_delta)
                lower_nll_lower_F1 += nll_delta < 0 and delta < 0
                higher_nll_higher_F1 += nll_delta > 0 and delta > 0
                blocks[images[row_id]].append(delta)
                wins += delta > 0;losses += delta < 0;ties += delta == 0
                changed_scored_routes += old['route_state'] != new['route_state']
            groups = list(blocks.values());all_groups[domain] = groups;rng = random.Random(42)
            draws = sorted(paired(groups, rng) for _ in range(10000))
            domain_results[domain] = dict(delta_F1=sum(map(sum, groups))/sum(map(len, groups)),
                image_95CI=[draws[249], draws[9749]],scored_QA=sum(map(len, groups)),
                F1_up=wins,F1_down=losses,F1_same=ties,changed_scored_routes=changed_scored_routes,
                mean_NLL_change_same_scored_QA=sum(scored_nll_changes)/len(scored_nll_changes),
                same_QA_lower_NLL_but_lower_F1=lower_nll_lower_F1,
                same_QA_higher_NLL_but_higher_F1=higher_nll_higher_F1,
                changed_routes_100NLL=sum(old_rows[i]['route_state'] != new_rows[i]['route_state'] for i in old_rows),
                mean_NLL_change_100=sum(new_rows[i]['nll']-old_rows[i]['nll'] for i in old_rows)/len(old_rows))
        rng = random.Random(42)
        draws = sorted(sum(paired(all_groups[d], rng) for d in DOMAINS)/5 for _ in range(10000))
        contrasts.append(dict(target=target,reference=ref,domains=domain_results,
            domain_macro_delta=sum(v['delta_F1'] for v in domain_results.values())/5,
            domain_macro_image_95CI=[draws[249],draws[9749]],
            uncertainty='Uncorrected image-cluster bootstrap, fixed trained model/seed; not multi-seed variation.'))
    summary = [r for r in read(REPORT/'VERIFIED_EVIDENCE.json')['rows'] if r['scope'] == 'diagnostic155']
    assert len(summary) == 4
    out = dict(CPU_only=True,new_optimizer_updates=0,new_predictions=0,
        build_proof_sha256=sha(REPORT/'build_complete.json'),analyzer_sha256=sha(Path(__file__)),
        views=summary,route_counts_500NLL=route_counts,contrasts=contrasts,
        all_four_conditions_reported=True,TEST_selection=False,diagnostic_only_not_full21454=True,
        limits=['Native AnyRes/BF16 diagnostics, not formalV2 main-table results.',
                '155 generated QA: natural10/medical73/geography32/finance10/traffic30; NLL routes use500.',
                'AP19 uses original task macro; confidence intervals here use domain macro, not AP19.',
                'Negative quality-rule effect applies to this fixed pool/input cohort, not every quality-based method.',
                'Neither a significant large full-test gain nor cross-domain source reuse is established.'])
    dest = REPORT/'first_four_view_outcome';dest.mkdir(exist_ok=True)
    (dest/'EVIDENCE.json').write_text(json.dumps(out,indent=2)+'\n')
    text = ['# 连续LoRA缓存四格：质量估计未带来额外收益', '',
        '- **探究的问题：** 避免新增专家冷启动后，保留旧功能状态和按TRAIN质量调用，能否改善最终表现？',
        '- **做的实验：** 相同原1313TRAIN／19阶段／93更新；全部19阶段A/B精确等于nativeSeq。固定四格、同500NLL／155生成，最多3个CPU状态。',
        '- **结果：** 下表为实际封存小诊断，全部条件保留。', '',
        '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏 | AP19 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        text.append('| '+row['view']+' | '+' | '.join(f"{row['domains'][d]:.3f}" for d in DOMAINS)+
                    f" | {row['domain_macro']:.3f} | {row['AP19']:.3f} |")
    text += ['', '| 对比 | 域宏差 | 图像95%区间 |', '| --- | ---: | --- |']
    for c in contrasts:
        lo,hi=c['domain_macro_image_95CI']
        text.append(f"| {c['target']} − {c['reference']} | {c['domain_macro_delta']:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    counts = Counter()
    for histogram in route_counts['cache_quality'].values():
        counts.update(histogram)
    quality_geo = contrasts[2]['domains']['geography']
    text += ['', '## 调用与目标不一致的实际观察', '',
        f"质量调用500题状态分布为 {dict(counts)}；这不是完全不使用输入，但其选择高度集中。几何调用仍使用状态12，质量调用本次未使用它。",
        f"相同32道遥感生成QA，质量相对几何的平均NLL变化{quality_geo['mean_NLL_change_same_scored_QA']:+.6f}，F1变化{quality_geo['delta_F1']:+.3f}pp；其中{quality_geo['same_QA_lower_NLL_but_lower_F1']}题同时出现NLL更低、F1更低。没有把100题NLL与32题F1当同一群体。",
        '这构成NLL排序不保证生成F1排序的局部反例；不证明全部差距来自代理目标，也未区分20条TRAIN锚的代表性、估计误差、源状态差异和生成决策。域宏配对区间均包含0，不能宣称总体稳定提升或稳定损害。']
    text += ['', '- **证明了什么：** 缓存几何调用的诊断点估计提高；TRAIN-NLL质量调用相对此几何调用的实际效果见区间，不能宣称质量模块成功。两单状态条件完全相同。行为空间与TRAIN质量估计都不自动保证生成F1；本实验未分离质量估计误差和NLL／生成目标差异。',
        '- **改进的方法：** 暂不叠加模块；按用户新要求优先进行独立五格初始化来源控制，分清完整旧输出与A方向的贡献。',
        '- **改进的效果：** 五格尚待，不预填。两个缓存完整21454确认仍保留、当前后移；没有用小诊断选赢家或删除负结果。', '',
        '逐域配对区间、增／降／不变题数、500题路由变化及NLL差见EVIDENCE.json。区间未经多重比较校正，仅固定权重下图像重采样；不能冒充多seed或完整测试证据。FWT／BWT不从最终诊断推算。实际资源见原run的决策、纯TRAIN窗口和内存报告，不将缓存张量字节称为总内存增长或轻量优势。']
    (dest/'RESEARCH_FINDINGS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(dict(views=summary,contrasts=[{k:v for k,v in c.items() if k!='domains'} for c in contrasts]),indent=2))


if __name__ == '__main__':
    main()
