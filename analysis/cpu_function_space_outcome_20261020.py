"""Summarize sealed CPU function-space evidence; no models or new forwards."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    e = json.loads((REPORT / 'EVIDENCE.json').read_text())
    c = json.loads((REPORT / 'COMPLETE.json').read_text())
    v = json.loads((REPORT / 'VERIFIED_RESULTS.json').read_text())
    ready = json.loads((REPORT / 'READY.json').read_text())
    assert c['status'] == 'CPU_TRAIN_FUNCTION_SPACE_COMPLETE'
    assert v['status'] == 'TEN_QA_CPU_FUNCTION_SPACE_INDEPENDENTLY_VERIFIED'
    assert c['evidence_sha256'] == v['evidence_sha256'] == sha(REPORT / 'EVIDENCE.json')
    assert e['source_sha256'] == c['source_sha256'] == ready['source_sha256'] == sha(ROOT / 'analysis/cpu_function_space_20261020_v2.py')
    assert v['report_source_sha256'] == sha(ROOT / 'analysis/cpu_function_space_report_20261020_v2.py')
    assert e['CPU_only'] and e['new_optimizer_updates'] == 0 and e['model_forward_calls'] == 11
    assert not e['official_TEST_read'] and not e['DEV_answers_used']
    assert e['teacher_has_trained_on_all_probe_QA']
    assert e['first_CPU_original_folded_NLL_exact'] and e['model_parameter_SHA_unchanged']
    assert e['seconds'] < 7200
    wanted = {x['id']: x for x in ready['selected']}
    records = e['records']
    assert len(records) == 10 and {x['id'] for x in records} == set(wanted)
    assert len({x['group'] for x in records}) == 10
    for x in records:
        assert all(x[k] == wanted[x['id']][k] for k in ['role', 'group', 'domain', 'task'])
        assert len(x['observations']) == len({y['name'] for y in x['observations']}) == 48
        assert math.isfinite(x['NLL'])
    probes = [x for x in records if x['role'] == 'probe']
    assert len(probes) == len({x['domain'] for x in probes}) == 5

    def errors(row, suffix=None):
        items = [y for y in row['observations'] if suffix is None or y['name'].endswith(suffix)]
        if suffix is not None:
            assert len(items) == 24
        total = math.fsum(y['reference_energy'] for y in items)
        assert total > 0
        return {k: math.sqrt(math.fsum(y[k] for y in items) / total)
                for k in ['weight_rank8_error', 'function_rank8_error']}

    by_domain = {x['domain']: errors(x) for x in probes}
    macro = {k: math.fsum(x[k] for x in by_domain.values()) / 5 for k in next(iter(by_domain.values()))}
    for ours, published in [('weight_rank8_error', 'weight_rank8_relative_error'),
                            ('function_rank8_error', 'function_rank8_relative_error')]:
        assert macro[ours] == v['domain_macro_relative_error'][published]
    families = {}
    for suffix in ['self_attn.q_proj', 'mlp.down_proj']:
        rows = [errors(x, suffix) for x in probes]
        families[suffix] = {k: math.fsum(x[k] for x in rows) / 5 for k in macro}
    pairs = [y for x in probes for y in x['observations']]
    counts = {
        'function_lower_error': sum(y['function_rank8_error'] < y['weight_rank8_error'] for y in pairs),
        'function_higher_error': sum(y['function_rank8_error'] > y['weight_rank8_error'] for y in pairs),
        'equal_error': sum(y['function_rank8_error'] == y['weight_rank8_error'] for y in pairs),
        'total_layer_QA_pairs': len(pairs),
    }
    calibration = {x['domain']: x['target_tokens'] for x in records if x['role'] == 'calibration'}
    assert len(calibration) == 5
    result = {
        'source_sha256': sha(__file__),
        'evidence_sha256': sha(REPORT / 'EVIDENCE.json'),
        'independent_report_sha256': sha(REPORT / 'VERIFIED_RESULTS.json'),
        'by_domain': by_domain,
        'domain_macro': macro,
        'relative_error_reduction': 1 - macro['function_rank8_error'] / macro['weight_rank8_error'],
        'secondary_family_domain_macros': families,
        'secondary_pair_counts': counts,
        'calibration_answer_predicting_positions': calibration,
        'calibration_total_positions': sum(calibration.values()),
        'measurement_seconds': e['seconds'],
        'new_model_forwards': 0,
        'new_model_updates': 0,
        'AnswerF1_or_deployment_advantage_proven': False,
    }
    (REPORT / 'OUTCOME.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    labels = {'natural': '自然', 'medical': '医学', 'geography': '遥感', 'finance': '金融', 'traffic': '交通'}
    lines = [
        '# K60：相同小空间，按输入保留作用比只按权重保留得更好', '',
        '## 问题、实验与结论', '',
        '| 探究的问题 | 做了什么实验，结果是什么 | 结论 |',
        '| --- | --- | --- |',
        '| 权重不容易低秩近似，是否就不能用小空间保留它的作用？ | 对已训练OSFT的48处更新，同为rank8，比较权重SVD与用5条校准输入拟合的输出方向；另外5个图像组的平均局部相对误差49.98%→37.33%。 | 这些输入上，选择空间的依据确实有影响；不能只看权重能量判断功能可近似性。 |',
        '| 是否所有位置都变好、可以直接删参数？ | 240个“层×探针”中224个误差降低、16个升高；没有构造或运行删减后的完整模型。 | 不能据此宣布参数无用、F1提高或可部署；不能把局部近似当完整压缩成果。 |', '',
        '## 五个探针的完整结果', '',
        '| 领域（各1条探针） | 权重方向rank8误差 | 输入拟合方向rank8误差 |',
        '| --- | ---: | ---: |',
    ]
    for domain, x in by_domain.items():
        lines.append(f"| {labels[domain]} | {100*x['weight_rank8_error']:.3f}% | {100*x['function_rank8_error']:.3f}% |")
    lines += [f"| 五域等权平均 | {100*macro['weight_rank8_error']:.3f}% | {100*macro['function_rank8_error']:.3f}% |", '',
        '先汇总每题48矩阵的误差平方／参考能量，再开平方，最后五域等权。参考为系数作用ΔW·X的FP64几何量；不是原FP16输出逐值差、答案误差或Answer F1。两条件使用相同分析精度。', '',
        '次要分解：q投影平均相对误差95.685%→65.872%，FFN down投影38.900%→30.903%；两类各先在题内汇总24层，再五域平均。它们不衡量两类位置对答案的重要性。', '',
        '## 必须保留的边界', '',
        '- Teacher已学习完整1313TRAIN，包含这10条；探针只是在拟合近似时未使用的图像组，不是teacher未见数据。每域仅1条探针，不推断总体显著性。',
        '- 校准按5条QA合并，答案预测位置分别为医学3、自然3、遥感3、金融2、交通73，共84。没有做token均衡，不能称五域等权校准；主结果的探针汇总才是五域等权。',
        '- 只测q/down两个位置类型、仅答案预测位置和局部线性作用，没有跨层误差传播、自由生成或压缩模型F1。权重谱K59与本轮重建上下文不同，不把两者数字直接混算。',
        '- 模型参数前后SHA不变；原CPU／折叠预定题NLL精确，全部10QA×48记录及源/权重/原图SHA已绑定。v1故障与独立上下文修复保留，没有拼接旧失败结果。',
        '- V2测量5276.18秒，约87.94分钟CPU、11次前向、0优化器更新；之前修复门槛1122.4秒及v1失败另计，不把测量耗时当整个研究成本。',
        '- OBSERVED_CPU_RESOURCE_SNAPSHOT只是一份运行中资源观察：RSS约5.55GiB、当时HWM约6.04GiB，另有PSS；不是训练前后增长、完整运行峰值或部署RAM。', '',
        '## 下一行动', '',
        '本轮支持继续检验按输入功能选择压缩空间，借鉴的是SVD-LLM等已有视角，不作为原创结论。只压更新仍保留完整基座，不能实现端侧目标；不再把小adapter文件当部署收益。', '',
        '原GPU1四条件小测保持冻结并等待实际资源，不以本轮探针更改rank、位置、学习率或筛选条件。若后续进入完整结构压缩，必须真正缩小矩阵/模型，以相同FIT训练的teacher及仅FIT校准，在原独立100DEV上筛选，再以完整1313TRAIN重训和21454TEST确认；现成正式teacher已见全部1313，不能直接拿它在该DEV上的表现挑方案。本轮没有启动该压缩实验。', '',
        '完整五域Answer F1与端侧显著优势均未实现，主表Ours及未测部署指标保持空白。', '',
        '[原独立报告](VERIFIED_RESULTS.md) · [完整480条逐题逐层记录](EVIDENCE.json) · [二次汇总](OUTCOME.json) · [预定协议](PROTOCOL.md)',
    ]
    (REPORT / 'FINDINGS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'domain_macro': macro, 'pairs': counts, 'measurement_seconds': e['seconds']}))


if __name__ == '__main__':
    main()
