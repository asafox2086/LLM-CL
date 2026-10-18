"""All paired realized interventions among six sealed diagnostic conditions.

Point estimates only: no TEST choices, optimizer, new prediction or GPU model.
"""
from collections import defaultdict
import itertools
import json
from pathlib import Path

import onevision_feedback_coverage_report as verified
from answer_f1 import answer_f1

ROOT, REPORT = verified.ROOT, verified.REPORT


def main():
    spec = verified.read(REPORT/'CPU_WORKER_PLAN.json')['spec']
    assert verified.complete('build', None, spec)
    verified.main()
    old = verified.previous
    source = old.SOURCE
    source_proof = verified.read(source/'build_complete.json')
    source_run = old.verify(source_proof, ROOT/'exp/onevision_state_cache_20261018/v1')
    old_proof = verified.read(old.REPORT/'build_complete.json')
    old_run = old.verify(old_proof, ROOT/'exp/onevision_answer_compatibility_20261018/v1')
    proof = verified.read(REPORT/'build_complete.json')
    run = old.verify(proof, ROOT/'exp/onevision_feedback_coverage_20261018/v1')
    fingerprints = {str(p.relative_to(ROOT)): verified.sha(p) for p in
                    [source/'build_complete.json', old.REPORT/'build_complete.json', REPORT/'build_complete.json']}
    directory = REPORT/'all_six_outcome'
    destination = directory/'EVIDENCE.json'
    if destination.exists():
        prior = verified.read(destination)
        if prior['completion_sha256'] == fingerprints and prior['analyzer_sha256'] == verified.sha(Path(__file__)):
            print(json.dumps(dict(unchanged_proofs=True, repeated_analysis=False)))
            return
    descriptors = [('seq', 'seq_geometry', source_run, source_proof),
        ('cache_geometry', 'cache_geometry', source_run, source_proof),
        ('old_F1', 'cache_AnswerF1', old_run, old_proof),
        ('old_F1_metric', 'cache_AnswerF1_metric', old_run, old_proof),
        ('coverage_F1', 'coverage_AnswerF1', run, proof),
        ('coverage_F1_metric', 'coverage_AnswerF1_metric', run, proof)]
    frames, scores = {}, {}
    for name, label, folder, artifact_proof in descriptors:
        flat, per_task, domains = [], defaultdict(list), {}
        for domain in ['natural', 'medical', 'geography', 'finance', 'traffic']:
            path = folder/f'evaluations/{label}/{domain}_image.jsonl'
            assert verified.sha(path) == artifact_proof['artifact_sha256'][str(path.relative_to(folder))]
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == len({row['id'] for row in rows}) == 100
            values = []
            for row in rows:
                if row.get('prediction') is not None:
                    f1 = answer_f1(row['prediction'], row['references'])
                    assert f1 == row['scores']['token_f1']
                    values.append(f1);per_task[row['native_task']].append(f1)
            domains[domain] = sum(values)/len(values)
            flat += rows
        frames[name] = flat
        assert len(per_task) == 19 and sum(map(len, per_task.values())) == 155
        scores[name] = dict(domains=domains, AP19=sum(sum(v)/len(v) for v in per_task.values())/19,
                            domain_macro=sum(domains.values())/5)
    contrasts = []
    for first, second in itertools.combinations(frames, 2):
        changed = up = down = same = text_changed = 0
        for a, b in zip(frames[first], frames[second]):
            assert a['id'] == b['id'] and a['references'] == b['references']
            different = a['route_state'] != b['route_state']
            changed += different
            if not different:
                assert all(a.get(k) == b.get(k) for k in ['prediction', 'scores', 'nll', 'references'])
            assert (a.get('prediction') is None) == (b.get('prediction') is None)
            if a.get('prediction') is not None:
                delta = b['scores']['token_f1']-a['scores']['token_f1']
                up += delta > 0;down += delta < 0;same += delta == 0
                text_changed += a['prediction'] != b['prediction']
        contrasts.append(dict(reference=first, target=second, all500_route_changes=changed,
            generated155_F1_up=up, generated155_F1_down=down, generated155_F1_same=same,
            generated155_text_changes=text_changed, AP19_delta=scores[second]['AP19']-scores[first]['AP19'],
            domain_macro_delta=scores[second]['domain_macro']-scores[first]['domain_macro']))
    result = dict(CPU_only=True, completion_sha256=fingerprints, analyzer_sha256=verified.sha(Path(__file__)),
        scores=scores, contrasts=contrasts, all15_pairwise_comparisons_reported=True,
        point_estimates_only_not_significance_tests=True, full_FWT=None, full_BWT=None,
        limits=['All six conditions use the same500 NLL/155 generation; not full21454 or formal V2.',
                'No source, memory, hyperparameter, scheduler or winner is selected from these TEST results.',
                'Both new full confirmations and all previous negative controls retained.',
                'Coverage also changes anchor content; not pure task-count causality.',
                'Full gain, generalization significance and memory superiority remain unproven.'])
    directory.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 反馈覆盖与距离学习：完整六条件诊断', '',
        '- **探究的问题：** 同20条反馈覆盖全部原生任务，能否提高实际调用和回答质量？',
        '- **做的实验：** 共享原1313TRAIN／19阶段／93更新的同一3状态；新旧20锚各60TRAIN生成／40匹配更新。新原500／155精确重现门槛通过，原评分、逐题ID及完整SHA独立核验。',
        '- **结果：** 下表全部六条件及全部15配对保留。', '',
        '| 条件 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP19 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name, item in scores.items():
        lines.append('| '+name+' | '+' | '.join(f'{item["domains"][d]:.3f}' for d in ['natural','medical','geography','finance','traffic'])+f' | {item["AP19"]:.3f} |')
    lines += ['', '| 对比 | AP19差 | 500题调用改变 | 155题F1升／降／平 |', '| --- | ---: | ---: | --- |']
    for item in contrasts:
        lines.append(f'| {item["target"]} − {item["reference"]} | {item["AP19_delta"]:+.3f} | {item["all500_route_changes"]} | {item["generated155_F1_up"]}／{item["generated155_F1_down"]}／{item["generated155_F1_same"]} |')
    lines += ['', '- **证明了什么：** 只约束本次固定池／锚／匹配器和同一诊断。真实参数变化、调用变化、回答变化分别计数；点估计不冒充显著性或完整五域确认。',
        '- **改进的方法：** 本次仅修正反馈分配；不据TEST重新筛状态／改参数。两个新完整确认与所有原对照保留。',
        '- **改进的效果：** 本表为155诊断；尚未证明稳定大幅提升，未填正式Ours。完整FWT/BWT仍需合法前缀格，缺失保持null。',
        '', '[完整逐题变化计数／全部对比／来源SHA](EVIDENCE.json) · [冻结协议](../PROTOCOL.md)']
    (directory/'RESEARCH_FINDINGS.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(scores=scores, learned_vs_fixed=contrasts[-1])))


if __name__ == '__main__':
    main()
