"""Read-only completed knowledge-fit controls; no GPU or fact/QA selection.

Post-hoc author-triple strata diagnose whether literal locality alone resolves
the negative transfer result. Literal overlap does NOT establish answerability.
"""
import hashlib
import json
import math
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

from answer_f1 import answer_f1

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/onevision_diagnostics_20261016'
CASES = ['rank1_correct', 'rank1_permuted', 'rank8_correct', 'rank8_permuted']
GROUPS = ['same_relation_fixed_tail', 'relation_only', 'relation_absent']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized(text):
    return ''.join(c for c in unicodedata.normalize('NFKC', text).casefold() if c.isalnum())


def locality(triple, facts):
    relation, tail = map(normalized, triple[1:])
    both = any(relation == normalized(f['relation']) and tail == normalized(f['value']) for f in facts)
    same = any(relation == normalized(f['relation']) for f in facts)
    return 'same_relation_fixed_tail' if both else 'relation_only' if same else 'relation_absent'


def paired(before, after, original):
    assert before and [r['id'] for r in before] == [r['id'] for r in after]
    clusters = defaultdict(list)
    changes = []
    for a, b in zip(before, after):
        value = b['scores']['token_f1']-a['scores']['token_f1']
        changes.append(value);clusters[tuple(original[a['id']]['images'])].append(value)
    groups = list(clusters.values());rng = random.Random(42);samples = []
    for _ in range(2000):
        selected = [groups[rng.randrange(len(groups))] for _ in groups]
        samples.append(sum(map(sum, selected))/sum(map(len, selected)))
    samples.sort()
    return {'QA': len(changes), 'image_clusters': len(groups),
            'before_F1': sum(r['scores']['token_f1'] for r in before)/len(before),
            'after_F1': sum(r['scores']['token_f1'] for r in after)/len(after),
            'delta_pp': sum(changes)/len(changes), 'image_cluster_95CI': [samples[49], samples[1949]],
            'quality_increased': sum(v > 0 for v in changes), 'quality_decreased': sum(v < 0 for v in changes),
            'quality_unchanged': sum(v == 0 for v in changes), 'uncorrected': True,
            'scope': 'Paired recipient-image bootstrap conditional on fixed weights, not seed-level inference'}


def checks():
    facts = [{'subject': 'lung', 'relation': 'belong to', 'value': 'respiratory system'}]
    assert locality(['vhead', 'BELONG TO', 'Respiratory System'], facts) == GROUPS[0]
    assert locality(['vhead', 'belong to', 'digestive system'], facts) == GROUPS[1]
    assert locality(['vhead', 'function', 'respiratory system'], facts) == GROUPS[2]
    ids = [{'id': 'x', 'scores': {'token_f1': 10.}}, {'id': 'y', 'scores': {'token_f1': 20.}}]
    images = {'x': {'images': ['same-image']}, 'y': {'images': ['same-image']}}
    unchanged = paired(ids, ids, images)
    assert unchanged['image_clusters'] == 1 and unchanged['delta_pp'] == 0 and unchanged['image_cluster_95CI'] == [0., 0.]
    try:
        paired(ids, list(reversed(ids)), images)
        raise RuntimeError('Changed ID order was accepted')
    except AssertionError:
        pass
    return ['Three exhaustive literal strata, independent of gold scores',
            'Same-image questions are paired clusters, not independent pseudo-replicates',
            'Changed outcome ID order rejected']


def main():
    tests = checks();sources = {str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))}

    def read(path):
        sources[str(path.relative_to(ROOT))] = sha(path)
        return json.loads(path.read_text())

    lock = read(ROOT/'summary/primary_metric_20261017/metric_lock.json')
    assert sha(ROOT/'analysis/answer_f1.py') == lock['metric_sha256'] and lock['option_mapping'] is False
    sources['analysis/answer_f1.py'] = lock['metric_sha256']
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    manifest = read(manifest_path)
    assert sha(manifest_path) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    original = {}
    for task in manifest['tasks']:
        if task['domain'] != 'medical':
            continue
        path = ROOT/task['files']['test']['path'];assert sha(path) == task['files']['test']['sha256']
        sources[str(path.relative_to(ROOT))] = sha(path)
        original.update({r['id']: dict(r, _task=task['task']) for r in map(json.loads, path.read_text().splitlines())})
    raw = {str(r['qid']): r for r in read(ROOT/'data/SLAKE/raw/test.json')}
    pool_path = ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json';pool = read(pool_path)
    assert len(pool['correct']) == 16 and pool['control_version'] == 2
    assignment = {}
    for identity, row in original.items():
        if row['_task'] != 11:
            continue
        source = raw[str(row['source_id'])]
        assert source['question'].strip() == row['question'].strip() and source['answer'].strip() == row['answer'].strip()
        assignment[identity] = locality(source['triple'], pool['correct'])
    assert len(assignment) == 267
    assert [sum(v == group for v in assignment.values()) for group in GROUPS] == [31, 40, 196]
    cache = {};recall = []
    for case in CASES:
        proof = read(REPORT/'knowledge_fit'/f'{case}_complete.json');assert proof['complete']
        spec = proof['spec'];directory = ROOT/proof['run']
        assert spec['pool_sha256'] == sha(pool_path) and spec['auxiliary_exposures_at64'] == 512
        for relative, digest in spec['source_sha256'].items():
            assert sha(ROOT/relative) == digest;sources[relative] = digest
        for step in [0, 8, 64]:
            for group in ['medical', 'medical_KG']:
                path = directory/'evaluations'/f'clinical_update_{step:02d}'/f'{group}_image.jsonl'
                assert sha(path) == proof['artifact_sha256'][str(path.relative_to(directory))]
                sources[str(path.relative_to(ROOT))] = sha(path)
                records = list(map(json.loads, path.read_text().splitlines()))
                assert [r['id'] for r in records] == spec['clinical_ids'][group]
                for row in records:
                    assert row['references'] == original[row['id']]['answers'] and math.isfinite(row['nll'])
                    assert answer_f1(row['prediction'], row['references']) == row['scores']['token_f1']
                cache[case, step, group] = records
            path = directory/'evaluations'/f'update_{step:02d}'/'relevant_true_text.jsonl'
            assert sha(path) == proof['artifact_sha256'][str(path.relative_to(directory))]
            sources[str(path.relative_to(ROOT))] = sha(path)
            facts = list(map(json.loads, path.read_text().splitlines()));assert len(facts) == 32
            for template in ['original', 'paraphrase']:
                selected = [r for r in facts if r['template'] == template];assert len(selected) == 16
                assert all(answer_f1(r['prediction'], [r['reference']]) == r['scores']['token_f1'] for r in selected)
                recall.append({'case': case, 'step': step, 'template': template,
                               'true_facts_F1': sum(r['scores']['token_f1'] for r in selected)/16})
    initial = []
    for group in ['medical', 'medical_KG']:
        canonical = cache[CASES[0], 0, group]
        for case in CASES[1:]:
            other = cache[case, 0, group]
            assert [(r['id'], r['references'], r['prediction'], r['scores']['token_f1'], r['nll']) for r in canonical] == [
                (r['id'], r['references'], r['prediction'], r['scores']['token_f1'], r['nll']) for r in other]
            initial.append({'case': case, 'group': group, 'QA': len(other), 'predictions_scores_NLL_exact': True})
    contrasts = []
    for rank in [1, 8]:
        for step in [8, 64]:
            for group in GROUPS:
                take = lambda rows: [r for r in rows if assignment[r['id']] == group]
                base = take(cache[f'rank{rank}_correct', 0, 'medical_KG'])
                correct = take(cache[f'rank{rank}_correct', step, 'medical_KG'])
                permuted = take(cache[f'rank{rank}_permuted', step, 'medical_KG'])
                contrasts.append({'rank': rank, 'step': step, 'stratum': group,
                                  'base_to_correct': paired(base, correct, original),
                                  'base_to_permuted': paired(base, permuted, original),
                                  'permuted_to_correct': paired(permuted, correct, original)})
    result = {'CPU_only': True, 'source_sha256': sources, 'CPU_checks': tests, 'common_initial_exact': initial,
              'mutually_exclusive_groups': {g: [i for i, v in assignment.items() if v == g] for g in GROUPS},
              'true_fact_recall': recall, 'contrasts': contrasts,
              'limits': ['Retrospective author-triple strata, not a predeclared holdout confirmation or semantic support annotation.',
                         'Relation/tail overlap does not establish matching image/head entity, fact sufficiency, aliases or answerability.',
                         'All strata/8 and64 steps/both ranks reported; no facts, method or budget selected from these scores.',
                         'Conditional single-fit image-cluster intervals, no multiple-testing correction or seed replication.',
                         'Medical100 and nativeKG267 overlap12, not367 independent QA. Only nativeKG267 is stratified here.',
                         'Successful text fact recall plus weak native QA transfer does not identify binding as the unique cause.',
                         'This is an auxiliary512-exposure intervention, not the formal1epoch1313-QA CL experiment or full21454 test.']}
    for path, expected in sources.items():
        assert sha(ROOT/path) == expected
    (REPORT/'KNOWLEDGE_LOCALITY_EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    text = ['# 记住知识后，字面相关题是否改善？', '',
            '只读四条已完成知识轨迹；不新增事实、训练或GPU推理。唯一Answer F1，无额外答案处理。', '',
            '四组训练前医学100题和KG267题的原ID、金标、预测、F1和NLL全部逐值一致，排除了这两组目标初始输出不同的混杂。两测试组重叠12题，不合并计数。', '',
            '| rank | 原生KG字面分组 | QA/图像数 | 起点F1 | 正确知识64步F1 | 错配64步F1 | 正确−起点及95%区间pp | 正确−错配及95%区间pp |',
            '|---|---|---:|---:|---:|---:|---|---|']
    for row in contrasts:
        if row['step'] != 64:
            continue
        a, b, c = (row[k] for k in ['base_to_correct', 'base_to_permuted', 'permuted_to_correct'])
        interval = lambda r: f"{r['delta_pp']:+.3f} [{r['image_cluster_95CI'][0]:+.3f}, {r['image_cluster_95CI'][1]:+.3f}]"
        text.append(f"| {row['rank']} | {row['stratum']} | {a['QA']}/{a['image_clusters']} | {a['before_F1']:.3f} | {a['after_F1']:.3f} | {b['after_F1']:.3f} | {interval(a)} | {interval(c)} |")
    text += ['', '**能支持：** 现有实验不能以“正确知识优于错误知识”代替“加知识优于不加”。31条同关系/固定tail字面匹配题的正确知识点估计也低于共同起点，至少不能仅凭总体覆盖不足就声称知识注入会成功。rank8的16事实文本回忆成功已另有证据，目标迁移仍需证明。', '',
             '**不能支持：** 不能据此断言知识无用、视觉绑定就是唯一瓶颈或该31题真正被知识覆盖。该分组使用作者三元组，是事后审计，不用于选事实/路由/模型；所有三组、两个rank、8/64两个预算完整记录。区间含0的地方不能宣称显著损害。', '',
             '**下一项可证伪问题：** 固定事实内容和预算，区分事实可记忆性、事实与图像实体的绑定、目标问答格式/监督；严格多域/少任务48组机制对照仍不可由本审计替代。', '']
    text += ['- '+limit for limit in result['limits']]+['']
    (REPORT/'KNOWLEDGE_LOCALITY_EVIDENCE.md').write_text('\n'.join(text))
    print(json.dumps({'CPU_only': True, 'initial_exact_comparisons': len(initial), 'stratified_contrasts': len(contrasts),
                      'same_relation_tail64': [r for r in contrasts if r['step'] == 64 and r['stratum'] == GROUPS[0]]}, ensure_ascii=False))


if __name__ == '__main__':
    main()
