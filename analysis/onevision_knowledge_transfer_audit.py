"""Post-hoc literal transfer audit; no model, label edits or replacement metric.

Anatomy surface-form flags are review candidates, not semantic correct answers.
"""
import collections, json, re, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_knowledge_mechanism_audit import normalize
from onevision_research_worker import digest
from run import freeze, score
from runtime_utils import tokens
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
CASES = ['rank1_correct', 'rank1_permuted', 'rank8_correct', 'rank8_permuted']


def without_prefix(text):
    text = text.strip()
    return text[len('the answer is '):].strip() if text.lower().startswith('the answer is ') else text


def anatomy_surface_flag(prediction, references, subjects):
    """Only one English subject token, optional article/simple appended s.

    Never handles qualifiers, lists, translation or clinical equivalence.
    """
    pred = tokens(without_prefix(prediction))
    if pred and pred[0] in ['the', 'a', 'an']: pred = pred[1:]
    for ref in references:
        gold = tokens(ref)
        if len(gold) == len(pred) == 1 and gold[0] in subjects:
            if pred[0] in [gold[0], gold[0] + 's']: return True
    return False


def checks():
    subjects = {'heart', 'lung', 'kidney'}
    assert anatomy_surface_flag('The Heart', ['Heart'], subjects)
    assert anatomy_surface_flag('lungs', ['Lung'], subjects)
    assert anatomy_surface_flag('The answer is the kidneys', ['Kidney'], subjects)
    for pred, ref in [('left lung', 'Lung'), ('lungs', 'Heart'), ('lung and liver', 'Lung'), ('肺', 'Lung')]:
        assert not anatomy_surface_flag(pred, [ref], subjects)
    assert score('lungs', ['Lung'])['exact_match'] == 0
    assert score('The Heart', ['Heart'])['exact_match'] == 0
    return {'literal_flags_only': True, 'qualified_wrong_organ_list_translation_rejected': True, 'original_EM_unchanged': True}


def main():
    test = checks(); _, data, _, _, _, _, _ = freeze(); original = data[11, 'test']; assert len(original) == 267
    sources = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    def read(p): sources[str(p.relative_to(ROOT))] = digest(p); return json.loads(p.read_text())
    pool = read(ROOT / 'data/OneVisionMechanisms/20261016/knowledge16.json')
    assert {normalize(f['value']) for f in pool['correct']} == {normalize(f['value']) for f in pool['permuted']}
    raw = {str(q['qid']): q for q in read(ROOT / 'data/SLAKE/raw/test.json')}
    subjects = {tokens(f['subject'])[0] for f in pool['correct'] if re.fullmatch('[A-Za-z]+', f['subject'])}
    values = {normalize(f['value']) for f in pool['correct']}; annotations = {}
    for row in original:
        q = raw[str(row['source_id'])]
        assert q['question'].strip() == row['question'].strip() and q['answer'].strip() == row['answer'].strip()
        relation, tail = map(normalize, q['triple'][1:])
        matches = [f for f in pool['correct'] if normalize(f['relation']) == relation and normalize(f['value']) == tail]
        if matches:
            group = 'same_relation_tail_gold_subject' if any(normalize(q['answer']) == normalize(f['subject']) for f in matches) else 'same_relation_tail_other_gold'
        else: group = 'relation_only' if any(normalize(f['relation']) == relation for f in pool['correct']) else 'relation_absent'
        annotations[row['id']] = {'literal_group': group, 'q_lang': q['q_lang'], 'answer_type': q['answer_type'],
                                  'author_triple': q['triple'], 'matching_fact_ids': [f['id'] for f in matches]}
    summary, grouped, examples = [], [], []
    for name in CASES:
        proof = read(REPORT / 'knowledge_fit' / (name + '_complete.json')); assert proof['complete']
        run = ROOT / proof['run']; assert proof['spec']['pool_sha256'] == sources['data/OneVisionMechanisms/20261016/knowledge16.json']
        for p, h in proof['spec']['source_sha256'].items(): assert digest(ROOT / p) == h
        for step in [0, 8, 64]:
            p = run / 'evaluations' / f'clinical_update_{step:02d}' / 'medical_KG_image.jsonl'
            assert digest(p) == proof['artifact_sha256'][str(p.relative_to(run))]; sources[str(p.relative_to(ROOT))] = digest(p)
            rows = [json.loads(s) for s in p.read_text().splitlines()]
            assert [x['id'] for x in rows] == [x['id'] for x in original]
            for row, source in zip(rows, original):
                assert row['references'] == source['answers']
                assert score(row['prediction'], row['references']) == row['scores']
            def summarize(part):
                flags = [x for x in part if x['scores']['exact_match'] != 100 and score(without_prefix(x['prediction']), x['references'])['exact_match'] != 100 and anatomy_surface_flag(x['prediction'], x['references'], subjects)]
                emitted = [x for x in part if normalize(x['prediction']) in values]
                return {'QA': len(part), 'primary_EM_correct': sum(x['scores']['exact_match'] == 100 for x in part),
                        'fixed_prefix_aux_correct': sum(score(without_prefix(x['prediction']), x['references'])['exact_match'] == 100 for x in part),
                        'additional_anatomy_surface_review_flags': len(flags), 'surface_review_ids': [x['id'] for x in flags],
                        'literal_trained_value_outputs': len(emitted), 'value_output_ids': [x['id'] for x in emitted],
                        'value_outputs_primary_wrong': sum(x['scores']['exact_match'] != 100 for x in emitted),
                        'Chinese_question_Latin_only_answer': sum(annotations[x['id']]['q_lang'] == 'zh' and bool(re.search('[A-Za-z]', x['prediction'])) and not re.search('[\u3400-\u9fff]', x['prediction']) for x in part),
                        'normalized_output_modes': collections.Counter(normalize(x['prediction']) for x in part).most_common(5),
                        'mean_NLL': float(np.mean([x['nll'] for x in part]))}
            summary.append({'case': name, 'step': step, **summarize(rows)})
            for dim in ['literal_group', 'q_lang', 'answer_type']:
                for label in sorted({annotations[x['id']][dim] for x in rows}):
                    grouped.append({'case': name, 'step': step, 'dimension': dim, 'label': label,
                                    **summarize([x for x in rows if annotations[x['id']][dim] == label])})
            if name == 'rank8_correct' and step == 64:
                for row, source in zip(rows, original):
                    if annotations[row['id']]['literal_group'] == 'same_relation_tail_gold_subject':
                        examples.append({'id': row['id'], 'question': source['question'], 'references': row['references'],
                                         'prediction': row['prediction'], 'primary_EM': row['scores']['exact_match'],
                                         'anatomy_surface_review_flag': anatomy_surface_flag(row['prediction'], row['references'], subjects)})
    assert len(examples) == 11 and len(summary) == 12
    result = {'sources_sha256': sources, 'CPU_rule_checks': test, 'TRAIN_pool_subject_vocabulary': sorted(subjects),
              'cohort_sizes': dict(collections.Counter(a['literal_group'] for a in annotations.values())),
              'correct_permuted_value_vocabulary_same': True, 'value_frequency_counts': {k: dict(collections.Counter(normalize(f['value']) for f in pool[k])) for k in ['correct', 'permuted']},
              'summaries': summary, 'groups': grouped, 'native_annotations': annotations, 'all11_reverse_subject_examples': examples,
              'scope': 'post-hoc descriptive audit of267QA; no training, threshold/routing/fact selection or new scoring',
              'limitations': ['same relation/tail plus gold subject is a literal reporting cohort, not guaranteed visual answerability',
                              '11-case cohort is post-hoc and tiny; cannot generalize its rates',
                              'article/simple-s flags are lexical review candidates, not clinical semantic correctness or a new accuracy',
                              'trained value output overlap does not prove causal memorization or exclusive origin in the fact pool',
                              'fact values have zero exact gold overlap in this cohort; all-wrong value emissions partly follow dataset composition',
                              'correct/permuted value vocabulary equal does not mean frequency or fit difficulty equal',
                              'concentration and language flags suggest mechanisms but do not replace controlled GPU interventions']}
    (REPORT / 'KNOWLEDGE_TRANSFER_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 知识写入后的迁移：输出偏移与评分表面形式', '',
             '只读已完成病例；逐题主评分重新复算一致，原数据／评分／参数不改。以下均为事后诊断，不是新临床准确率。', '',
             '| 条件 | 步数 | 主EM正确/267 | 已训练value词表中的输出（均为原EM错） | 额外冠词／单复数待核查 | 中文题纯拉丁字母回答 |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for x in summary:
        lines.append(f"| {x['case']} | {x['step']} | {x['primary_EM_correct']} | {x['literal_trained_value_outputs']} | {x['additional_anatomy_surface_review_flags']} | {x['Chinese_question_Latin_only_answer']} |")
    lines += ['', '## 不能把严格0分直接解释为完全没有迁移', '',
              '作者三元组同relation／tail且gold等于同一匹配事实subject的11题中，rank8正确知识64步原EM为0。逐题列出全部11例，避免只挑有利例子；它们与训练时subject→relation→value的问法不同，并依赖图像。', '',
              '同一11题的负对照也必须同时报告：所有条件／时点原EM均为0/11。下面仅列额外词形待核查数，不能加到主正确数。', '',
              '| 条件 | 0步 | 8步 | 64步 |', '| --- | ---: | ---: | ---: |']
    for name in CASES:
        cohort = [x for x in grouped if x['case'] == name and x['dimension'] == 'literal_group' and x['label'] == 'same_relation_tail_gold_subject']
        assert len(cohort) == 3 and all(x['QA'] == 11 and x['primary_EM_correct'] == 0 for x in cohort)
        lines.append('| ' + name + ' | ' + ' | '.join(str(x['additional_anatomy_surface_review_flags']) for x in cohort) + ' |')
    lines += ['', 'rank8错配组64步也有10条词形待核查，正确组为9条。因此这组现象不能证明正确事实带来了特异性的关系迁移；以下11例仅展示正确组的具体输出，不作为正负对照结论。', '',
              '| ID | 参考答案 | 原生成 | 仅冠词／简单s词形差待核查 |', '| --- | --- | --- | --- |']
    for x in examples: lines.append(f"| {x['id']} | {' / '.join(x['references'])} | {x['prediction'].strip()} | {'是' if x['anatomy_surface_review_flag'] else '否'} |")
    lines += ['', '词形规则只识别TRAIN事实subject词表中的单英文器官名，允许一个前导冠词及简单附加s；不合并左右器官、列表、翻译或错误器官。此列不是人工医学判定，不能加到主正确数中，也未用于调参。', '',
              '## 当前可以与不可以得出的结论', '',
              '- 记住正向事实后，输出仍会集中到少数器官／系统词；KG问法、关系、语言与输出形式需要分开检查，不能只说知识不足或全部视觉失效。',
              '- 原主EM很低时，同时存在实际错误与表面形式差异；不能从这11题原EM0推断模型完全不会对应关系，也不能把词形待核查当作泛化已解决。',
              '- 词表输出频率变化是现象证据；没有训练信息隔离的干预前，不把它当成唯一原因。正确／错配完整结果和各原生分组见JSON。', '',
              '全部组别、输出模式、逐题ID与SHA见[JSON](KNOWLEDGE_TRANSFER_AUDIT.json)。']
    (REPORT / 'KNOWLEDGE_TRANSFER_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'cases': 4, 'native_KG_QA': 267, 'reverse_subject_examples': len(examples), 'literal_group_sizes': result['cohort_sizes'], 'GPU_used': False}))


if __name__ == '__main__': main()
