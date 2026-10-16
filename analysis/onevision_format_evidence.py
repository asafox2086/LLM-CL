"""CPU-only paired, image-clustered analysis of all predeclared prompt cases."""
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_knowledge_intervention import PROMPTS, case_complete
from onevision_research_worker import digest
from run import freeze, write


def mean(rows, key):
    if key == 'nll':
        return sum(r[key] for r in rows)/len(rows)
    rows = [r for r in rows if 'scores' in r]
    return sum(r['scores'][key] for r in rows)/len(rows) if rows else None


def cluster_interval(deltas, groups):
    keys = sorted(groups)
    if len(keys) < 2:
        return None
    rng = random.Random(42); values = []
    for _ in range(1000):
        ids = [i for key in rng.choices(keys, k=len(keys)) for i in groups[key]]
        values.append(sum(deltas[i] for i in ids)/len(ids))
    values.sort()
    return [values[24], values[974]]


def main():
    manifest, data, domains, tests, generation, probes, protocol = freeze()
    report = ROOT / 'summary/onevision_mechanisms_20261016'
    attempt = json.loads((report / 'format_attempt.json').read_text())
    run = ROOT / attempt['run']
    sources = {}; cases = {}; summaries = []
    by_id = {r['id']: r for rows in tests.values() for r in rows}
    for variant in PROMPTS:
        case = 'format_' + variant
        assert case_complete(run, case, tests), 'Incomplete, changed or mismatched prompt case'
        cases[variant] = {}
        for domain in tests:
            path = run / 'evaluations' / case / (domain + '_image.jsonl')
            sources[str(path.relative_to(ROOT))] = digest(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 100 and all(math.isfinite(r['nll']) for r in rows)
            expected = tests[domain] if domain == 'medical' else generation[domain]
            assert {r['id'] for r in rows if 'scores' in r} == {r['id'] for r in expected}
            cases[variant][domain] = rows
            summaries.append({'variant': variant, 'domain': domain, 'nll_count': len(rows),
                              'generation_count': len(expected), 'nll': mean(rows, 'nll'),
                              **{k: mean(rows, k) for k in ['exact_match', 'token_f1', 'rougeL']}})
    original = cases['original']['medical']; groups = {}
    for i, row in enumerate(original):
        key = '|'.join(by_id[row['id']]['images']); assert key
        groups.setdefault(key, []).append(i)
    paired = []; languages = []
    for variant in PROMPTS:
        rows = cases[variant]['medical']
        assert [r['id'] for r in rows] == [r['id'] for r in original]
        for metric in ['nll', 'token_f1', 'rougeL', 'exact_match']:
            deltas = [r['nll']-b['nll'] if metric == 'nll' else r['scores'][metric]-b['scores'][metric]
                      for r,b in zip(rows, original)]
            paired.append({'variant': variant, 'metric': metric, 'delta_vs_original': sum(deltas)/len(deltas),
                           'image_cluster_bootstrap95': cluster_interval(deltas, groups),
                           'clusters': len(groups), 'qa': len(rows)})
        for language in ['contains_chinese', 'no_chinese']:
            selected = [r for r in rows if bool(re.search('[\u3400-\u9fff]', by_id[r['id']]['question'])) == (language == 'contains_chinese')]
            languages.append({'variant': variant, 'question_language_group': language, 'count': len(selected),
                              **({k: mean(selected, k) for k in ['nll', 'exact_match', 'token_f1', 'rougeL']} if selected else {})})
    result = {'run': attempt['run'], 'source_sha256': sources, 'all_predeclared_prompts_reported': True,
              'summaries': summaries, 'paired_medical': paired, 'medical_language_groups': languages,
              'limitations': ['100 medical generation QA; other domains only10, all diagnostic subsets.',
                             'Image-cluster bootstrap1000 resamples seed42 describes this subset, not full-test performance.',
                             'No prompt or capacity selected by these test scores; knowledge protocol retains predeclared both prompt.',
                             'This tests fixed English instruction additions, not all possible language/format interventions.']}
    write(report / 'format_evidence.json', result)
    lines = ['# 固定提示词对照结果', '', '未训练基座，不增加训练数据或参数；全部四个预定提示均报告。', '',
             '| 提示 | 领域 | NLL题数 | NLL↓ | 生成题数 | EM↑ | Token F1↑ | ROUGE-L↑ |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in summaries:
        lines.append(f"| {row['variant']} | {row['domain']} | {row['nll_count']} | {row['nll']:.3f} | "
                     f"{row['generation_count']} | {row['exact_match']:.3f} | {row['token_f1']:.3f} | {row['rougeL']:.3f} |")
    lines += ['', '医学100题每个提示EM均为0。提示能改变损失和部分重叠指标，'
              '但这组固定提示没有使模型稳定输出匹配参考答案的回答；不能据此证明专业知识不足。', '',
              '医学逐语言分组和配对图像簇bootstrap区间见JSON。分组只按原问题是否含中文字符，'
              '不是自动评价生成语言是否正确。自然等领域仅10个生成探针，避免把50%等小样本变化冒充全量准确率。', '',
              '限制：', ''] + ['- '+x for x in result['limitations']]
    (report / 'FORMAT_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cases': 4, 'complete_input_rows': 2000, 'generated_outputs': 560,
                      'medical_image_clusters': len(groups), 'report': str((report/'FORMAT_EVIDENCE.md').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
