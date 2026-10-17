"""Read-only snapshot of completed domains, never of the live traffic suffix."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
FULL_RUN = ROOT / 'exp/onevision_capacity_20261016/cap1_full_confirm/20261017_091853'
DOMAINS = ['natural', 'medical', 'geography', 'finance']


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def score(prediction, refs):
    a = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', prediction.lower())
    values = []
    for ref in refs:
        b = re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', ref.lower())
        exact = 100. * (a == b)
        if not a or not b:
            values.append(dict(exact_match=exact, token_f1=exact, rougeL=exact))
            continue
        common = sum((Counter(a) & Counter(b)).values())
        masks = {}
        for i, word in enumerate(b):
            masks[word] = masks.get(word, 0) | (1 << i)
        state = 0
        for word in a:
            union = masks.get(word, 0) | state
            state = union & ~(union - ((state << 1) | 1))
        values.append(dict(exact_match=exact, token_f1=100 * (2 * common / (len(a) + len(b))),
                           rougeL=200 * state.bit_count() / (len(a) + len(b))))
    return {name: max(x[name] for x in values) for name in values[0]}


def main():
    sources = {}

    def read(p):
        sources[str(p.relative_to(ROOT))] = sha(p)
        return json.loads(p.read_text())

    proof_path = REPORT / 'capacity/cap1_complete.json'
    proof = read(proof_path)
    assert proof['complete'] and proof['cap'] == 1
    original = ROOT / proof['run']
    assert sha(original / 'latest.pt') == proof['final_state_sha256']
    renderer = read(FULL_RUN / 'renderer_protocol.json')
    gate = read(FULL_RUN / 'renderer_gpu_gate.json')
    assert gate['passed']
    manifest = read(ROOT / 'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')
    results = {}
    for domain in DOMAINS:
        expected = []
        for task in manifest['tasks']:
            if task['domain'] == domain:
                f = task['files']['test']
                p = ROOT / f['path']
                assert sha(p) == f['sha256']
                sources[str(p.relative_to(ROOT))] = sha(p)
                expected.extend(dict(json.loads(line), _audit_task=task['task']) for line in p.read_text().splitlines())
        path = FULL_RUN / 'evaluations/final_full_input_router' / (domain + '_image.jsonl')
        before = sha(path)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in rows] == [r['id'] for r in expected]
        for row, source in zip(rows, expected):
            assert row['references'] == source['answers']
            assert row['native_task'] == source['_audit_task']
            assert row['scores'] == score(row['prediction'], row['references'])
        old_path = original / 'evaluations/final_input_router' / (domain + '_image.jsonl')
        assert sha(old_path) == proof['artifact_sha256'][str(old_path.relative_to(original))]
        old_rows = [json.loads(line) for line in old_path.read_text().splitlines()]
        old_map = {r['id']: r for r in old_rows}
        assert len(old_map) == 100
        current = {r['id']: r for r in rows}
        paired = [(a, current[a['id']]) for a in old_rows]
        generated = [(a, b) for a, b in paired if 'prediction' in a]

        def stats(part):
            return {'QA': len(part), 'EM_correct': sum(r['scores']['exact_match'] == 100 for r in part),
                    **{k: sum(r['scores'][k] for r in part) / len(part)
                       for k in ['exact_match', 'token_f1', 'rougeL']}}

        results[domain] = {'full_domain': stats(rows),
            'previously_inspected100': stats([r for r in rows if r['id'] in old_map]),
            'outside_previously_inspected100': stats([r for r in rows if r['id'] not in old_map]),
            'NLL_overlap_QA': len(paired),
            'max_abs_overlap_NLL_error': max(abs(a['nll'] - b['nll']) for a, b in paired),
            'old_generated_overlap_QA': len(generated),
            'generated_prediction_and_score_exact': sum(a['prediction'] == b['prediction'] and a['scores'] == b['scores'] for a, b in generated)}
        results[domain]['native_tasks'] = {}
        for tid in sorted({r['native_task'] for r in rows}):
            part = [r for r in rows if r['native_task'] == tid]
            task = manifest['tasks'][tid - 1]
            results[domain]['native_tasks'][str(tid)] = {
                **stats(part), 'train_QA': task['files']['train']['count'],
                'native_label': task.get('native_label'),
                'normalized_output_modes': Counter(' '.join(re.findall(r'[a-z0-9]+|[\u3400-\u9fff]', r['prediction'].lower())) for r in part).most_common(5)}
        assert before == sha(path)
        sources[str(path.relative_to(ROOT))] = before
        sources[str(old_path.relative_to(ROOT))] = sha(old_path)
    result = {'completed_domains_only': results, 'source_sha256': sources,
              'final_weights_sha256': proof['final_state_sha256'],
              'all21454_complete': False, 'traffic_not_read': True,
              'scope': 'existing cap1 diagnostic final weights, not shared-framework formal baseline',
              'no_AP_FWT_BWT_or_method_superiority_claim': True,
              'renderer_repair': renderer['repair']}
    (REPORT / 'FULL_COMPLETED_DOMAINS_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# cap1完整确认：已结束四域的只读核验', '',
        '交通域仍运行，本文件只读取已结束的四域，不称21454题全部完成。它核验既有诊断权重，不进入新双骨干正式主表，也不是新方法／自动最优容量结果。', '',
        '| 域 | 完整QA | EM正确数 | EM | F1 | ROUGE-L | 旧诊断生成重合完全一致 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for domain, value in results.items():
        s = value['full_domain']
        lines.append(f"| {domain} | {s['QA']} | {s['EM_correct']} | {s['exact_match']:.3f} | {s['token_f1']:.3f} | {s['rougeL']:.3f} | {value['generated_prediction_and_score_exact']}/{value['old_generated_overlap_QA']} |")
    lines += ['', '每域此前看过100题与其余QA分别报告在JSON，不将其余子集重新挑成优化目标。逐题顺序、gold、原主评分均重新核验，已完成文件前后SHA不变。没有基座全量和完整阶段对照，不能从这些终点分数计算完整FWT/BWT或判断专业知识是唯一原因。', '',
              '[逐域分组、重合误差和SHA](FULL_COMPLETED_DOMAINS_AUDIT.json)。']
    (REPORT / 'FULL_COMPLETED_DOMAINS_AUDIT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(results, ensure_ascii=False))


if __name__ == '__main__':
    main()
