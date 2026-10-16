"""CPU-only paired audit of already completed base visual ablations."""
import hashlib
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_statistics(rows, variant, images):
    # Positive NLL difference means the original image is better.
    delta = [b['nll']-a['nll'] for a,b in zip(rows, variant)]
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault('|'.join(images[row['id']]), []).append(i)
    keys = sorted(groups); rng = random.Random(42); samples = []
    if len(keys) > 1:
        for _ in range(1000):
            ids = [i for key in rng.choices(keys, k=len(keys)) for i in groups[key]]
            samples.append(sum(delta[i] for i in ids)/len(ids))
        samples.sort()
    return {'qa': len(rows), 'image_clusters': len(keys), 'mean_nll_advantage_original': sum(delta)/len(delta),
            'original_lower_nll_count': sum(d > 0 for d in delta),
            'image_cluster_bootstrap95': [samples[24],samples[974]] if samples else None}


def main():
    pointer = json.loads((REPORT/'research_current_attempt.json').read_text())
    run = ROOT / pointer['old_control_run']; sources = {}
    protocol = json.loads((ROOT/'exp/onevision_diagnostics_20261016/frozen_protocol.json').read_text())
    manifest_path = ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json'
    assert sha(manifest_path) == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    images = {}; domains = {}
    for task in json.loads(manifest_path.read_text())['tasks']:
        info = task['files']['test']; path = ROOT/info['path']; assert sha(path) == info['sha256']
        for line in path.read_text().splitlines():
            row = json.loads(line); assert row['id'] not in images and row['images']
            images[row['id']] = row['images']
        domains[task['task']] = task['domain']
    groups = []; input_audit = {}
    for domain, expected in protocol['diagnostic_test_ids'].items():
        views = {}
        for view in ['image','text','wrong']:
            path = run/'evaluations/base'/f'{domain}_{view}.jsonl'
            sources[str(path.relative_to(ROOT))] = sha(path)
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert [r['id'] for r in rows] == expected
            assert all(math.isfinite(r['nll']) and r['view'] == view for r in rows)
            assert all((r['visual_tokens'] == 0) if view == 'text' else (r['visual_tokens'] > 0) for r in rows)
            if view == 'wrong':
                assert all(r['wrong_image_source_id'] in expected and
                           images[r['id']] != images[r['wrong_image_source_id']] for r in rows)
                for i, row in enumerate(rows):
                    actual_source = next(expected[(i+j)%len(expected)] for j in range(1,len(expected))
                                         if images[expected[(i+j)%len(expected)]] != images[row['id']])
                    assert row['wrong_image_source_id'] == actual_source
            views[view] = rows
        input_audit[domain] = {'ids_exact': True, 'text_zero_visual_tokens': True,
                               'image_and_wrong_have_visual_tokens': True, 'wrong_image_paths_distinct': True,
                               'deterministic_wrong_source_matches_runner_rule': True}
        for native in [None] + sorted({r['native_task'] for r in views['image']}):
            selected = {v:[r for r in rows if native is None or r['native_task'] == native]
                        for v,rows in views.items()}
            groups.append({'domain': domain, 'native_task': native,
                           'mean_nll': {v:sum(r['nll'] for r in rows)/len(rows) for v,rows in selected.items()},
                           'image_vs_text': paired_statistics(selected['image'], selected['text'], images),
                           'image_vs_wrong': paired_statistics(selected['image'], selected['wrong'], images)})
    result = {'run': str(run.relative_to(ROOT)), 'code_sha256': sha(Path(__file__)),
              'source_sha256': sources, 'input_audit': input_audit, 'groups': groups,
              'limitations': ['Untrained base only; no claim trained adapters have the same behavior.',
                             'Teacher-forced NLL on500 diagnostic QA; not accuracy or full21454 test.',
                             'Wrong image is deterministic from same-domain diagnostic pool, not necessarily contradictory content.',
                             'Wrong image audit checks distinct asset paths, not pixel-level or semantic uniqueness.',
                             'Removing an image changes prompt length; not a perfectly isolated vision-encoder intervention.',
                             'Image-cluster bootstrap1000 seed42; native subgroups can have very few samples.',
                             'Raw question/answer/image files remain unchanged; altered input views are evaluation ablations only.']}
    (REPORT/'visual_evidence.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# 基座视觉消融：原图、无图、错图', '',
             '固定同一500个诊断QA，正NLL优势表示原图更好。输入审计：无图视觉token为0；'
             '原图／错图均有视觉token；错图引用路径与原图不同且符合原确定性规则，逐题ID一致。', '',
             '| 领域 | 题数 | 原图NLL | 无图NLL | 错图NLL | 原图胜过错图题数 | 原图对错图NLL优势及95%区间 |',
             '|---|---:|---:|---:|---:|---:|---|']
    for row in groups:
        if row['native_task'] is not None: continue
        stats = row['image_vs_wrong']; ci = stats['image_cluster_bootstrap95']
        lines.append(f"| {row['domain']} | {stats['qa']} | {row['mean_nll']['image']:.3f} | "
                     f"{row['mean_nll']['text']:.3f} | {row['mean_nll']['wrong']:.3f} | "
                     f"{stats['original_lower_nll_count']} | {stats['mean_nll_advantage_original']:+.3f} "
                     + (f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '区间不可估计')+' |')
    lines += ['', '医学和遥感域没有显示稳定的正确图像NLL优势，而自然和交通表现不同。'
              '这使视觉条件利用、语言／答案行为成为需要与知识缺口一起检验的解释；'
              '不能仅据无图损失更低就断言视觉编码错误、数据无效或专业知识不需要。', '',
              '19个原生任务分组、原图对无图区间及原始文件哈希见JSON；小组样本量明确保留。', '', '限制：', '']
    lines += ['- '+item for item in result['limitations']]
    (REPORT/'VISUAL_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'domains': len(input_audit), 'native_groups': len(groups)-len(input_audit),
                      'all_three_views_verified': True, 'report': str((REPORT/'VISUAL_EVIDENCE.md').relative_to(ROOT))}))


if __name__ == '__main__':
    main()
