"""Read-only CPU evidence from completed sequential stages, never partial scores.

Gradient/displacement products are in the actual A/B parameterization. They are
local first-order diagnostics, not proof of a conflict mechanism or deltaW cosine.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'


def main():
    torch.set_num_threads(2)
    pointer = json.loads((REPORT / 'research_current_attempt.json').read_text())
    run = ROOT / pointer['run']
    progress = json.loads((run / 'progress.json').read_text())
    last = progress['evaluated_stage']
    assert last >= 2
    sources = {}

    def read(path, tensor=False):
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return (torch.load(path, map_location='cpu', weights_only=False)
                if tensor else json.loads(path.read_text()))

    summaries = {s: read(run / f'evaluations/sequential_T{s:02d}/summary.json')
                 for s in range(1, last + 1)}
    rows = []
    pairs = []
    for s in range(2, last + 1):
        means_without_first_layer = {}
        previous = read(run / f'durable_states/stage_{s-1:02d}.pt', True)['adapter']
        current = read(run / f'durable_states/stage_{s:02d}.pt', True)['adapter']
        for domain in ['natural', 'medical', 'geography', 'finance', 'traffic']:
            before = read(run / f'raw_probe_gradients/sequential_T{s-1:02d}/{domain}.pt', True)
            after = read(run / f'raw_probe_gradients/sequential_T{s:02d}/{domain}.pt', True)
            assert before['ids'] == after['ids']
            parts = []
            for name, layout in before['parameter_layout'].items():
                module, branch, suffix = name.rsplit('.', 2)
                assert suffix == 'weight' and branch in ['A', 'B']
                diff = current[module][branch].double() - previous[module][branch].double()
                assert list(diff.shape) == layout['shape']
                assert sum(p.numel() for p in parts) == layout['offset']
                parts.append(diff.flatten())
            delta = torch.cat(parts)
            gradients = before['per_example_preclip_gradient'].double()
            predicted = gradients @ delta
            actual = torch.tensor(after['nll'], dtype=torch.float64) - torch.tensor(before['nll'], dtype=torch.float64)
            next_gradients = after['per_example_preclip_gradient'].double()
            first_layer_mask = torch.zeros(next_gradients.shape[1], dtype=torch.bool)
            for name, layout in after['parameter_layout'].items():
                if '.layers.0.' in name:
                    first_layer_mask[layout['offset']:layout['offset']+layout['count']] = True
            norms = next_gradients.norm(dim=1)
            first_layer_fraction = next_gradients[:, first_layer_mask].square().sum(1) / norms.square().clamp_min(1e-30)
            means_without_first_layer[domain] = next_gradients[:, ~first_layer_mask].mean(0)
            first = summaries[1][domain + ':image']['mean_nll']
            now = summaries[s][domain + ':image']['mean_nll']
            old = summaries[s-1][domain + ':image']['mean_nll']
            rows.append({'stage': s, 'domain': domain,
                         'diagnostic_test_n': summaries[s][domain + ':image']['count'],
                         'diagnostic_test_nll': now, 'change_from_previous': now-old,
                         'change_from_stage1': now-first,
                         'train_probe_ids': before['ids'],
                         'train_probe_predicted_delta_nll': predicted.tolist(),
                         'train_probe_actual_delta_nll': actual.tolist(),
                         'mean_first_order_delta_nll': predicted.mean().item(),
                         'mean_actual_probe_delta_nll': actual.mean().item(),
                         'parameter_displacement_norm': delta.norm().item(),
                         'next_probe_gradient_l2': norms.tolist(),
                         'next_probe_layer0_energy_fraction': first_layer_fraction.tolist(),
                         'sign_agreement_count': int(((predicted > 0) == (actual > 0)).sum()),
                         'probe_count': len(before['ids'])})
        grad = read(run / f'gradients/sequential_T{s:02d}.json')
        matrix = grad['same_model_parameterization_gradient_cosine']
        domains = list(matrix)
        entries = [matrix[a][b] for i, a in enumerate(domains) for b in domains[i+1:]]
        excluded = []
        for i, a in enumerate(domains):
            for b in domains[i+1:]:
                x, y = means_without_first_layer[a], means_without_first_layer[b]
                excluded.append({'domain_a': a, 'domain_b': b,
                                 'cosine': float(torch.dot(x, y) / (x.norm()*y.norm()).clamp_min(1e-30))})
        pairs.append({'stage': s, 'negative_domain_pairs': sum(v < 0 for v in entries),
                      'total_pairs': len(entries), 'minimum_cosine': min(entries),
                      'maximum_cosine': max(entries),
                      'excluding_layer0_pairs': excluded,
                      'excluding_layer0_min_cosine': min(v['cosine'] for v in excluded),
                      'excluding_layer0_max_cosine': max(v['cosine'] for v in excluded)})
    output = {'run': pointer['run'], 'evaluated_through_stage': last,
              'trained_stage_at_read': progress['trained_stage'],
              'source_sha256': sources, 'rows': rows, 'gradient_pair_summaries': pairs,
              'limitations': [
                  '500 fixed diagnostic QA, 100/domain, NOT full21454 test.',
                  'NLL is teacher-forced loss; it is not generation accuracy.',
                  'Only four fixed TRAIN probes/domain; no population gradient inference.',
                  'A/B first-order approximation can fail for finite steps and is not causal proof.',
                  'Native tasks have unequal training counts; isolated old controls use different dropout RNG.',
                  'Stages2-11 are all medicine, so this prefix cannot establish multi-domain harm.'
              ]}
    target = REPORT / 'sequential_evidence.json'
    temporary = target.with_suffix('.partial')
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    lines = ['# OneVision顺序阶段诊断（未完成正式全量测试）', '',
             f'已评测至T{last:02d}；每域100题NLL。正变化表示损失上升。', '',
             '| 阶段 | 领域 | NLL | 相对前阶段 | 相对T01 | 训练探针一阶预测变化 | 探针实际变化 |',
             '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| T{row['stage']:02d} | {row['domain']} | {row['diagnostic_test_nll']:.4f} | "
                     f"{row['change_from_previous']:+.4f} | {row['change_from_stage1']:+.4f} | "
                     f"{row['mean_first_order_delta_nll']:+.4f} | {row['mean_actual_probe_delta_nll']:+.4f} |")
    lines += ['', '限制：', ''] + ['- ' + item for item in output['limitations']]
    lines += ['', '## 梯度方向是否被单层支配', '',
              '| 阶段 | 全参数最小余弦 | 排除第0层后的最小余弦 | 全参数最大余弦 | 排除第0层后的最大余弦 |',
              '|---|---:|---:|---:|---:|']
    for pair in pairs:
        lines.append(f"| T{pair['stage']:02d} | {pair['minimum_cosine']:.4f} | "
                     f"{pair['excluding_layer0_min_cosine']:.4f} | {pair['maximum_cosine']:.4f} | "
                     f"{pair['excluding_layer0_max_cosine']:.4f} |")
    lines += ['', '逐探针范数和第0层梯度能量比例见JSON。排除某层只是诊断分解，'
              '不是已验证训练干预，也不能单凭该分解断言是数值错误。']
    (REPORT / 'SEQUENTIAL_EVIDENCE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'evaluated_stage': last, 'domain_stage_rows': len(rows),
                      'evidence': str(target.relative_to(ROOT)), 'gradient_pairs': pairs}))


if __name__ == '__main__':
    main()
