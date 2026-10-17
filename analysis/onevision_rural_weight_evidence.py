"""Read durable weight interventions only; no test-guided selection."""
import collections, json, math, sys
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / 'analysis'))
from onevision_knowledge_pair_audit import paired_cluster
from onevision_research_worker import digest
from geometry import inner
from run import freeze
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
CASES = ['uniform', 'class_balanced', 'urban_high_negative']


def read(p): return json.loads(p.read_text())
def records(p): return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]


def main():
    proofs = {name: REPORT / 'rural_weights' / (name + '_complete.json') for name in CASES}
    available = {name: p for name, p in proofs.items() if p.exists()}
    if not available:
        print('Pending: no complete weight-control case; no scores published.'); return
    assert 'uniform' in available
    torch.set_num_threads(4); _, data, _, _, _, _, protocol = freeze()
    groups = {'TRAIN': data[12, 'train'], 'TEST': data[12, 'test']}
    assert len(groups['TRAIN']) == 6 and len(groups['TEST']) == 100
    sources = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
    summaries, gradients, geometry, memory, cache, replication = [], [], [], [], {}, []
    uniform_spec = read(available['uniform'])['spec']
    for name, marker in available.items():
        proof = read(marker); spec = proof['spec']; run = ROOT / proof['run']
        assert proof['complete'] and proof['uniform_exact_gate'] and not proof['full21454']
        assert spec == uniform_spec and spec['cases'] == CASES and spec['updates'] == 4
        assert spec['manifest_sha256'] == protocol['source_manifest_sha256']
        assert [r['id'] for r in groups['TRAIN']] == spec['train_ids']
        assert [r['id'] for r in groups['TEST']] == spec['test_ids']
        sources[str(marker.relative_to(ROOT))] = digest(marker)
        for p, h in spec['source_sha256'].items(): assert digest(ROOT / p) == h
        for p, h in proof['artifact_sha256'].items(): assert digest(run / p) == h
        required = [*run.rglob('*.pt'), *run.rglob('*.jsonl'), run / 'training.json', run / 'memory_summary.json', run / 'protocol.json']
        assert all(str(p.relative_to(run)) in proof['artifact_sha256'] for p in required)
        assert read(run / 'protocol.json') == spec
        events = read(run / 'training.json')['records']; assert [r['update'] for r in events] == [1, 2, 3, 4]
        for event in events:
            step = event['update']; raw = torch.load(run / 'raw_gradients' / f'update_{step}.pt', map_location='cpu', weights_only=True)
            v = raw['gradient_preclip'].double(); assert torch.isfinite(v).all()
            assert v.numel() == sum(math.prod(shape) for _, shape in raw['layout'])
            norm = float(torch.linalg.vector_norm(v)); error = abs(norm - event['gradient_l2_preclip']) / max(norm, 1e-12)
            assert error < 1e-5 and event['learning_rate_used'] == 1e-4 and event['cumulative_QA_exposures'] == step * 6
            gradients.append({'case': name, 'step': step, 'L2': norm, 'relative_norm_error': error})
        initial = torch.load(run / 'start.pt', map_location='cpu', weights_only=True)['adapter']
        for step in [1, 4]:
            current = torch.load(run / f'update_{step}.pt', map_location='cpu', weights_only=True)['adapter']
            squared = sum(inner(current[k], current[k]) + inner(initial[k], initial[k]) - 2 * inner(current[k], initial[k]) for k in initial)
            assert squared >= -1e-8
            geometry.append({'case': name, 'step': step, 'actual_deltaW_displacement_Frobenius': 4 * math.sqrt(max(0, squared))})
            if name == 'uniform':
                original = torch.load(ROOT / spec['basis_case'] / f'update_{step}.pt', map_location='cpu', weights_only=True)['adapter']
                assert all(torch.equal(current[k][f], original[k][f]) for k in initial for f in ['A', 'B'])
            for group, expected in groups.items():
                path = run / 'evaluations' / f'endpoint_{step}' / (group + '_image.jsonl'); rows = records(path)
                assert [r['id'] for r in rows] == [r['id'] for r in expected]
                assert all(r['references'] == e['answers'] for r, e in zip(rows, expected)); cache[name, step, group] = rows
                for gold in [None, 'urban', 'rural']:
                    selected = [r for r, e in zip(rows, expected) if gold is None or e['answer'] == gold]
                    summaries.append({'case': name, 'step': step, 'split': group, 'gold_class': gold or 'all', 'QA': len(selected),
                                      'EM_correct': sum(r['scores']['exact_match'] == 100 for r in selected),
                                      'token_F1': float(np.mean([r['scores']['token_f1'] for r in selected])),
                                      'ROUGE_L': float(np.mean([r['scores']['rougeL'] for r in selected])),
                                      'NLL': float(np.mean([r['nll'] for r in selected])),
                                      'predictions': dict(collections.Counter(r['prediction'] for r in selected))})
                if name == 'uniform' and group == 'TEST':
                    previous = ROOT / spec['basis_case'] / 'evaluations' / f'endpoint_{step}' / 'native_image.jsonl'
                    old = records(previous); sources[str(previous.relative_to(ROOT))] = digest(previous)
                    assert [r['id'] for r in old] == [r['id'] for r in rows]
                    assert all(a['prediction'] == b['prediction'] and a['scores'] == b['scores'] for a, b in zip(old, rows))
                    replication.append({'step': step, 'QA': len(rows), 'predictions_scores_exact': True,
                                        'max_NLL_error': max(abs(a['nll'] - b['nll']) for a, b in zip(old, rows))})
        memory.append({'case': name, 'training_memory_bytes': read(run / 'memory_summary.json')})
    pairs = []
    for treatment, control in [('class_balanced', 'uniform'), ('urban_high_negative', 'uniform'), ('class_balanced', 'urban_high_negative')]:
        if treatment not in available or control not in available: continue
        for step in [1, 4]:
            for group, expected in groups.items():
                a, b = cache[treatment, step, group], cache[control, step, group]; keys = [tuple(r['images']) for r in expected]
                metrics = {metric: paired_cluster([x['scores'][metric] - y['scores'][metric] for x, y in zip(a, b)], keys)
                           for metric in ['exact_match', 'token_f1', 'rougeL']}
                metrics['NLL'] = paired_cluster([x['nll'] - y['nll'] for x, y in zip(a, b)], keys)
                pairs.append({'treatment': treatment, 'control': control, 'step': step, 'split': group, 'metrics': metrics})
    result = {'completed_cases': list(available), 'pending_cases': [n for n in CASES if n not in available], 'sources_sha256': sources,
              'summaries': summaries, 'raw_gradient_audit': gradients, 'deltaW_geometry': geometry, 'training_memory': memory,
              'paired_differences': pairs, 'uniform_replication': replication,
              'limitations': ['single native task,6TRAIN/100TEST, one start/seed; no full21454 or general domain-count conclusion',
                              'all conditions fixed beforehand; no test-based weighting, threshold or condition choice',
                              'class weighting changes the objective and optimization path, not an isolated representation edit',
                              'same coefficient multiset does not equal same per-example semantics or gradients',
                              'four updates are a diagnostic intervention, not formal1epoch CL',
                              'image-cluster10000 bootstrap uncorrected; TRAIN6 uncertainty is not across training seeds']}
    assert not torch.cuda.is_initialized()
    (REPORT / 'RURAL_WEIGHT_EVIDENCE.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 遥感损失权重对照：完成病例证据', '', '仅纳入完成证明和源码／文件SHA校验通过的病例；完整三组未完成前不下全对照结论。', '',
             '| 条件 | 更新 | split | 类别 | EM正确/总数 | F1 | ROUGE-L | NLL |', '| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |']
    for r in summaries:
        lines.append(f"| {r['case']} | {r['step']} | {r['split']} | {r['gold_class']} | {r['EM_correct']}/{r['QA']} | {r['token_F1']:.2f} | {r['ROUGE_L']:.2f} | {r['NLL']:.4f} |")
    lines += ['', '逐题预测分布、配对差异和区间、参数位移、原始梯度审计及训练内存字节见[JSON](RURAL_WEIGHT_EVIDENCE.json)。不能把改损失权重的收益解释为对所有领域／方法的唯一病因，也不据此提出新CL方法。']
    (REPORT / 'RURAL_WEIGHT_EVIDENCE.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'completed_cases': list(available), 'summary_cells': len(summaries), 'GPU_used': False}))


if __name__ == '__main__': main()
