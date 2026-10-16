"""Native-task isolated controls with the exact sequential stage RNG rule.

Own-domain diagnostic100 only. This removes the older dropout-RNG mismatch;
it does not replace the full21454 test or control total historical exposures.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / 'exp/onevision_diagnostics_20261016'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / 'analysis'))
from run import freeze, write
from onevision_research_worker import ResearchExperiment, atomic_torch, digest
from onevision_knowledge_intervention import Intervention, case_complete, valid_checkpoint, reset_seed


def compare_native_history(run, sequential, manifest, tests):
    """Compare identical native QA; positive NLL deltas mean worse, not accuracy."""
    source_hashes = {}

    def load(path):
        source_hashes[str(path.relative_to(ROOT))] = digest(path)
        return [json.loads(line) for line in path.read_text().splitlines()]

    rows = []
    for task in manifest['tasks']:
        tid, domain = task['task'], task['domain']
        isolated = load(run / f'evaluations/matched_native_T{tid:02d}/{domain}_image.jsonl')
        acquisition = load(sequential / f'evaluations/sequential_T{tid:02d}/{domain}_image.jsonl')
        final = load(sequential / f'evaluations/sequential_T19/{domain}_image.jsonl')
        expected = [r['id'] for r in tests[domain]]
        assert all([r['id'] for r in group] == expected for group in [isolated, acquisition, final])
        groups = [[r for r in group if r['native_task'] == tid] for group in [isolated, acquisition, final]]
        assert len(groups[0]) > 0
        means = [sum(r['nll'] for r in group)/len(group) for group in groups]
        rows.append({'task': tid, 'domain': domain, 'native_diagnostic_count': len(groups[0]),
                     'isolated_nll': means[0], 'sequential_at_acquisition_nll': means[1], 'sequential_final_nll': means[2],
                     'history_effect_at_acquisition': means[1]-means[0],
                     'change_after_acquisition': means[2]-means[1],
                     'final_gap_vs_isolated': means[2]-means[0],
                     'prior_task_count': tid-1,
                     'prior_domain_count': len({t['domain'] for t in manifest['tasks'][:tid-1]})})
    write(run / 'native_history_comparison.json', {'rows': rows, 'source_sha256': source_hashes,
          'interpretation': 'Positive deltas are worse teacher-forced NLL, not generation accuracy or ForgetRate.',
          'confounds': 'Identical own-task budget/RNG/test IDs; sequential has extra historical training. Native counts differ. Only diagnostic subset.'})
    lines = ['# 匹配随机数的原生任务对照', '',
             '同一原生测试题；正NLL差值表示更差。当前是诊断子集，不是完整测试准确率。', '',
             '| 任务 | 领域 | 题数 | 独立NLL | 顺序刚学完NLL | 顺序最终NLL | 历史影响（刚学完−独立） | 后续变化（最终−刚学完） |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f"| T{row['task']:02d} | {row['domain']} | {row['native_diagnostic_count']} | "
                     f"{row['isolated_nll']:.4f} | {row['sequential_at_acquisition_nll']:.4f} | "
                     f"{row['sequential_final_nll']:.4f} | {row['history_effect_at_acquisition']:+.4f} | "
                     f"{row['change_after_acquisition']:+.4f} |")
    lines += ['', '独立和顺序使用同一任务训练预算、seed42和原测试ID；顺序额外接受历史训练。'
              '该对照隔离历史影响，但没有消除原生任务样本量、内容与顺序混杂；不能仅按领域数作因果解释。']
    (ROOT / 'summary/onevision_mechanisms_20261016/MATCHED_NATIVE.md').write_text('\n'.join(lines)+'\n')


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    torch.set_num_threads(6)
    reset_seed()
    manifest, data, domains, tests, generation, probes, protocol = freeze()
    report = ROOT / 'summary/onevision_mechanisms_20261016'
    source_paths = [Path(__file__), HERE / 'run.py',
                    ROOT / 'analysis/onevision_research_worker.py',
                    ROOT / 'analysis/onevision_knowledge_intervention.py']
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in source_paths}
    reference = json.loads((ROOT / 'summary/onevision_diagnostics_20261016/research_current_attempt.json').read_text())
    sequential = ROOT / reference['run']
    assert json.loads((sequential / 'progress.json').read_text())['evaluated_stage'] == 19
    initial_path = sequential / 'initial_adapter.pt'
    initial_meta = json.loads(initial_path.with_suffix('.json').read_text())
    assert digest(initial_path) == initial_meta['sha256']
    spec = {'source_sha256': hashes, 'seed': 42, 'rng_rule': 'reset seed42 immediately before each task training',
            'sequential_run': reference['run'], 'initial_adapter_sha256': initial_meta['sha256'],
            'training': 'exact original frozen 1% task QA; one epoch; native batch16/scheduler',
            'test': 'same frozen own-domain100 QA and original10 generation IDs; NOT full21454 test',
            'counterfactual': 'isolated from common base vs sequential history; total past training differs'}
    pointer = report / 'matched_native_attempt.json'
    if pointer.exists():
        info = json.loads(pointer.read_text())
        assert info['spec'] == spec, 'Changed study source requires an explicit new attempt'
        run = ROOT / info['run']
    else:
        run = ROOT / 'exp/onevision_mechanisms_20261016/matched_native' / time.strftime('%Y%m%d_%H%M%S')
        run.mkdir(parents=True)
        write(pointer, {'run': str(run.relative_to(ROOT)), 'spec': spec})
        write(run / 'protocol.json', spec)
    x = Intervention(manifest, run)
    initial = x.original_initial
    ref_initial = torch.load(initial_path, map_location='cpu', weights_only=False)['adapter']
    assert all(torch.equal(initial[k][b], ref_initial[k][b]) for k in initial for b in ['A', 'B'])
    write(run / 'initial_equivalence.json', {'all_A_B_exact_equal': True, 'sequential_reference': reference['run']})
    for task in manifest['tasks']:
        tid, domain = task['task'], task['domain']
        # Manifest IDs are integer task numbers, checked against the frozen loader.
        assert isinstance(tid, int) and (tid, 'train') in data
        case = f'matched_native_T{tid:02d}'
        selected_test = {domain: tests[domain]}
        if case_complete(run, case, selected_test):
            continue
        write(report / 'status.json', {'phase': 'matched_native_controls', 'task': tid,
                                       'run': str(run.relative_to(ROOT)), 'pid': os.getpid()})
        x.configure(initial, finetune_shared=True)
        path = run / 'adapters' / (case + '.pt')
        if valid_checkpoint(path):
            checkpoint = torch.load(path, map_location='cpu', weights_only=True)
            assert checkpoint['source_sha256'] == hashes
            x.restore(checkpoint['adapter'])
        else:
            # Never reuse old dropout streams or resume a partial task's optimizer.
            reset_seed()
            adapter = x.train(case, data[tid, 'train'])
            atomic_torch(path, {'adapter': adapter, 'rank': 8, 'alpha': 32,
                                'source_sha256': hashes, 'native_task': tid,
                                'train_ids': [r['id'] for r in data[tid, 'train']]})
        if tid == 1:
            reference_path = sequential / 'durable_states/stage_01.pt'
            assert digest(reference_path) == json.loads(reference_path.with_suffix('.json').read_text())['sha256']
            first = torch.load(reference_path, map_location='cpu', weights_only=False)['adapter']
            now = x.snapshot()
            error = max(float((now[k][b]-first[k][b]).abs().max()) for k in now for b in ['A', 'B'])
            write(run / 'first_stage_reproduction.json', {'max_abs_A_B_difference': error,
                  'tolerance': 5e-6, 'passed': error < 5e-6, 'reference_sha256': digest(reference_path)})
            assert error < 5e-6, 'Matched stage1 failed reproduction; investigate before causal comparisons'
        ResearchExperiment.gradients(x, case, {domain: probes[domain]})
        x.model.eval()
        x.evaluate(case, selected_test, {domain: generation[domain]})
    compare_native_history(run, sequential, manifest, tests)
    x.monitor.snapshot('training_after', model=x.model)
    x.monitor.finish(x.model)
    write(report / 'matched_native_complete.json', {'complete': True, 'run': str(run.relative_to(ROOT)),
          'source_sha256': hashes, 'formal_full_test': False, 'native_tasks': 19})


if __name__ == '__main__':
    main()
