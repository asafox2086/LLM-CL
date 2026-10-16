"""CPU-only retrospective reuse audit, selecting strictly on frozen TRAIN probes.

Only past-domain independently trained adapters are eligible at each domain's
arrival. This is not a trained online router, a Kmax experiment, or a CL ranking.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/onevision_diagnostics_20261016'
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choose_past(domain_index, losses):
    candidates = ['base'] + ['isolated_' + d for d in DOMAINS[:domain_index]]
    domain = DOMAINS[domain_index]
    ranked = sorted(candidates, key=lambda name: (losses[name][domain], candidates.index(name)))
    margin = losses[ranked[1]][domain]-losses[ranked[0]][domain] if len(ranked) > 1 else None
    return ranked[0], candidates, margin


def main():
    pointer = json.loads((REPORT / 'research_current_attempt.json').read_text())
    run = ROOT / pointer['old_control_run']
    protocol = json.loads((ROOT / 'exp/onevision_diagnostics_20261016/frozen_protocol.json').read_text())
    losses = {}; train_hashes = {}
    for name in ['base'] + ['isolated_' + d for d in DOMAINS]:
        path = run / 'gradients' / (name + '.json')
        info = json.loads(path.read_text())
        assert info['train_only'] and info['probe_qa_per_domain'] == 2
        losses[name] = info['train_probe_nll']
        train_hashes[str(path.relative_to(ROOT))] = sha(path)
        if name != 'base':
            weight = run / 'adapters' / (name + '.pt')
            assert sha(weight) == pointer['completed_control_sha256'][str(weight.relative_to(ROOT))]
    # Freeze choices BEFORE opening any diagnostic test result or score summary.
    decisions = []
    for i, domain in enumerate(DOMAINS):
        selected, candidates, margin = choose_past(i, losses)
        decisions.append({'domain': domain, 'selected': selected, 'eligible': candidates,
                          'train_probe_ids': protocol['train_probe_ids'][domain][:2],
                          'candidate_train_nll': {name: losses[name][domain] for name in candidates},
                          'best_second_train_nll_margin': margin,
                          'current_or_future_adapter_eligible': False})
    frozen = {'selection': 'minimum mean NLL on original two frozen TRAIN probes/domain; tie follows candidate order',
              'run': str(run.relative_to(ROOT)), 'decisions': decisions, 'train_source_sha256': train_hashes,
              'code_sha256': sha(Path(__file__)), 'diagnostic_test_scores_used_for_selection': False}
    decision_file = REPORT / 'past_adapter_train_choices.json'
    if decision_file.exists():
        assert json.loads(decision_file.read_text()) == frozen, 'Frozen route audit changed; use a new version'
    else:
        decision_file.write_text(json.dumps(frozen, indent=2)+'\n')
        decision_file.chmod(0o444)
    rows = []; test_hashes = {}

    def read_scores(name, domain):
        path = run / 'evaluations' / name / (domain + '_image.jsonl')
        data = [json.loads(line) for line in path.read_text().splitlines()]
        assert [r['id'] for r in data] == protocol['diagnostic_test_ids'][domain]
        assert len(data) == 100
        test_hashes[str(path.relative_to(ROOT))] = sha(path)
        generated = [r for r in data if 'scores' in r]
        assert {r['id'] for r in generated} == set(protocol['generation_ids'][domain])
        return {'nll': sum(r['nll'] for r in data)/len(data), 'nll_count': len(data),
                'generated_count': len(generated),
                **{k: sum(r['scores'][k] for r in generated)/len(generated)
                   for k in ['exact_match', 'token_f1', 'rougeL']}}

    for decision in decisions:
        domain = decision['domain']; chosen = decision['selected']
        base = read_scores('base', domain); selected = read_scores(chosen, domain)
        own = read_scores('isolated_'+domain, domain)
        rows.append({'domain': domain, 'selected_previous_adapter': chosen,
                     'selection_train_probe_count': 2,
                     'base': base, 'selected': selected, 'own_independent': own,
                     'selected_minus_base_nll': selected['nll']-base['nll'],
                     'best_second_train_nll_margin': decision['best_second_train_nll_margin'],
                     'own_domain_train_count': len(protocol['train_ids'][domain])})
    result = {'frozen_choices_sha256': sha(decision_file), 'rows': rows, 'test_source_sha256': test_hashes,
              'limitations': [
                  'Offline audit of completed independent adapters, not actual online training or learned routing.',
                  'Retrospective diagnostic: these test subsets were already inspected during research; not untouched confirmation.',
                  'Domain identity known; only two in-training probes/domain, no independent validation set.',
                  'Earlier adapters have unequal training budgets; no claim generic knowledge replaces professional knowledge.',
                  'Old isolated control dropout streams differ; does not fix the failed matched native reproduction gate.',
                  'Per-domain100 NLL and10 generated outputs; not full21454 test or final CL performance.',
                  'No modules are added by this audit. Kmax1/2/3/5 automatic reuse/growth remains untested.'
              ]}
    target = REPORT / 'past_adapter_reuse_audit.json'
    target.write_text(json.dumps(result, indent=2)+'\n')
    lines = ['# 只用训练探针选择旧LoRA：离线审计', '',
             '每域到来时只允许原基座和此前领域的独立LoRA；先冻结TRAIN探针决策，再读取测试输出。'
             '本报告不是在线算法或Kmax容量实验。', '',
             '| 当前领域 | 只从此前选择 | 原基座NLL | 所选旧适配器NLL | 本域独立LoRA NLL | 训练探针最好／次好差 |',
             '|---|---|---:|---:|---:|---:|']
    for row in rows:
        margin = row['best_second_train_nll_margin']
        lines.append(f"| {row['domain']} | {row['selected_previous_adapter']} | {row['base']['nll']:.3f} | "
                     f"{row['selected']['nll']:.3f} | {row['own_independent']['nll']:.3f} | "
                     + ('—' if margin is None else f'{margin:.4f}')+' |')
    lines += ['', '医学域未用自己的LoRA时，TRAIN探针选择自然域旧LoRA，NLL明显低于原基座。'
              '这支持继续检验复用价值，但自然域训练347QA、医学99QA，且格式／语言适应可能贡献增益，'
              '不能因此断定专业知识已经足够。', '',
              '交通域候选TRAIN损失差很小；仅两个探针的选择可靠性不足，必须用预定的更充分TRAIN探针'
              '和真实在线干预再核验，不能按这批测试分数更改选择。', '', '限制：', '']
    lines += ['- '+text for text in result['limitations']]
    (REPORT / 'PAST_ADAPTER_REUSE_AUDIT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'rows': len(rows), 'selected': [r['selected'] for r in decisions],
                      'choices_frozen_before_test_read': True, 'report': str(target.relative_to(ROOT))}))


if __name__ == '__main__':
    main()
