"""Read-only comparison of existing checkpoint-time predictions; no GPU or training.

The legacy full-data run stopped after S3. Its predictions are not regenerated.
V2 pct01 is the meaningful reference; legacy pct01 stays explicitly invalid.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'analysis'))
from answer_f1 import VERSION, answer_f1

OUT = ROOT / 'summary/two_backbone_pct01_20261017/partial_full_vs_pct01_20261019'
RUNS = {
    'legacy_full': 'exp/five_domain_native_cl_20261016_sparse/runs/full/SeqFT_LoRA/training_20261016_205816',
    'valid_pct01': 'exp/two_backbone_native_cl_20261017_v2/runs/qwen/pct01/SeqFT_LoRA/training_20261017_161209',
    'invalid_legacy_pct01': 'exp/five_domain_native_cl_20261016_sparse/runs/pct01/SeqFT_LoRA/training_20261016_160137',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def readj(path):
    return json.loads(path.read_text())


def load_cell(run, stage, task, expected):
    p = run / f'evaluations/stage_{stage:02}/T{task:02}_test_predictions.jsonl'
    rows = [json.loads(line) for line in p.read_text().splitlines()]
    assert len(rows) == expected
    assert len({r['id'] for r in rows}) == expected
    values = [answer_f1(r['prediction'], r['references']) for r in rows]
    assert all(abs(v-r['scores']['token_f1']) < 1e-10 for v, r in zip(values, rows))
    return rows, {'path': str(p.relative_to(ROOT)), 'sha256': sha(p),
                  'qa': expected, 'answer_f1': sum(values)/expected,
                  'prompt_token_counts': [r['prompt_tokens'] for r in rows]}


def main():
    evidence = {'scope': 'existing checkpoint-time predictions, CPU rescore; NOT fresh checkpoint inference',
                'metric': VERSION, 'runs': {}, 'comparisons': [],
                'full_training_complete': False, 'pure_data_size_causal_control': False,
                'limitations': ['Full run only completed S1-S3, not 19 stages.',
                    'V2 pct01 differs in image-input/cache and scheduler/scaler implementation.',
                    'One epoch gives different optimizer-update budgets as well as unique data counts.',
                    'Legacy pct01 has zero-used-LR updates and remains invalid; not restored to the main table.',
                    'No new GPU inference or checkpoint reload was performed by this script.']}
    cached = {}
    for name, rel in RUNS.items():
        p = ROOT / rel
        config = readj(p/'config.json')
        events = [json.loads(s) for s in (p/'logs/events.jsonl').read_text().splitlines()]
        item = {'path': rel, 'config_sha256': sha(p/'config.json'),
                'train_fraction': config['train_fraction'], 'stage_updates': {}, 'cells': {},
                'source_sha256': config['execution_source_sha256']}
        for stage in range(1, 4):
            assert readj(p/f'stages/stage_{stage:02}.json')['complete']
            group = [e for e in events if e.get('event') == 'train_update' and e['stage'] == stage]
            updates = math.ceil(config['task_counts'][f'T{stage:02}']['train']/16)
            assert len(group) == updates and [e['step'] for e in group] == list(range(1, updates+1))
            assert all(math.isfinite(e['loss']) and math.isfinite(e['gradient_l2_preclip']) for e in group)
            if 'learning_rate_used' in group[0]:
                used = [e['learning_rate_used'][0] for e in group]
                origin = 'explicit learning_rate_used'
            else:
                # Frozen legacy run logs LR AFTER scheduler.step(). HF scheduler
                # starts at zero because ceil(.03*updates) >= 1 for every stage.
                src = (p/'runner_source/run.py').read_text()
                assert 'math.ceil(.03*updates)' in src and 'scheduler.get_last_lr()' in src
                used = [0.] + [e['learning_rate'][0] for e in group[:-1]]
                origin = 'reconstructed from frozen ceil-warmup source and preceding post-step LR'
            item['stage_updates'][str(stage)] = {
                'train_qa': config['task_counts'][f'T{stage:02}']['train'],
                'optimizer_calls': updates, 'positive_used_lr': sum(x > 0 for x in used),
                'zero_used_lr': sum(x == 0 for x in used), 'lr_evidence': origin,
                'events_sha256': sha(p/'logs/events.jsonl')}
            for task in [stage, stage+1]:
                rows, cell = load_cell(p, stage, task, config['task_counts'][f'T{task:02}']['test'])
                key = f'S{stage}/T{task:02}'
                item['cells'][key] = cell
                cached[name, key] = rows
        cp = readj(p/'checkpoints/stage_03.json')
        cp_path = p/cp['path']
        # Read actual weights for both compared S3 endpoints; do not load graphs.
        if name != 'invalid_legacy_pct01':
            actual = sha(cp_path)
            assert actual == cp['sha256'] and cp_path.stat().st_size == cp['bytes']
            cp['actual_file_sha256_verified'] = True
        else:
            cp['actual_file_sha256_verified'] = False
        item['stage3_checkpoint'] = cp
        evidence['runs'][name] = item
    for key in evidence['runs']['legacy_full']['cells']:
        ref = {r['id']: r for r in cached['legacy_full', key]}
        row = {'cell': key, 'qa': len(ref)}
        for name in RUNS:
            group = cached[name, key]
            assert set(ref) == {r['id'] for r in group}
            assert all(r['references'] == ref[r['id']]['references'] and
                       r['native_label'] == ref[r['id']]['native_label'] and
                       r['metadata'] == ref[r['id']]['metadata'] for r in group)
            row[name] = evidence['runs'][name]['cells'][key]['answer_f1']
            row[name+'_prompt_tokens_changed_vs_full'] = sum(
                r['prompt_tokens'] != ref[r['id']]['prompt_tokens'] for r in group)
        row['full_minus_valid_pct01'] = row['legacy_full']-row['valid_pct01']
        evidence['comparisons'].append(row)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'EVIDENCE.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2)+'\n')
    labels = {'S1/T01': '自然问答，学完自然后', 'S1/T02': '医学 Modality，尚未训练',
              'S2/T02': '医学 Modality，刚学完', 'S2/T03': '医学 Position，尚未训练',
              'S3/T03': '医学 Position，刚学完', 'S3/T04': '医学 Organ，尚未训练'}
    lines = ['# 已有100%与1%阶段checkpoint的表现', '',
        '骨干均为 Qwen2-VL-2B，方法均为顺序 LoRA。100%只完成前三个训练阶段；本报告重算这些checkpoint保存时已有的完整测试预测，未重新加载权重生成答案。', '',
        '## 可用结果（原始 Answer F1，0–100）', '',
        '| 评测时点与任务 | 测试题数 | 100%训练 | 有效1%训练（V2） | 差值 |',
        '|---|---:|---:|---:|---:|']
    for c in evidence['comparisons']:
        lines.append(f"| {labels[c['cell']]} | {c['qa']} | {c['legacy_full']:.3f} | {c['valid_pct01']:.3f} | {c['full_minus_valid_pct01']:+.3f} |")
    lines += ['', '100%前三阶段训练量分别为34,602／1,070／1,695题；1%为347／11／17题。比较格逐题ID、参考答案、原生题型和元数据相同，评分由原始预测重新计算；两端S3权重文件实际SHA已核对。', '',
        '## 结论与边界', '',
        '自然问答刚学后100%为85.817，1%为86.050，几乎持平；医学Modality与Position刚学后，100%分别高43.458与33.832分。因此，已有模型的差距集中在这两个医学任务，不能笼统说所有任务都因1%而训练不好。', '',
        '医学任务学前也已有差距：Modality为63.816对34.222，Position为38.579对23.416。各自训练带来的涨分分别为19.956对6.092、26.633对7.964；最终差距同时包含此前训练的迁移差异与本任务习得差异，并非全部由当前任务新增。', '',
        '这些是同一训练阶段、同一测试题目的已有表现，可以回答“多数据的已训练模型表现如何”。它们不是完整19阶段结果，也不能作为只有数据量不同的严格因果实验：有效1%使用新版共同图像输入与训练框架，100%来自旧框架；一轮训练的更新次数也不同。逐题prompt token数也确认两套输入有实际差异，不能声称只换了训练集比例。', '',
        '不得把这里的差值全归因于数据量，不以这些TEST结果挑超参或停止点，不填入完整主表。100%阶段4的空答案失败发生在S3之后，不影响上述已封存预测。', '',
        '## 旧1%结果：仅保留审计，不作为有效对照', '',
        '| 阶段 | 100% 正LR/总更新 | 有效1% 正LR/总更新 | 旧1% 正LR/总更新 |',
        '|---|---:|---:|---:|']
    for s in range(1, 4):
        vals = [evidence['runs'][n]['stage_updates'][str(s)] for n in RUNS]
        lines.append('| S%d | %s |' % (s, ' | '.join(f"{v['positive_used_lr']}/{v['optimizer_calls']}" for v in vals)))
    lines += ['', '旧日志的learning_rate是scheduler.step之后的值；实际使用值由冻结源码和前一条日志还原，不能把日志中的非零值直接当作该步实际学习率。旧1% S2单步实际LR为0；S3两步中仅一步非零。旧结果不恢复为有效基线。', '',
        '证据： [EVIDENCE.json](EVIDENCE.json)。脚本：`analysis/qwen_partial_full_vs_pct01_20261019.py`。所有原训练、预测、checkpoint均只读；未抢占GPU或中断基线。', '']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(evidence['comparisons'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
