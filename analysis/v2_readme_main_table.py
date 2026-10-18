"""Render the outer README main table from sealed formal V2 proofs only.

CPU-only documentation command, never starts inference or training. Missing
results remain empty; research AnyRes/155 diagnostics cannot enter this table.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = ROOT/'summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json'
BEGIN = '<!-- FORMAL_MAIN_TABLE_BEGIN -->'
END = '<!-- FORMAL_MAIN_TABLE_END -->'
METHODS = [('SeqFT_LoRA', 'SeqFT-LoRA'), ('FOREVER', 'FOREVER'),
           ('CL_MoE', 'CL-MoE'), ('SMoLoRA', 'SMoLoRA'), ('PASs_MoE', 'PASs-MoE'),
           ('PaRSP', 'PaRSP'), ('OSFT', 'OSFT'), ('SpaRTA', 'SpaRTA'), ('Ours', 'Ours')]
BACKBONES = [('qwen', 'Qwen2-VL-2B-Instruct'),
             ('onevision', 'LLaVA-OneVision-Qwen2-0.5B-OV')]
DOMAINS = ['natural', 'medical', 'geography', 'finance', 'traffic']
COLLECTOR_SHA = 'd7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a'


def verified_rows(index):
    assert index['CPU_only'] and index['collector_sha256'] == COLLECTOR_SHA
    assert index['primary_AP19_unchanged'] and not index['TEST_best_run_selection']
    result = {}
    for item in index['methods']:
        key = (item['backbone'], item['method'])
        assert key not in result and item['method'] in index['methods_in_scope']
        path = (ROOT/item['proof_file']).resolve()
        assert path.is_relative_to(INDEX.parent/'completed') and path.suffix == '.json'
        proof = json.loads(path.read_text(encoding='utf-8'))
        assert (proof['candidate']['backbone'], proof['candidate']['method']) == key
        assert proof['candidate']['collector_sha256'] == COLLECTOR_SHA
        assert proof['CPU_only'] and proof['rescored_cells'] == 74 and proof['final_QA'] == 21454
        assert proof['positive_finite_training_stages'] == 19
        assert proof['primary_metric'] == 'existing raw Answer F1'
        assert len(proof['checkpoint_sha256']) == 64
        assert proof['aggregate'] == item['aggregate'] and proof['domains'] == item['domains']
        values = [proof['domains'][d]['F1'] for d in DOMAINS]
        values += [proof['aggregate'][m] for m in ['AP', 'FWT', 'BWT']]
        assert all(math.isfinite(v) for v in values)
        result[key] = values
    assert len(result) == index['verified_method_backbone_pairs']
    return result


def block(rows):
    lines = [BEGIN, '## 主实验：双骨干1%多领域持续学习', '',
             'Acc采用唯一的**Answer F1（0–100）**。训练集为冻结的1%（1313条），测试集为完整21454条；五个数据集按19个原生任务顺序训练，seed=42。空白表示尚无完整核验结果，不能视作0分。', '',
             '| 方法／Backbone | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | AP ↑ | FWT ↑ | BWT ↑ |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for backbone, name in BACKBONES:
        lines.append('| **'+name+'** | | | | | | | | |')
        for method, label in METHODS:
            values = rows.get((backbone, method))
            cells = [f'{v:.3f}' for v in values] if values is not None else ['']*8
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
    lines += ['', '五个数据集列为训练结束后的完整域内QA平均F1；AP为19个原生任务等权平均，FWT/BWT按已测的必要阶段计算。两种权重口径不同。FOREVER（ACL 2026）待图文适配与准入；Ours尚无可填入此正式表的完整结果。ER-LoRA已按用户要求移出比较，已有结果保留在历史证据中。', '',
              '[方法分类、发表信息及状态](summary/two_backbone_pct01_20261017/BASELINES.md) · '
              '[完整主表与协议](summary/two_backbone_pct01_20261017/MAIN_TABLE.md) · '
              '[逐题核验索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json)', END]
    return '\n'.join(lines)


def render(text, content):
    newline = '\r\n' if '\r\n' in text else '\n'
    content = content.replace('\n', newline)
    if BEGIN in text or END in text:
        assert text.count(BEGIN) == text.count(END) == 1
        left, rest = text.split(BEGIN)
        _, right = rest.split(END)
        return left+content+right
    marker = '## 当前实验与阅读入口'
    assert text.count(marker) == 1
    return text.replace(marker, content+newline+newline+marker, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    source = INDEX.read_bytes()
    rows = verified_rows(json.loads(source))
    target = ROOT/'README.md'
    before = target.read_bytes()
    after = render(before.decode('utf-8'), block(rows)).encode('utf-8')
    assert render(after.decode('utf-8'), block(rows)).encode('utf-8') == after
    assert INDEX.read_bytes() == source, 'Concurrent index publication; retry from new sealed index'
    if args.write and before != after:
        assert target.read_bytes() == before, 'Concurrent README edit'
        tmp = target.with_suffix('.md.main_table.tmp')
        tmp.write_bytes(after)
        tmp.replace(target)
    print(json.dumps({'CPU_only': True, 'verified_rows': len(rows), 'backbones': len(BACKBONES),
                      'planned_methods_per_backbone': len(METHODS)-1, 'ours_rows': 2,
                      'missing_results_empty': True, 'changed': before != after,
                      'README_written': args.write, 'index_sha256': hashlib.sha256(source).hexdigest()}))


if __name__ == '__main__':
    main()
