"""Extend the outer formal main table; never edit the running result collector.

Formal Answer F1 cells still pass the original sealed-proof verifier. Deployment
cells remain blank until a measured, checkpoint-bound profile is provided.
"""
import argparse
import hashlib
import json
from pathlib import Path
import v2_readme_main_table as formal

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/edge_deployment_20261019'
HEADERS = ['最终部署<br>MiB ↓', '新增部署<br>MiB ↓', '推理峰值内存<br>CPU RAM／GPU VRAM MiB ↓',
           '首词延迟<br>CPU／GPU s ↓', '生成速度<br>CPU／GPU tok/s ↑', '常驻适配器<br>数量 ↓']
FIELDS = ['final_package_MiB', 'package_growth_MiB', 'cpu_peak_RSS_MiB',
          'ttft_seconds', 'decode_tokens_per_second', 'resident_adapters']


def table_with_edge(rows):
    # No deployment measurements have yet been admitted. Do not estimate values
    # from a pickle checkpoint's size or the count of trainable parameters.
    registry = json.loads((REPORT/'RESULTS.json').read_text())
    assert registry['schema'] == 'cpu-edge-deployment-v1'
    assert registry['profiles'] == [], 'Add a checkpoint/device proof verifier before publishing measured profiles'
    raw = [line for line in formal.block(rows).splitlines() if line.startswith('|')]
    result = []
    for i, line in enumerate(raw):
        cells = [s.strip() for s in line.split('|')[1:-1]]
        assert len(cells) == 11
        extra = HEADERS if i == 0 else (['---:']*6 if i == 1 else ['']*6)
        result.append('| '+' | '.join(cells+extra)+' |')
    return '\n'.join(result)


def render(text):
    index_bytes = formal.INDEX.read_bytes()
    rows = formal.verified_rows(json.loads(index_bytes))
    left, rest = text.split(formal.BEGIN)
    block, right = rest.split(formal.END)
    lines = block.splitlines()
    positions = [i for i, line in enumerate(lines) if line.startswith('|')]
    assert positions == list(range(min(positions), max(positions)+1))
    # Existing published accuracy numbers must match the sealed original proof.
    current, backbone = {}, None
    labels = {label: method for method,label in formal.METHODS}
    for i in positions[2:]:
        cells = [s.strip() for s in lines[i].split('|')[1:-1]]
        if cells[0].startswith('**'):
            backbone = next(k for k,v in formal.BACKBONES if v in cells[0])
        else:
            key = (backbone, labels[cells[0]])
            existing = cells[3:11]
            if key in rows:
                expected = [f'{v:.3f}' for v in rows[key]]
                # A background verifier can seal a new result after a README
                # snapshot. Fill a previously blank row from that proof; never
                # silently replace conflicting published scores.
                assert existing == expected or existing == ['']*8
                cells[3:11] = expected
                lines[i] = '| '+' | '.join(cells)+' |'
            current[key] = cells[3:11]
    for key, vals in rows.items():
        assert current[key] == [f'{v:.3f}' for v in vals]
    new_table = table_with_edge(rows)
    lines[min(positions):max(positions)+1] = new_table.splitlines()
    note = ('端侧部署包括**个人电脑CPU／独立GPU及手机**。最终部署大小包含基座、视觉编码器、投影和全部必需适配器／专家／路由／索引；'
            '新增部署大小为同格式、同精度下最终包减初始包。训练优化器、日志和历史checkpoint不计入部署包。'
            '参数值更新不等于参数量增加，可折叠更新按实际导出后大小计量。CPU RAM与GPU显存分别报告，延迟与速度注明CPU／GPU设备档；'
            '方法对比须使用相同设备、线程、精度、输入及生成预算。服务器CPU及服务器安装的消费级GPU须标明实际整机环境，不能称手机实测。'
            '当前部署列均未实测，保持空白。量化／剪枝等改变模型输出的部署版本须单列并验证其Answer F1。'
            ' [指标与一般性问题协议](summary/edge_deployment_20261019/PROTOCOL.md)。')
    marker = '<!-- CPU_EDGE_METRICS_NOTE -->'
    old = '\n'.join(lines)
    if marker in old:
        before, tail = old.split(marker)
        _, after = tail.split('<!-- CPU_EDGE_METRICS_NOTE_END -->')
        old = before+marker+'\n'+note+'\n<!-- CPU_EDGE_METRICS_NOTE_END -->'+after
    else:
        old = old.rstrip()+'\n\n'+marker+'\n'+note+'\n<!-- CPU_EDGE_METRICS_NOTE_END -->\n'
    assert formal.INDEX.read_bytes() == index_bytes
    return left+formal.BEGIN+old+'\n'+formal.END+right


def main():
    parser = argparse.ArgumentParser();parser.add_argument('--write',action='store_true');a=parser.parse_args()
    path=ROOT/'README.md';before=path.read_text();after=render(before)
    assert render(after)==after
    # Two backbones x nine displayed rows, five task scores and AP/FWT/BWT stay intact.
    block=after.split(formal.BEGIN)[1].split(formal.END)[0]
    table=[line for line in block.splitlines() if line.startswith('|')]
    assert len(table)==22 and all(len(line.split('|')[1:-1])==17 for line in table)
    if a.write:
        assert path.read_text()==before
        path.write_text(after)
    REPORT.mkdir(exist_ok=True)
    (REPORT/'TABLE_CHECK.json').write_text(json.dumps(dict(CPU_only=True,rows=22,columns=17,
        deployment_columns=6,empty_until_measured=True,formal_scores_unchanged=True,
        idempotent=True,original_collector_unchanged=True,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(json.dumps(dict(table_updated=a.write,deployment_columns=6,measurements_admitted=0)))


if __name__=='__main__':main()
