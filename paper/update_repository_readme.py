"""Update the paper table only; preserve experiment and dataset documentation."""
import html
import json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CATS=['pretraining','replay','regularization','architecture','alignment']

def render():
    current=(ROOT/'README.md').read_text(encoding='utf-8')
    heading='## 方法 × 方向二维表'
    start=current.index(heading)
    end=current.find('\n## ',start+len(heading))
    if end<0:raise ValueError('Expected experiment/data sections after the paper table')
    records=json.loads((ROOT/'paper/continual_learning_manifest.json').read_text())
    counts=Counter(p['category'] for p in records)
    ordered=sorted(records,key=lambda p:(CATS.index(p['category']),-p['year'],p['venue']!='CVPR',p['venue']!='ICCV',p['venue'],p['method']))
    ft=sum(counts[c] for c in CATS[1:4])
    text=[heading,'',f'已整理 **{len(records)} 篇** 2024–2026 年正式会议论文：持续预训练 {counts["pretraining"]} 篇、持续微调 {ft} 篇、持续对齐 {counts["alignment"]} 篇。各列直接列方法名，持续微调细分回放、正则化和架构；较新论文优先，同年优先 CVPR／ICCV。横向同一行的方法没有对应关系。年份按正式会议届次；逐篇核验见 [发表核验](paper/verification_audit.md)。','', '<table>','<thead>', '<tr><th rowspan="2">持续预训练</th><th colspan="3">持续微调</th><th rowspan="2">持续对齐</th></tr>', '<tr><th>基于回放</th><th>基于正则化</th><th>基于架构</th></tr>', '</thead>','<tbody>']
    columns=[[p for p in ordered if p['category']==c] for c in CATS]
    for index in range(max(map(len,columns))):
        cells=[]
        for column in columns:
            if index>=len(column):cells.append('');continue
            p=column[index];esc=lambda x:html.escape(str(x),quote=True)
            note='（作者扩展版）' if p.get('pdf_version') else ''
            cells.append(f'<a href="{esc(p["page"])}" title="{esc(p["title"])}"><strong>{esc(p["method"])}</strong></a><br>{esc(p["venue"])} {p["year"]}<br><a href="paper/{esc(p["file"])}">PDF</a>{note}')
        text.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    text+=['</tbody>','</table>','','归类按主要机制，混合机制、原论文模型和生成任务见 [完整索引](paper/README.md)。部分 CPT 条目讨论知识／上下文扩展，并非全部有序贯抗遗忘实验。回放方法仍保留在论文表中，本轮不做回放实验；DGG 含历史样本回放，也不参与本轮。','']
    return current[:start]+'\n'.join(text)+current[end:]

if __name__=='__main__':
    (ROOT/'README.md').write_text(render(),encoding='utf-8')
