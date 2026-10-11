"""Download, verify and render the manually reviewed LLM continual-learning collection.

The manifest is the curated source of truth; this script does not select papers or
infer acceptance from arXiv. Run: python3 collect_cl_papers.py [--download] [--render].
"""
import argparse, hashlib, html, json, subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CATS = ['pretraining', 'replay', 'regularization', 'architecture', 'alignment']
NAMES = dict(pretraining='持续预训练', replay='基于回放', regularization='基于正则化', architecture='基于架构', alignment='持续对齐')

def ordered(records):
    return sorted(records, key=lambda r: (-r['year'], r['venue'] != 'CVPR', r['venue'] != 'ICCV', r['venue'], r['method']))

def render(records):
    count=Counter(r['category'] for r in records)
    years=Counter(r['year'] for r in records)
    venues=Counter(r['venue'] for r in records)
    n=len(records); ft=sum(count[c] for c in CATS[1:4])
    out=['# 2024–2026 顶会大语言模型持续学习论文', '',
         '逐篇检索与正文核验日期：**2026-10-11**。年份按正式会议届次，而非 arXiv 首次上传年份或网站上线年份。仅收录已进入正式主会论文集的论文，不含 workshop、Findings、仅录用名单或未发表预印本。', '',
         f'共 **{n} 篇**：持续预训练 {count["pretraining"]} 篇、持续微调 **{ft} 篇**、持续对齐 {count["alignment"]} 篇。年份：'+ '、'.join(f'{y} 年 {years[y]} 篇' for y in sorted(years,reverse=True))+'。会议：'+'、'.join(f'{v} {venues[v]} 篇' for v in ['CVPR','ICCV','ACL','ICML','ICLR','NeurIPS','AAAI'] if venues[v])+'。各列按年份降序，同年优先 CVPR / ICCV。', '',
         '## 目录', '', '```text', 'paper/', '├── README.md', '├── verification_audit.md          # 逐篇发表与任务核验；旧清单处理记录', '├── continual_learning_manifest.json', '├── collect_cl_papers.py', f'├── pre_training/                  # {count["pretraining"]}', f'├── pre_finetuning/                # {ft}', f'│   ├── method_replay/             # {count["replay"]}', f'│   ├── method_regularization/     # {count["regularization"]}', f'│   └── method_architecture/       # {count["architecture"]}', f'└── pre_alignment/                # {count["alignment"]}', '```', '',
         '服务器路径：`/data2/liyapeng_grp/program/LLMCL/paper`。原有顶层医学论文不计入本清单。', '',
         '## 收录与分类口径', '',
         '- **模型与任务**：文本 LLM 或带 LLM 解码器的生成式 MLLM，实际研究持续适配、序贯指令学习或持续能力保持。排除 CLIP / ViT 分类、图像分割、扩散模型、普通视觉自监督，以及仅把分类标签转成文本的 LLM 实验。混合评测论文必须包含实质性的生成实验；本表只以生成、代码、数学、开放式问答、摘要或描述任务为收录依据。',
         '- **持续预训练**：使用语言建模或知识蒸馏目标继续训练。Diffract、Data Mixing Agent 等研究领域适配与能力保持；藏语 CPT、Cite Pretrain、Glyph 属语言、知识或上下文扩展型 CPT，**不能据此宣称具有标准序贯多任务抗遗忘实验**。DiSC 是对已后训练模型进行文档知识更新，按训练目标归入此类。',
         '- **持续微调**：序贯监督指令或任务适配。回放、正则化、架构按主要抗遗忘机制归档，每篇只存一份；混合机制在详情中注明。ECA 的 alignment 是跨模态映射适配，归入微调的架构类。',
         '- **持续对齐**：LifeAlign / CPPO 是序贯偏好或奖励对齐；HPA 研究已安全对齐模型在持续视觉指令微调中的安全保持，属于持续安全对齐保持，**不是新的 RLHF 算法**。',
         '- **较新优先**：优先 2026，适当保留 2025 的代表方法。2024 仅保留明确研究动态偏好持续 RLHF 的 CPPO。', '',
         '## 方法 × 方向二维表', '',
         '横轴为方向，持续微调下设三个子类；纵向列出方法（点击方法名查看正式发表页面）。同一行不同列之间没有对应关系。', '', '<table>', '<thead>',
         f'<tr><th rowspan="2">持续预训练（{count["pretraining"]}）</th><th colspan="3">持续微调（{ft}）</th><th rowspan="2">持续对齐（{count["alignment"]}）</th></tr>',
         '<tr>'+''.join(f'<th>{NAMES[c]}（{count[c]}）</th>' for c in CATS[1:4])+'</tr>', '</thead>', '<tbody>']
    columns=[[r for r in ordered(records) if r['category']==c] for c in CATS]
    for i in range(max(map(len,columns))):
        out.append('<tr>')
        for col in columns:
            if i>=len(col): out.append('<td>—</td>'); continue
            r=col[i]; esc=lambda s:html.escape(str(s),quote=True)
            extra='<br>作者扩展版' if r.get('pdf_version') else ''
            out.append(f'<td><a href="{esc(r["page"])}" title="{esc(r["title"])}"><strong>{esc(r["method"])}</strong></a><br>{r["venue"]} {r["year"]}<br><a href="{esc(r["file"])}">PDF</a>{extra}</td>')
        out.append('</tr>')
    out+=['</tbody>','</table>', '', '## 逐篇详情', '', '<details>', '<summary>展开完整标题、模型、生成任务和分类依据</summary>', '', '| 方法与正式论文 | 会议 | 模型及生成任务 | 主要机制与归类 |', '|---|---|---|---|']
    for c in CATS:
        for r in [r for r in ordered(records) if r['category']==c]:
            clean=lambda s:s.replace('|','／').replace('\n',' ')
            out.append(f'| **{r["method"]}** — [{clean(r["title"])}]({r["page"]}) | {r["venue"]} {r["year"]} | {clean(r["model"])}；{clean(r["generation_tasks"])} | {NAMES[c]}：{clean(r["mechanism"])}；{clean(r["scope"])} |')
    out+=['', '</details>', '', '## 核验与下载', '',
          '每篇的正式来源、发表年份依据、正文模型／任务定位、SHA-256 和本地 PDF 路径见 [逐篇核验记录](verification_audit.md) 及 [manifest](continual_learning_manifest.json)。旧清单 28 篇全部重新检索，保留和移除理由逐项记录。归类由整理者按方法正文判断。', '',
          '**LifeAlign** 已通过 [AAAI 正式论文页](https://ojs.aaai.org/index.php/AAAI/article/view/40428) 核实发表（2026-03-14，DOI: 10.1609/aaai.v40i37.40428）。服务器无法访问 AAAI 下载端点，本地保存正式页面注明的 arXiv 作者扩展版，表中明确标记；并非将未发表预印本计入。其余 PDF 均来自正式会议论文集或 PMLR 官方资产仓库。', '',
          '复现：Python 3 与 Poppler 的 `pdftotext`，下载时另需 requests。运行 `python3 collect_cl_papers.py` 校验所有 PDF；`--download` 补充缺失文件，`--render` 从人工核验过的 manifest 重建 README。脚本不会自动认定录用、替换筛选标准或删除其他文件。', '',
          '这是按上述口径整理的精选清单，不是全部相关论文的穷尽性综述。']
    return '\n'.join(out)+'\n'

def verify(records,download=False):
    for r in records:
        path=(ROOT/r['file']).resolve()
        assert ROOT.resolve() in path.parents, r['file']
        if not path.exists() and download:
            import requests
            response=requests.get(r['pdf'],timeout=120); response.raise_for_status()
            data=response.content
            assert data.startswith(b'%PDF-'),r['method']
            assert hashlib.sha256(data).hexdigest()==r['sha256'], 'Remote PDF changed; review the new version manually: '+r['method']
            path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(data)
        data=path.read_bytes()
        assert data.startswith(b'%PDF-') and len(data)==r['bytes'],r['file']
        assert hashlib.sha256(data).hexdigest()==r['sha256'],r['file']
        subprocess.run(['pdftotext',str(path),'-'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    print(f'Verified {len(records)} PDFs: header, size, SHA-256, and full text extraction.')

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--download',action='store_true'); p.add_argument('--render',action='store_true'); args=p.parse_args()
    records=json.loads((ROOT/'continual_learning_manifest.json').read_text(encoding='utf-8'))
    verify(records,args.download)
    if args.render: (ROOT/'README.md').write_text(render(records),encoding='utf-8')
