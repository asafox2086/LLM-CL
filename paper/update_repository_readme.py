"""Render the repository overview from the reviewed paper manifest; no downloads."""
import html,json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
CATS=['pretraining','replay','regularization','architecture','alignment']
def render():
    records=json.loads((ROOT/'paper/continual_learning_manifest.json').read_text())
    counts=Counter(p['category'] for p in records)
    ordered=sorted(records,key=lambda p:(CATS.index(p['category']),-p['year'],p['venue']!='CVPR',p['venue']!='ICCV',p['venue'],p['method']))
    ft=sum(counts[c] for c in CATS[1:4])
    text=['# LLMCL：小模型、小数据的多模态持续学习','',
      '本仓库整理生成式大语言模型／多模态大模型的持续学习论文、作者源码和数据依据，并开展五领域十任务的轻量化持续微调实验。当前协议为 **Qwen2-VL-2B-Instruct、每任务训练 100 条／测试 50 条、无验证集、seed 42、每任务训练 1 轮、不保存 checkpoint**。','',
      '## 方法 × 方向二维表','',
      f'已整理 **{len(records)} 篇** 2024–2026 年正式会议论文：持续预训练 {counts["pretraining"]} 篇、持续微调 {ft} 篇、持续对齐 {counts["alignment"]} 篇。每行一个方法，横轴为研究方向，持续微调细分回放、正则化和架构；**●** 表示归档方向。点击方法名进入正式论文页，点击 PDF 查看仓库文件。年份按正式会议届次；逐篇核验记录见 [paper/verification_audit.md](paper/verification_audit.md)。','',
      '<table>','<thead>',
      '<tr><th rowspan="2">方法／论文</th><th rowspan="2">会议</th><th rowspan="2">持续预训练</th><th colspan="3">持续微调</th><th rowspan="2">持续对齐</th></tr>',
      '<tr><th>基于回放</th><th>基于正则化</th><th>基于架构</th></tr>',
      '</thead>','<tbody>']
    for p in ordered:
        esc=lambda x:html.escape(str(x),quote=True)
        pdfnote='（作者扩展版）' if p.get('pdf_version') else ''
        name=f'<a href="{esc(p["page"])}" title="{esc(p["title"])}"><strong>{esc(p["method"])}</strong></a><br><a href="paper/{esc(p["file"])}">PDF</a>{pdfnote}'
        cells=[name,f'{esc(p["venue"])} {p["year"]}']+['●' if p['category']==c else '—' for c in CATS]
        text.append('<tr>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    text+=['</tbody>','</table>','',
      '归类按主要机制，混合机制、原论文模型及生成任务见 [论文完整索引](paper/README.md)。预训练条目包含语言／知识／上下文扩展型 CPT，不全部具有序贯抗遗忘实验；HPA 归入持续安全对齐保持。**回放方法仍保留在论文表中，本轮不做回放实验；DGG 含历史样本回放，也不参与本轮。**','',
      '## 本轮实验','',
      '| 任务 | 领域 | 模态 | 数据集 | 训练／测试 |','|---|---|---|---|---:|',
      '| T1 | 自然／通用 | 文本 | SQuAD 1.1 | 100／50 |',
      '| T2 | 自然／通用 | 图像 | VQAv2、GQA、TextVQA | 100／50 |',
      '| T3 | 医学 | 文本 | MedQuAD | 100／50 |',
      '| T4 | 医学 | 图像 | VQA-RAD、PathVQA、SLAKE | 100／50 |',
      '| T5 | 法学 | 文本 | CUAD | 100／50 |',
      '| T6 | 法学 | 图像 | CUAD 原始合同 PDF 页 | 100／50 |',
      '| T7 | 金融 | 文本 | FinQA | 100／50 |',
      '| T8 | 金融 | 图像 | TAT-DQA | 100／50 |',
      '| T9 | 艺术文化 | 文本 | AQUA 知识问答子集 | 100／50 |',
      '| T10 | 艺术文化 | 图像 | AQUA 视觉问答子集 | 100／50 |','',
      '均使用原始问题、答案和标注；图像任务使用真实图像／原始文档页，不将分类标签包装成开放问答。CUAD 图像视图和 AQUA 文本视图的适配口径、采样及检索细节见 [固定协议](rules/012_small_data_results_only.md) 与 [数据溯源](data/DATASET_PROVENANCE.md)。不同方法共享相同的训练／测试样本 ID。','',
      '每个方法从同一基座开始，先评测全部十任务，再按 **T1→T10** 连续训练；每训练完一个任务，评测全部十任务，各 50 条，形成 **11×10** 测试矩阵。验证集已取消，不进行验证推理或最佳 checkpoint 选择。阶段间模型和方法必需的历史参考留在内存中；进程中断需从基座重跑，旧预测保留但不与新运行混算。','',
      '保存逐题预测与参考答案、训练日志、每任务及每源数据集的 EM 正确数／总数、Token F1、ROUGE-L。完整运行后分别计算 **AP、ForgetRate、FWT、BWT**，输出热图和性能曲线（PNG／PDF／绘图数据）。统一 EM 与词面分数不冒称各数据集的官方指标。','',
      '计划比较二维表中的 **SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER** 八种非回放论文方法。不运行 SeqLoRA。必须保留作者核心机制，并公开 Qwen2-VL 与小数据适配差异；下载源码不等于完成适配或开始实验。','',
      '当前状态以 [对比进度](summary/ten_task_cl_20261012/comparison_report.md) 为准；方法依据和固定源码版本见 [code/README.md](code/README.md)。多卡按空闲状态分配独立方法。','',
      '## 目录与入口','',
      '```text','LLMCL/','├── README.md                         # 当前实验与论文二维表','├── paper/                            # 论文、PDF、发表核验','│   ├── pre_training/','│   ├── pre_finetuning/','│   │   ├── method_replay/','│   │   ├── method_regularization/','│   │   └── method_architecture/','│   └── pre_alignment/',
      '├── code/                             # 各方法作者源码与适配依据','├── data/                             # 数据资产、来源依据与固定样本清单','├── model/                            # 固定模型资产','├── exp/','│   ├── ten_task_cl_20261012/          # 当前入口、运行日志和逐题结果','│   └── ago/                         # 历史实验与旧脚本','├── summary/','│   ├── ten_task_cl_20261012/          # 当前汇总、指标和图','│   └── ago/                         # 历史报告（含旧 CV 与医学实验）','└── rules/                            # 实验协议与调整记录','```','',
      '[当前运行说明](exp/ten_task_cl_20261012/README.md) · [执行策略](exp/ten_task_cl_20261012/execution_policy.json) · [固定样本 ID 与校验值](data/TenTaskCL/current_small_manifest.json) · [历史报告](summary/ago/README.md) · [历史实验](exp/ago/README.md)','',
      '服务器仓库：`/data2/liyapeng_grp/program/LLMCL`。确认没有现有队列后启动：','',
      '```bash','.vision-env/bin/python exp/ten_task_cl_20261012/comparison_queue.py','```','',
      '论文表依据已核验的清单生成；只更新根 README、无需重新下载：','',
      '```bash','python3 paper/update_repository_readme.py','```','',
      '根目录不再保留旧医学报告的兼容符号链接。旧实验保存在 `exp/ago/`，旧报告保存在 `summary/ago/`；历史实验的模型、配额和结论与本轮分别解读。`CV_data/`、`source_code/`、`sample/` 为保留的历史数据、源码与展示资产，尚未并入当前实验目录。']
    return '\n'.join(text)+'\n'
if __name__=='__main__':
    (ROOT/'README.md').write_text(render(),encoding='utf-8')
