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
      '## 当前服务器与迁移状态','','主仓库已迁移至 `/data1/liyapeng/program/LLMCL`，通过 Tailscale `liyapeng@100.87.141.22:8666` 工作。资源内容与环境检查已通过；约 177 GiB 的旧重复仓库已清理。未整包迁移的 GeoChat 原始数据及维持旧端连接的 Tailscale 环境暂时保留。详见 [迁移与清理报告](summary/ten_task_cl_20261012/migration_report.md)。','',
      '## 方法 × 方向二维表','',
      f'已整理 **{len(records)} 篇** 2024–2026 年正式会议论文：持续预训练 {counts["pretraining"]} 篇、持续微调 {ft} 篇、持续对齐 {counts["alignment"]} 篇。各方向列直接列出方法名，持续微调细分回放、正则化和架构；列内优先较新论文，同年优先 CVPR／ICCV。横向同一行的方法之间没有对应关系。点击方法名进入正式论文页，点击 PDF 查看仓库文件。年份按正式会议届次；逐篇核验记录见 [paper/verification_audit.md](paper/verification_audit.md)。','',
      '<table>','<thead>',
      '<tr><th rowspan="2">持续预训练</th><th colspan="3">持续微调</th><th rowspan="2">持续对齐</th></tr>',
      '<tr><th>基于回放</th><th>基于正则化</th><th>基于架构</th></tr>',
      '</thead>','<tbody>']
    columns=[[p for p in ordered if p['category']==category] for category in CATS]
    for index in range(max(map(len,columns))):
        cells=[]
        for column in columns:
            if index>=len(column):cells.append('');continue
            p=column[index]
            esc=lambda x:html.escape(str(x),quote=True)
            pdfnote='（作者扩展版）' if p.get('pdf_version') else ''
            cells.append(f'<a href="{esc(p["page"])}" title="{esc(p["title"])}"><strong>{esc(p["method"])}</strong></a><br>{esc(p["venue"])} {p["year"]}<br><a href="paper/{esc(p["file"])}">PDF</a>{pdfnote}')
        text.append('<tr>'+''.join('<td>'+cell+'</td>' for cell in cells)+'</tr>')
    text+=['</tbody>','</table>','',
      '归类按主要机制，混合机制、原论文模型及生成任务见 [论文完整索引](paper/README.md)。预训练条目包含语言／知识／上下文扩展型 CPT，不全部具有序贯抗遗忘实验；HPA 归入持续安全对齐保持。**回放方法仍保留在论文表中，本轮不做回放实验；DGG 含历史样本回放，也不参与本轮。**','',
      '## 本轮实验','',
      '新任务数据核验通过：T5 GeoQuestions1089 计数／人口文本问答、T6 GeoChat 遥感开放问答、T8 MME-Finance 原始图表开放问答。各任务训练 100／测试 50，详见 [逐项核验及划分依据](data/DATASET_VERIFICATION_20261014.md)。FinChart-Bench 未通过多篇已发表复用要求，保留候选记录，不进入本轮任务。','',
      '**数据核验完成，方法适配／GPU 预检待完成。** 五个领域为自然／通用、医学、地理／遥感、金融、艺术文化。法律域已按用户授权替换。金融图像仅使用 K 线、技术指标和统计图表，排除文字文档页；无验证集，每任务 100／50，seed 42。','',
      '| 任务 | 领域 | 模态 | 数据集 | 训练／测试 |','|---|---|---|---|---:|',
      '| T1 | 自然／通用 | 文本 | SQuAD 1.1 | 100／50 |',
      '| T2 | 自然／通用 | 图像 | VQAv2、GQA、TextVQA | 100／50 |',
      '| T3 | 医学 | 文本 | MedQuAD | 100／50 |',
      '| T4 | 医学 | 图像 | VQA-RAD、PathVQA、SLAKE | 100／50 |',
      '| T5 | 地理／遥感 | 文本 | GeoQuestions1089：计数／人口问答 | 100／50 |',
      '| T6 | 地理／遥感 | 图像 | GeoChat_Instruct：开放 QA | 100／50 |',
      '| T7 | 金融 | 文本 | FinQA | 100／50 |',
      '| T8 | 金融 | 图像 | MME-Finance：PC 原始图表 QA | 100／50 |',
      '| T9 | 艺术文化 | 文本 | AQUA 知识问答子集 | 100／50 |',
      '| T10 | 艺术文化 | 图像 | AQUA 视觉问答子集 | 100／50 |','',
      '原始问题、答案和图像保持不变；新任务采用论文支持的问答类型和作者图像类别。新数据没有官方训练／测试划分，因此本研究按 seed 42 构造固定小数据划分，明确披露，绝不称为作者官方划分。T6 按图像隔离，T8 按底层图表隔离；不同方法共享相同的样本 ID。见 [核验报告](data/DATASET_VERIFICATION_20261014.md) 与 [固定协议](rules/012_small_data_results_only.md)。','',
      '每个方法从同一基座开始，先评测全部十任务，再按 **T1→T10** 连续训练；每训练完一个任务，评测全部十任务，各 50 条，形成 **11×10** 测试矩阵。验证集已取消，不进行验证推理或最佳 checkpoint 选择。阶段间模型和方法必需的历史参考留在内存中；进程中断需从基座重跑，旧预测保留但不与新运行混算。','',
      '保存逐题预测与参考答案、训练日志、每任务及每源数据集的 EM 正确数／总数、Token F1、ROUGE-L。完整运行后分别计算 **AP、ForgetRate、FWT、BWT**，输出热图和性能曲线（PNG／PDF／绘图数据）。统一 EM 与词面分数不冒称各数据集的官方指标。','',
      '计划比较二维表中的 **SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER** 八种非回放论文方法。不运行 SeqLoRA。必须保留作者核心机制，并公开 Qwen2-VL 与小数据适配差异；下载源码不等于完成适配或开始实验。','',
      '当前状态以 [对比进度](summary/ten_task_cl_20261012/comparison_report.md) 为准；方法依据和固定源码版本见 [code/README.md](code/README.md)。多卡按空闲状态分配独立方法。','',
      '## 目录与入口','',
      '```text','LLMCL/','├── README.md                         # 当前实验与论文二维表','├── paper/                            # 论文、PDF、发表核验','│   ├── pre_training/','│   ├── pre_finetuning/','│   │   ├── method_replay/','│   │   ├── method_regularization/','│   │   └── method_architecture/','│   └── pre_alignment/',
      '├── code/                             # 各方法作者源码与适配依据','├── data/                             # 数据资产、来源依据与固定样本清单','├── model/                            # 固定模型资产','├── exp/','│   ├── ten_task_cl_20261012/          # 当前入口、运行日志和逐题结果','│   └── ago/                         # 历史实验与旧脚本','├── summary/','│   ├── ten_task_cl_20261012/          # 当前汇总、指标和图','│   └── ago/                         # 历史报告（含旧 CV 与医学实验）','└── rules/                            # 实验协议与调整记录','```','',
      '[当前运行说明](exp/ten_task_cl_20261012/README.md) · [执行策略](exp/ten_task_cl_20261012/execution_policy.json) · [固定样本 ID 与校验值](data/TenTaskCL/current_small_manifest.json) · [历史报告](summary/ago/README.md) · [历史实验](exp/ago/README.md)','',
      '服务器仓库迁移目标：`/data1/liyapeng/program/LLMCL`。确认没有现有队列后启动：','',
      '```bash','.vision-env/bin/python exp/ten_task_cl_20261012/comparison_queue.py','```','',
      '论文表依据已核验的清单生成；只更新根 README、无需重新下载：','',
      '```bash','python3 paper/update_repository_readme.py','```','',
      '根目录不再保留旧医学报告的兼容符号链接。旧实验保存在 `exp/ago/`，旧报告保存在 `summary/ago/`；历史实验的模型、配额和结论与本轮分别解读。`sample/` 只展示本轮十任务数据；旧示例资产归档至 `summary/ago/sample_assets/`。`CV_data/`、`source_code/` 保留历史数据与源码。']
    samples=json.loads((ROOT/'data/TenTaskCL/samples.json').read_text())['examples']
    sample_lines=['## 本轮数据集 sample 示例','',
      '每个任务取固定训练清单的第一条，展示真实问题和参考答案；这是数据样例，不是模型回答。表中长文本仅作展示节选，原始问题／答案未修改。点击任务编号查看完整问题、所有参考答案、上下文及图像。','',
      '| 任务／来源 | 问题（原文节选） | 参考答案（原文节选） | 图像输入 |','|---|---|---|---|']
    def excerpt(value,limit):
        value=str(value).replace('\n',' ').replace('|','／')
        return html.escape(value[:limit])+('…（节选）' if len(value)>limit else '')
    for example in samples:
        task=example['task'];row=example['raw_example']
        if False: # All ten current tasks now verified; no pending placeholders.
            sample_lines.append(f'| T{task:02d} | 数据集修订中 | 暂不展示已停用的旧样例 | 等待核验新样例 |')
            continue
        answers=row.get('answers',[row['answer']])
        image=(f'<img src="{example["sample_images"][0]}" width="180" alt="T{task:02d} 原始输入图像">' if example['sample_images'] else '文本输入，完整上下文见样例页')
        sample_lines.append(f'| [T{task:02d}](data/TenTaskCL/samples.md#t{task:02d}) · {example["dataset"]} | {excerpt(row["question"],140)} | {excerpt(answers[0],140)} | {image} |')
    position=text.index('## 目录与入口')
    text[position:position]=sample_lines+['']
    return '\n'.join(text)+'\n'
if __name__=='__main__':
    (ROOT/'README.md').write_text(render(),encoding='utf-8')
