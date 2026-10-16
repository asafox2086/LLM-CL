"""Render the root dataset report from audited native metadata, preserving the paper table."""
import html
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
ID=Path(__file__).resolve().parent.name
BASE=f'data/TenTaskCL/{ID}'
audit=json.loads((ROOT/BASE/'readme_dataset_audit.json').read_text())
rows={x['dataset']:x for x in audit['records']}
# Name, task, modality, introduction, actual download source, scope.
catalog=[
('SQuAD11','T1','文本','Pranav Rajpurkar 等 · EMNLP 2016','https://aclanthology.org/D16-1264/','https://rajpurkar.github.io/SQuAD-explorer/','官方 train / dev；段落阅读问答'),
('TextVQA','T2','**多模态**','Amanpreet Singh 等 · CVPR 2019','https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html','https://huggingface.co/datasets/lmms-lab/textvqa','官方 train / validation 的格式镜像；真实图像中的文字问答'),
('MedQuAD','T3','文本','Asma Ben Abacha、Dina Demner-Fushman · BMC Bioinformatics 2019','https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/','https://github.com/abachaa/MedQuAD','公开有答案的 XML；本研究按原始文档分组划分'),
('VQA_RAD','T4','**多模态**','Jason J. Lau 等 · Scientific Data 2018','https://www.nature.com/articles/sdata2018251','https://huggingface.co/datasets/flaviagiammarino/vqa-rad','所下载镜像的 train / test；不是声称原论文所有记录，保留 yes/no'),
('SLAKE','T4','**多模态**','Bo Liu 等 · ISBI 2021','https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/','https://www.med-vqa.com/slake/','作者 train / test；中英双语、OPEN/CLOSED 均保留'),
('GeoQuestions1089','T5','文本','Sergios-Anestis Kefalidis 等 · ISWC 2023','https://iswc2023.semanticweb.org/wp-content/uploads/2023/11/142660264.pdf','https://github.com/AI-team-UoA/GeoQuestions1089','原始地理问答与结果表；本研究按查询组划分'),
('RSVQA-LR','T6','**多模态**','Sylvain Lobry 等 · TGRS 2020','https://arxiv.org/abs/2003.07333','https://zenodo.org/records/6344334','作者原始 LR train / test；遵守 active 划分掩码，保留全部题型与原数字'),
('FinQA','T7','文本＋原表格','Zhiyu Chen 等 · EMNLP 2021','https://aclanthology.org/2021.emnlp-main.300/','https://github.com/czyssrs/FinQA','作者 train / test；表格数值推理，保留 yes/no'),
('MME-Finance','T8','**多模态**','Ziliang Gan 等 · ACM MM 2025','https://doi.org/10.1145/3746027.3758230','https://hithink-research.github.io/MME-Finance/','原始英文 MMfin.tsv 全部 1,171 QA；本研究按底层图表组划分'),
('AQUA','T9／T10','文本／**多模态**','Noa Garcia 等 · ECCV VISART Workshop 2020','https://github.com/noagarcia/ArtVQA','https://github.com/noagarcia/ArtVQA/tree/master/AQUA','作者 train / test；按原标记区分知识、视觉；同一数据集只列一次'),
]

def esc(value,limit=None):
    value=str(value)
    if limit and len(value)>limit:value=value[:limit]+'…（展示节选）'
    return html.escape(value).replace('|','&#124;').replace('\r\n','<br>').replace('\n','<br>')

text=['## 本轮实验与数据状态','',
f'当前实验编号为 **`{ID}`**。旧的 100 训练／50 测试实验已停止，代码及结果归档到 [exp/ago](exp/ago/ten_task_cl_20261014_geo_fin/README.md) 与 [summary/ago](summary/ago/ten_task_cl_20261014_geo_fin/comparison_report.md)，不与本轮混算。','',
'基座 **Qwen2-VL-2B-Instruct**；seed **42**；每任务 **1 轮**；无验证集；不保存模型、优化器或恢复 checkpoint。五领域按自然／通用→医学→地理／遥感→金融→艺术文化排列，每领域先文本后图像，共 **10 个任务**。','',
'| 条件 | 训练池 | 测试池 | GPU |','|---|---|---|---|',
'| full | 所选公开发布配置的全部训练 QA | 全部公开有答案的评测 QA | GPU3，方法串行 |',
'| pct01 | 同一训练池按来源和原始题型分层取 1% | 与 full 完全相同的测试 QA | GPU1／2，并行方法 |','',
'**GPU0 保留。** 两组均比较 SeqFT-LoRA 朴素顺序微调控制，以及 SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。控制基线属于标准顺序微调，不作为新算法。','',
'**当前是数据资产准备阶段，尚未开始本轮正式训练。** 下表已从服务器原始 JSON／XML／Parquet 核对元数据数量；这不等于全部图像已齐，也不是已经冻结的最终运行清单。2026-10-16 按用户要求撤出 VQAv2、GQA、LLaVA-Med、GeoChat 大规模池；T2 仅 TextVQA，T4 为 VQA-RAD＋SLAKE，T6 为 RSVQA-LR。RSVQA-LR 官方原文件正在直连下载，AQUA 的 SemArt 原图仍在准备。旧资产不进入本轮，也不删除原始文件。实际状态见 [队列进度](summary/'+ID+'/progress.md)。','',
'## 数据集：来源、数量与划分','',
f'核验时间：`{audit["audited_at"]}`。**数量单位统一为 QA 对**，不是图像数，也不是多轮对话数。“全量”限定在表中明确的发布配置；原始分类、是非、选择、定位和开放题均保留，不再人工剔除题型。完整机器记录及原始样例见 [数据审计 JSON]('+BASE+'/readme_dataset_audit.json)。','',
'| 数据集 | 任务／模态 | 提出者与发表来源 | 下载／原始资源 | 全量训练 QA | 1% 训练 QA | 全量测试 QA | 配置与划分 |',
'|---|---|---|---|---:|---:|---:|---|']
for name,task,modality,origin,paper,download,scope in catalog:
    x=rows[name]
    text.append(f'| {name} | {task} · {modality} | [{origin}]({paper}) | [资源]({download}) | {x["full_train"]:,} | {x["pct01_train"]:,} | {x["full_test"]:,} | {scope} |')
text += ['',
'**单来源和单任务训练均少于 10 万条；十任务全量合计仍约 28.2 万，不能混称整个实验只有几万。** 详见 [多领域数据集调研：谁使用、哪个论文、实验位置](data/MULTIDOMAIN_DATASET_SURVEY_20261016.md)。', '',
'**原始规模与使用范围需要区分：**','',
'- **RSVQA-LR**：原始 77,232 QA／772 图像；官方 train 57,223／val 10,005／test 10,004，图像 572／100／100。JSON 内其他角色的 inactive 占位条目不属于该划分；按作者 active 标记读取，不是按题型过滤。本轮不用 val。',
'- **MedQuAD**：原始集合 **47,457 QA**；当前公开源有 **16,407 条非空参考答案**，另 **31,034** 条公开答案缺失。缺失标签无法监督训练／评分，数量和原因单列，不补造答案，也不是按题型删除。',
'- **GeoQuestions1089**：原始 **1,089** 问，公开文件 **1,085** 条有答案，**4** 条为 null；本轮保留所有可评分题型，包括布尔结果、列表与原始 CSV，不只取数值题。',
'- **VQA-RAD**：本表是所下载第三方格式镜像的 **1,793／451** 划分，不能直接冒称原论文完整原始集合；来源已单独标明。',
'- **MME-Finance**：选择英文 MMfin.tsv **1,171 QA**，包含原始图表、表格、文档及其他图像类别；不再只选 PC K 线。它不等于论文所有语言／扩展版本总量。',
'- **AQUA**：官方训练 **69,812**、val **5,124**、测试 **4,912**；本轮不使用 val。作者知识标记对应 T9：训练 **40,244**、1% **403**、测试 **3,642**；视觉标记对应 T10：训练 **29,568**、1% **296**、测试 **1,270**。T9 使用问题文本视图，是本研究的模态设置，不是另一个原生纯文本数据集；缺失画作上下文可能影响可回答性。SLAKE 的官方 validation **2,099** 同样不用。','',
'官方 train／test（或公开带答案的 dev／validation／testdev）优先；无官方划分时，本研究按原始文档／查询／图像／底层图表分组，以 `SHA256(42:group)` 排序，取前 `ceil(20% × 组数)` 作测试，其余训练。组大小不同，QA 比例不必恰好为 80/20。三项研究划分为 MedQuAD、GeoQuestions1089、MME-Finance；RSVQA-LR 使用作者现成划分，不能称作者官方划分。','',
'1% 组在每个来源训练池取 `ceil(0.01 × N)`，按原始题型比例以最大余数分配名额，再按 `SHA256(42:QA_ID)` 固定选取；AQUA 两个任务分别取整。所有方法共享同一清单和训练顺序；两组完整测试文件的 SHA256 必须相同。图像缺失不会通过静默丢样本处理，资产核验通过后再冻结运行清单。','',
'### 十任务数量汇总','',
'| 任务 | 领域／模态 | 全量训练 QA | 1% 训练 QA | 两组共同完整测试 QA |','|---|---|---:|---:|---:|']
tasksets=[(1,'自然／通用 · 文本',['SQuAD11']),(2,'自然／通用 · 图像',['TextVQA']),
          (3,'医学 · 文本',['MedQuAD']),(4,'医学 · 图像',['VQA_RAD','SLAKE']),
          (5,'地理／遥感 · 文本',['GeoQuestions1089']),(6,'地理／遥感 · 图像',['RSVQA-LR']),
          (7,'金融 · 文本＋表格',['FinQA']),(8,'金融 · 图像',['MME-Finance'])]
totals=[0,0,0]
for t,domain,names in tasksets:
    values=[sum(rows[n][key] for n in names) for key in ['full_train','pct01_train','full_test']]
    totals=[a+b for a,b in zip(totals,values)]
    text.append(f'| T{t} | {domain} | '+ ' | '.join(f'{x:,}' for x in values)+' |')
for t,domain,key in [(9,'艺术文化 · 文本','knowledge'),(10,'艺术文化 · 图像','visual')]:
    tr=rows['AQUA']['notes']['train'][key];te=rows['AQUA']['notes']['test'][key];values=[tr,math.ceil(.01*tr),te]
    totals=[a+b for a,b in zip(totals,values)]
    text.append(f'| T{t} | {domain} | '+' | '.join(f'{x:,}' for x in values)+' |')
text += ['| **合计** | **十任务** | '+' | '.join(f'**{x:,}**' for x in totals)+' |','',
'### 后续论文使用依据','',
'完整 [多领域数据集调研](data/MULTIDOMAIN_DATASET_SURVEY_20261016.md) 记录 CL-CrossVQA（WACV 2025）、CoIN（NeurIPS 2024）、SMoLoRA（ICCV 2025）、MMMU（CVPR 2024）、AdaptLLM（ICLR 2024）等实际使用的数据、实验位置和用途。TextVQA 被 CoIN／SMoLoRA 连续训练使用，AQUA 视觉子集被 CL-CrossVQA 使用，RSVQA-LR 被 GeoChat／LHRS-Bot 评测使用。原始提出者与后续使用者分别记载，分类式 VQA、纯文本任务与生成式多模态 CL 不混称。旧溯源资料见 [来源与引用](data/DATASET_PROVENANCE.md)，本轮范围以上表与最新审计为准。','',
'## 原始问答与图像示例','',
'以下均为源数据实际记录，不是模型生成结果；长文本在表格中仅展示节选，完整问题、所有参考、上下文和源 ID 见 [完整样例页]('+BASE+'/samples.md)。图像为原始字节复制，页面仅缩放展示。样例用于说明数据内容，不声称已经选入本轮 1% 清单。尚未到达的原图明确留空，避免用其他图片替代。','',
'| 数据集／任务 | 原始问题 | 原始参考答案 | 原始图像／输入 |','|---|---|---|---|']
samplepage=['# 本轮原始数据样例','',f'来源审计时间：{audit["audited_at"]}。只作展示，不构成冻结训练清单。源 ID、问题和答案保留原值。','']
for name,task,*_ in catalog:
    x=rows[name];ex=x['example'];slug=name.lower().replace('_','-')
    img=x['sample_image']
    visual=f'<img src="{img}" width="180" alt="{name} 原始输入">' if img else ('原图尚未就绪' if name=='RSVQA-LR' else '文本；完整上下文见样例页')
    text.append(f'| [{name} · {task}]({BASE}/samples.md#{slug}) | {esc(ex["question"],180)} | {esc(ex["answer"],180)} | {visual} |')
    samplepage += [f'<a id="{slug}"></a>',f'## {name} · {task}','',f'原始 ID：`{ex["id"]}`；来源角色：`{ex.get("source_split","见审计 JSON")}`。','',
                   '**原始问题**','',ex['question'],'','**全部参考答案**','','```json',json.dumps(ex.get('answers',[ex['answer']]),ensure_ascii=False,indent=2),'```','']
    if ex.get('context'):samplepage += ['<details><summary>完整输入上下文</summary>','',ex['context'],'','</details>','']
    if img:samplepage += [f'![{name} 原始输入](../../../{img})','']
    if name=='AQUA':
        ex=x['text_example'];samplepage += ['### T9 文本视图补充示例','',f'源 ID：`{ex["id"]}`。','',ex['question'],'',ex['answer'],'']
text += ['',
'## 训练记录、结果与入口','',
'每个方法训练前及每完成 T1–T10 的一个阶段，都完整评测十个任务，得到 **11×10** 矩阵。保留逐题预测／参考、每任务和每来源的 EM 正确数／总数、Token F1、ROUGE-L，最后计算 AP、ForgetRate、FWT、BWT。上述统一词面指标不冒称各基准的原生语义、定位或数值指标。','',
'全量和 1% 组均记录每步 loss、裁剪前后梯度范数、学习率、耗时；每 5 秒和阶段边界记录 CPU RSS／PSS／USS 与 GPU allocated／reserved／进程占用。`memory_summary.json` 单列**训练前、最后阶段训练后释放优化器与梯度后的具体内存／显存、差值与峰值**，区分临时开销和跨任务持久增长。','',
f'- [实验说明](exp/{ID}/README.md) · [执行策略](exp/{ID}/execution_policy.json) · [新协议](rules/013_native_full_and_one_percent.md)',
f'- [当前进度](summary/{ID}/progress.md)：新日志与逐题结果在 `exp/{ID}/runs/`，新报告在 `summary/{ID}/`。',
'- [论文](paper/README.md) 与 [作者源码](code/README.md)；`data/` 存原始资产和划分，`model/` 存基座。',
'- 历史代码及结果统一存于 `exp/ago/`、`summary/ago/`。旧版小数据样例仍是历史资料，不作为本轮数据清单。','',
'论文表更新器只更新“方法 × 方向二维表”段，不覆盖上述实验与数据集说明：','',
'```bash','python3 paper/update_repository_readme.py','```','']
readme=ROOT/'README.md'
prefix=readme.read_text(encoding='utf-8').split('## 本轮实验',1)[0]
intro='本仓库整理生成式大语言模型／多模态大模型的持续学习论文、作者源码和数据依据，并开展五领域十任务的轻量化持续微调实验。本轮比较 **全量训练与 1% 训练，两组均完整测试**；基座 Qwen2-VL-2B-Instruct，seed42，无验证集，不保存 checkpoint。'
parts=prefix.split('\n');parts[2]=intro;prefix='\n'.join(parts)
start=prefix.index('## 当前服务器');end=prefix.index('## 方法 × 方向二维表')
prefix=prefix[:start]+'## 当前服务器\n\n主仓库位于新服务器 `/data1/liyapeng/program/LLMCL`，Tailscale 节点 `100.87.141.22`，SSH 端口 `8666`。数据资产准备进度见下文；旧的 100／50 实验已停止并归档。\n\n'+prefix[end:]
readme.write_text(prefix+'\n'.join(text),encoding='utf-8')
(ROOT/BASE/'samples.md').write_text('\n'.join(samplepage),encoding='utf-8')
print('Updated root README and samples; QA totals:',totals)
