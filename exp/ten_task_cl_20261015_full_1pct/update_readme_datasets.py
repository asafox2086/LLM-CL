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
('VQAv2','T2','**多模态**','Yash Goyal 等 · CVPR 2017','https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html','https://huggingface.co/datasets/lmms-lab/VQAv2','官方 train / validation 的格式镜像；含 yes/no、数字与其他答案'),
('GQA','T2','**多模态**','Drew A. Hudson、Christopher D. Manning · CVPR 2019','https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html','https://huggingface.co/datasets/lmms-lab/GQA','官方 balanced 配置；train / 有答案的 testdev'),
('TextVQA','T2','**多模态**','Amanpreet Singh 等 · CVPR 2019','https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html','https://huggingface.co/datasets/lmms-lab/textvqa','官方 train / validation 的格式镜像；真实图像中的文字问答'),
('MedQuAD','T3','文本','Asma Ben Abacha、Dina Demner-Fushman · BMC Bioinformatics 2019','https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/','https://github.com/abachaa/MedQuAD','公开有答案的 XML；本研究按原始文档分组划分'),
('VQA_RAD','T4','**多模态**','Jason J. Lau 等 · Scientific Data 2018','https://www.nature.com/articles/sdata2018251','https://huggingface.co/datasets/flaviagiammarino/vqa-rad','所下载镜像的 train / test；不是声称原论文所有记录，保留 yes/no'),
('SLAKE','T4','**多模态**','Bo Liu 等 · ISBI 2021','https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/','https://www.med-vqa.com/slake/','作者 train / test；中英双语、OPEN/CLOSED 均保留'),
('LLaVA-Med','T4','**多模态**','Chunyuan Li、Cliff Wong、Sheng Zhang 等 · NeurIPS 2023 Datasets and Benchmarks','https://proceedings.neurips.cc/paper_files/paper/2023/file/5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf','https://github.com/microsoft/LLaVA-Med','作者 instruct60k / 独立 qa50；替换 PathVQA；含作者 GPT-4 标注'),
('GeoQuestions1089','T5','文本','Sergios-Anestis Kefalidis 等 · ISWC 2023','https://iswc2023.semanticweb.org/wp-content/uploads/2023/11/142660264.pdf','https://github.com/AI-team-UoA/GeoQuestions1089','原始地理问答与结果表；本研究按查询组划分'),
('GeoChat','T6','**多模态**','Kartik Kuckreja 等 · CVPR 2024','https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html','https://github.com/mbzuai-oryx/GeoChat','原始 GeoChat_Instruct 全部对话题型；本研究按图像划分'),
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
'**当前是数据资产准备阶段，尚未开始本轮正式训练。** 下表已从服务器原始 JSON／XML／Parquet 核对元数据数量；这不等于全部图像已齐，也不是已经冻结的最终运行清单。LLaVA-Med 问答标注已到达，但历史 PMC 原图链接出现 404，图像恢复待解决；GeoChat 原始分卷已到达，完整图像提取／校验及 AQUA 的 SemArt 原图准备仍待完成。实际状态见 [队列进度](summary/'+ID+'/progress.md)。','',
'## 数据集：来源、数量与划分','',
f'核验时间：`{audit["audited_at"]}`。**数量单位统一为 QA 对**，不是图像数，也不是多轮对话数。“全量”限定在表中明确的发布配置；原始分类、是非、选择、定位和开放题均保留，不再人工剔除题型。完整机器记录及原始样例见 [数据审计 JSON]('+BASE+'/readme_dataset_audit.json)。','',
'| 数据集 | 任务／模态 | 提出者与发表来源 | 下载／原始资源 | 全量训练 QA | 1% 训练 QA | 全量测试 QA | 配置与划分 |',
'|---|---|---|---|---:|---:|---:|---|']
for name,task,modality,origin,paper,download,scope in catalog:
    x=rows[name]
    text.append(f'| {name} | {task} · {modality} | [{origin}]({paper}) | [资源]({download}) | {x["full_train"]:,} | {x["pct01_train"]:,} | {x["full_test"]:,} | {scope} |')
text += ['',
'**原始规模与使用范围需要区分：**','',
'- **LLaVA-Med**：文件名为 instruct60k，当前作者文件实有 **56,702 段训练对话／图像**，展开为 **162,640 对 QA**；独立评测为 **50 图像、193 QA**。训练和评测参考含作者 GPT-4 标注，不能称全人工临床金标准。原图未补齐，不能称该项训练就绪。',
'- **GeoChat**：原始文件含 **308,861 段对话**，展开为 **685,851 对 QA**；不再只选旧版开放问答或计数题。多轮历史只作为该轮原始上下文，不把整段对话误算成一条 QA。',
'- **MedQuAD**：原始集合 **47,457 QA**；当前公开源有 **16,407 条非空参考答案**，另 **31,034** 条公开答案缺失。缺失标签无法监督训练／评分，数量和原因单列，不补造答案，也不是按题型删除。',
'- **GeoQuestions1089**：原始 **1,089** 问，公开文件 **1,085** 条有答案，**4** 条为 null；本轮保留所有可评分题型，包括布尔结果、列表与原始 CSV，不只取数值题。',
'- **VQA-RAD**：本表是所下载第三方格式镜像的 **1,793／451** 划分，不能直接冒称原论文完整原始集合；来源已单独标明。',
'- **GQA**：选择作者 balanced 配置，不冒称全部生成问题；另有 val **132,062**、隐藏 test **95,336**，本轮使用有公开答案的 testdev **12,578**。SQuAD、VQAv2、TextVQA 同样明确使用公开 dev／validation 作完整评测池，不能冒称官方隐藏 test 成绩。',
'- **MME-Finance**：选择英文 MMfin.tsv **1,171 QA**，包含原始图表、表格、文档及其他图像类别；不再只选 PC K 线。它不等于论文所有语言／扩展版本总量。',
'- **AQUA**：官方训练 **69,812**、val **5,124**、测试 **4,912**；本轮不使用 val。作者知识标记对应 T9：训练 **40,244**、1% **403**、测试 **3,642**；视觉标记对应 T10：训练 **29,568**、1% **296**、测试 **1,270**。T9 使用问题文本视图，是本研究的模态设置，不是另一个原生纯文本数据集；缺失画作上下文可能影响可回答性。SLAKE 的官方 validation **2,099** 同样不用。','',
'官方 train／test（或公开带答案的 dev／validation／testdev）优先；无官方划分时，本研究按原始文档／查询／图像／底层图表分组，以 `SHA256(42:group)` 排序，取前 `ceil(20% × 组数)` 作测试，其余训练。组大小不同，QA 比例不必恰好为 80/20。四项研究划分为 MedQuAD、GeoQuestions1089、GeoChat、MME-Finance，不能称作者官方划分。','',
'1% 组在每个来源训练池取 `ceil(0.01 × N)`，按原始题型比例以最大余数分配名额，再按 `SHA256(42:QA_ID)` 固定选取；AQUA 两个任务分别取整。所有方法共享同一清单和训练顺序；两组完整测试文件的 SHA256 必须相同。图像缺失不会通过静默丢样本处理，资产核验通过后再冻结运行清单。','',
'### 十任务数量汇总','',
'| 任务 | 领域／模态 | 全量训练 QA | 1% 训练 QA | 两组共同完整测试 QA |','|---|---|---:|---:|---:|']
tasksets=[(1,'自然／通用 · 文本',['SQuAD11']),(2,'自然／通用 · 图像',['VQAv2','GQA','TextVQA']),
          (3,'医学 · 文本',['MedQuAD']),(4,'医学 · 图像',['VQA_RAD','SLAKE','LLaVA-Med']),
          (5,'地理／遥感 · 文本',['GeoQuestions1089']),(6,'地理／遥感 · 图像',['GeoChat']),
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
'既有来源与后续使用记录见 [数据溯源与引用](data/DATASET_PROVENANCE.md) 和 [BibTeX](data/dataset_references.bib)；其中旧候选状态和旧筛选描述属于历史记录，本轮范围以上表和本轮审计为准。例如 SQuAD 被 BERT／MobileBERT 使用，VQAv2／GQA／TextVQA 被 SMoLoRA 使用；AQUA 被 [ArtQuest（WACV 2024）](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf) 评测，GeoChat 被 [TEOChat（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html) 用于指令训练。新增 LLaVA-Med 的独立复用为 [STLLaVA-Med（NeurIPS 2024 Datasets and Benchmarks）](https://papers.neurips.cc/paper_files/paper/2024/file/aec33ab89b5986605cd7c331396e7e5c-Paper-Datasets_and_Benchmarks_Track.pdf)。论文使用这些数据不等于使用本研究完全相同的分组、模态或评价方式。','',
'## 原始问答与图像示例','',
'以下均为源数据实际记录，不是模型生成结果；长文本在表格中仅展示节选，完整问题、所有参考、上下文和源 ID 见 [完整样例页]('+BASE+'/samples.md)。图像为原始字节复制，页面仅缩放展示。样例用于说明数据内容，不声称已经选入本轮 1% 清单。尚无原图的 LLaVA-Med 明确留空，避免用其他图片替代。','',
'| 数据集／任务 | 原始问题 | 原始参考答案 | 原始图像／输入 |','|---|---|---|---|']
samplepage=['# 本轮原始数据样例','',f'来源审计时间：{audit["audited_at"]}。只作展示，不构成冻结训练清单。源 ID、问题和答案保留原值。','']
for name,task,*_ in catalog:
    x=rows[name];ex=x['example'];slug=name.lower().replace('_','-')
    img=x['sample_image']
    visual=f'<img src="{img}" width="180" alt="{name} 原始输入">' if img else ('原图尚未就绪' if name=='LLaVA-Med' else '文本；完整上下文见样例页')
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
start=prefix.index('## 当前服务器与迁移状态');end=prefix.index('## 方法 × 方向二维表')
prefix=prefix[:start]+'## 当前服务器\n\n主仓库位于新服务器 `/data1/liyapeng/program/LLMCL`，Tailscale 节点 `100.87.141.22`，SSH 端口 `8666`。数据资产准备进度见下文；旧的 100／50 实验已停止并归档。\n\n'+prefix[end:]
readme.write_text(prefix+'\n'.join(text),encoding='utf-8')
(ROOT/BASE/'samples.md').write_text('\n'.join(samplepage),encoding='utf-8')
print('Updated root README and samples; QA totals:',totals)
