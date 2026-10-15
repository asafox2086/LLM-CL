# LLMCL：小模型、小数据的多模态持续学习

本仓库整理生成式大语言模型／多模态大模型的持续学习论文、作者源码和数据依据，并开展五领域十任务的轻量化持续微调实验。本轮比较 **全量训练与 1% 训练，两组均完整测试**；基座 Qwen2-VL-2B-Instruct，seed42，无验证集，不保存 checkpoint。

## 当前服务器

主仓库位于新服务器 `/data1/liyapeng/program/LLMCL`，Tailscale 节点 `100.87.141.22`，SSH 端口 `8666`。数据资产准备进度见下文；旧的 100／50 实验已停止并归档。

## 方法 × 方向二维表

已整理 **29 篇** 2024–2026 年正式会议论文：持续预训练 6 篇、持续微调 20 篇、持续对齐 3 篇。各方向列直接列出方法名，持续微调细分回放、正则化和架构；列内优先较新论文，同年优先 CVPR／ICCV。横向同一行的方法之间没有对应关系。点击方法名进入正式论文页，点击 PDF 查看仓库文件。年份按正式会议届次；逐篇核验记录见 [paper/verification_audit.md](paper/verification_audit.md)。

<table>
<thead>
<tr><th rowspan="2">持续预训练</th><th colspan="3">持续微调</th><th rowspan="2">持续对齐</th></tr>
<tr><th>基于回放</th><th>基于正则化</th><th>基于架构</th></tr>
</thead>
<tbody>
<tr><td><a href="https://aclanthology.org/2026.acl-long.427/" title="Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training"><strong>Data Mixing Agent</strong></a><br>ACL 2026<br><a href="paper/pre_training/2026_ACL_Data_Mixing_Agent.pdf">PDF</a></td><td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Gao_Re-evaluating_Continual_VQA_Toward_Fair_and_Robust_Evaluation_for_Multimodal_CVPR_2026_paper.html" title="Re-evaluating Continual VQA: Toward Fair and Robust Evaluation for Multimodal Continual Learning"><strong>MaDQ / UCo-VQA</strong></a><br>CVPR 2026<br><a href="paper/pre_finetuning/method_replay/2026_CVPR_MaDQ_UCo_VQA.pdf">PDF</a></td><td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Li_Multimodal_Continual_Instruction_Tuning_with_Dynamic_Gradient_Guidance_CVPR_2026_paper.html" title="Multimodal Continual Instruction Tuning with Dynamic Gradient Guidance"><strong>DGG</strong></a><br>CVPR 2026<br><a href="paper/pre_finetuning/method_regularization/2026_CVPR_DGG.pdf">PDF</a></td><td><a href="https://aclanthology.org/2026.acl-long.1474/" title="PASs-MoE: Mitigating Misaligned Co-drift among Router and Experts via Pathway Activation Subspaces for Continual Learning"><strong>PASs-MoE</strong></a><br>ACL 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ACL_PASs_MoE.pdf">PDF</a></td><td><a href="https://openaccess.thecvf.com/content/CVPR2026/html/Wang_Harmonious_Parameter_Adaptation_in_Continual_Visual_Instruction_Tuning_for_Safety-Aligned_CVPR_2026_paper.html" title="Harmonious Parameter Adaptation in Continual Visual Instruction Tuning for Safety-Aligned MLLMs"><strong>HPA</strong></a><br>CVPR 2026<br><a href="paper/pre_alignment/2026_CVPR_HPA.pdf">PDF</a></td></tr>
<tr><td><a href="https://aclanthology.org/2026.acl-long.1722/" title="Glyph: Scaling Context Windows via Visual-Text Compression"><strong>Glyph</strong></a><br>ACL 2026<br><a href="paper/pre_training/2026_ACL_Glyph.pdf">PDF</a></td><td><a href="https://aclanthology.org/2026.acl-long.1144/" title="FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning"><strong>FOREVER</strong></a><br>ACL 2026<br><a href="paper/pre_finetuning/method_replay/2026_ACL_FOREVER.pdf">PDF</a></td><td><a href="https://aclanthology.org/2026.acl-long.1244/" title="Harmonizing the Past, Present, and Future: A Null-Space Constrained Region-Specific Method for Continual Learning in LLMs"><strong>PaRSP</strong></a><br>ACL 2026<br><a href="paper/pre_finetuning/method_regularization/2026_ACL_PaRSP.pdf">PDF</a></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/0dedca10992d34035c9a9776254cfa8c-Abstract-Conference.html" title="Merge before Forget: A Single LoRA Continual Learning via Continual Merging"><strong>SLAO / Merge before Forget</strong></a><br>ICLR 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ICLR_SLAO_Merge_before_Forget.pdf">PDF</a></td><td><a href="https://ojs.aaai.org/index.php/AAAI/article/view/40428" title="LifeAlign: Lifelong Alignment for Large Language Models with Memory-Augmented Focalized Preference Optimization"><strong>LifeAlign</strong></a><br>AAAI 2026<br><a href="paper/pre_alignment/2026_AAAI_LifeAlign.pdf">PDF</a>（作者扩展版）</td></tr>
<tr><td><a href="https://aclanthology.org/2026.acl-long.1866/" title="From Curated Data to Scalable Models: Continual Pre-training of Dense and MoE Large Language Models for Tibetan"><strong>Tibetan Dense / MoE CPT</strong></a><br>ACL 2026<br><a href="paper/pre_training/2026_ACL_Tibetan_Dense_MoE_CPT.pdf">PDF</a></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/17fc58b1ba92abbb72c5f457e140d851-Abstract-Conference.html" title="Forget Forgetting: Continual Learning in a World of Abundant Memory"><strong>Weight Space Consolidation</strong></a><br>ICLR 2026<br><a href="paper/pre_finetuning/method_replay/2026_ICLR_Weight_Space_Consolidation.pdf">PDF</a></td><td><a href="https://aclanthology.org/2026.acl-long.334/" title="Spectral Disentanglement: Rank-Aware Task Adaptation for Rehearsal-free Continual Learning in LLMs"><strong>SpaRTA</strong></a><br>ACL 2026<br><a href="paper/pre_finetuning/method_regularization/2026_ACL_SpaRTA.pdf">PDF</a></td><td><a href="https://proceedings.mlr.press/v306/wang26le.html" title="CoPE: Continual Probe-guided Expansion for Large Vision-Language Models"><strong>CoPE</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ICML_CoPE.pdf">PDF</a></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2024/hash/6246f93ebf4f2d76ad2bcb3158220d34-Abstract-Conference.html" title="CPPO: Continual Learning for Reinforcement Learning with Human Feedback"><strong>CPPO</strong></a><br>ICLR 2024<br><a href="paper/pre_alignment/2024_ICLR_CPPO.pdf">PDF</a></td></tr>
<tr><td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/528f0a943a54903939e77509f874d8e8-Abstract-Conference.html" title="Cite Pretrain: Retrieval-Free Knowledge Attribution for Large Language Models"><strong>Cite Pretrain / Active Indexing</strong></a><br>ICLR 2026<br><a href="paper/pre_training/2026_ICLR_Cite_Pretrain_Active_Indexing.pdf">PDF</a></td><td><a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/087b346ca4d6dd6fcd0663ff6c9a6b69-Abstract-Conference.html" title="Self-Evolving Pseudo-Rehearsal for Catastrophic Forgetting with Task Similarity in LLMs"><strong>SERS</strong></a><br>NeurIPS 2025<br><a href="paper/pre_finetuning/method_replay/2025_NeurIPS_SERS.pdf">PDF</a></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html" title="Sculpting Subspaces: Constrained Full Fine-Tuning in LLMs for Continual Learning"><strong>OSFT</strong></a><br>ICLR 2026<br><a href="paper/pre_finetuning/method_regularization/2026_ICLR_OSFT.pdf">PDF</a></td><td><a href="https://proceedings.mlr.press/v306/kong26h.html" title="ECA: Efficient Continual Alignment for Open-Ended Image-to-Text Generation"><strong>ECA</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ICML_ECA.pdf">PDF</a></td><td></td></tr>
<tr><td><a href="https://proceedings.mlr.press/v306/padmanabhan26a.html" title="Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities"><strong>DiSC</strong></a><br>ICML 2026<br><a href="paper/pre_training/2026_ICML_DiSC.pdf">PDF</a></td><td></td><td><a href="https://proceedings.mlr.press/v306/xia26j.html" title="Adversarial Latent Embedding Repair for LLM Continual Learning"><strong>ALER</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_regularization/2026_ICML_ALER.pdf">PDF</a></td><td><a href="https://proceedings.mlr.press/v306/xu26cj.html" title="Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tuning"><strong>Hyper-LLaVA</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ICML_Hyper_LLaVA.pdf">PDF</a></td><td></td></tr>
<tr><td><a href="https://proceedings.mlr.press/v306/borodin26a.html" title="Diffract: Spectral View of LLM Domain Adaptation"><strong>Diffract</strong></a><br>ICML 2026<br><a href="paper/pre_training/2026_ICML_Diffract.pdf">PDF</a></td><td></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html" title="Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning"><strong>FV-guided Training</strong></a><br>ICLR 2025<br><a href="paper/pre_finetuning/method_regularization/2025_ICLR_FV_guided_Training.pdf">PDF</a></td><td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html" title="CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering"><strong>CL-MoE</strong></a><br>CVPR 2025<br><a href="paper/pre_finetuning/method_architecture/2025_CVPR_CL_MoE.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html" title="Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model"><strong>Any-SSR</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_Any_SSR.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html" title="Federated Continual Instruction Tuning"><strong>FCIT</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_FCIT.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html" title="Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models"><strong>MVP / Instruction-Grounded Visual Projectors</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_MVP_Instruction_Grounded_Visual_Projectors.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html" title="SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning"><strong>SMoLoRA</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_SMoLoRA.pdf">PDF</a></td><td></td></tr>
</tbody>
</table>

归类按主要机制，混合机制、原论文模型及生成任务见 [论文完整索引](paper/README.md)。预训练条目包含语言／知识／上下文扩展型 CPT，不全部具有序贯抗遗忘实验；HPA 归入持续安全对齐保持。**回放方法仍保留在论文表中，本轮不做回放实验；DGG 含历史样本回放，也不参与本轮。**

## 本轮实验与数据状态

当前实验编号为 **`ten_task_cl_20261015_full_1pct`**。旧的 100 训练／50 测试实验已停止，代码及结果归档到 [exp/ago](exp/ago/ten_task_cl_20261014_geo_fin/README.md) 与 [summary/ago](summary/ago/ten_task_cl_20261014_geo_fin/comparison_report.md)，不与本轮混算。

基座 **Qwen2-VL-2B-Instruct**；seed **42**；每任务 **1 轮**；无验证集；不保存模型、优化器或恢复 checkpoint。五领域按自然／通用→医学→地理／遥感→金融→艺术文化排列，每领域先文本后图像，共 **10 个任务**。

| 条件 | 训练池 | 测试池 | GPU |
|---|---|---|---|
| full | 所选公开发布配置的全部训练 QA | 全部公开有答案的评测 QA | GPU3，方法串行 |
| pct01 | 同一训练池按来源和原始题型分层取 1% | 与 full 完全相同的测试 QA | GPU1／2，并行方法 |

**GPU0 保留。** 两组均比较 SeqFT-LoRA 朴素顺序微调控制，以及 SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。控制基线属于标准顺序微调，不作为新算法。

**当前是数据资产准备阶段，尚未开始本轮正式训练。** 下表已从服务器原始 JSON／XML／Parquet 核对元数据数量；这不等于全部图像已齐，也不是已经冻结的最终运行清单。LLaVA-Med 问答标注已到达，但历史 PMC 原图链接出现 404，图像恢复待解决；GeoChat 原始分卷已到达，完整图像提取／校验及 AQUA 的 SemArt 原图准备仍待完成。实际状态见 [队列进度](summary/ten_task_cl_20261015_full_1pct/progress.md)。

## 数据集：来源、数量与划分

核验时间：`2026-10-15T23:50:25.762344+08:00`。**数量单位统一为 QA 对**，不是图像数，也不是多轮对话数。“全量”限定在表中明确的发布配置；原始分类、是非、选择、定位和开放题均保留，不再人工剔除题型。完整机器记录及原始样例见 [数据审计 JSON](data/TenTaskCL/ten_task_cl_20261015_full_1pct/readme_dataset_audit.json)。

| 数据集 | 任务／模态 | 提出者与发表来源 | 下载／原始资源 | 全量训练 QA | 1% 训练 QA | 全量测试 QA | 配置与划分 |
|---|---|---|---|---:|---:|---:|---|
| SQuAD11 | T1 · 文本 | [Pranav Rajpurkar 等 · EMNLP 2016](https://aclanthology.org/D16-1264/) | [资源](https://rajpurkar.github.io/SQuAD-explorer/) | 87,599 | 876 | 10,570 | 官方 train / dev；段落阅读问答 |
| VQAv2 | T2 · **多模态** | [Yash Goyal 等 · CVPR 2017](https://openaccess.thecvf.com/content_cvpr_2017/html/Goyal_Making_the_v_CVPR_2017_paper.html) | [资源](https://huggingface.co/datasets/lmms-lab/VQAv2) | 443,757 | 4,438 | 214,354 | 官方 train / validation 的格式镜像；含 yes/no、数字与其他答案 |
| GQA | T2 · **多模态** | [Drew A. Hudson、Christopher D. Manning · CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Hudson_GQA_A_New_Dataset_for_Real-World_Visual_Reasoning_and_Compositional_CVPR_2019_paper.html) | [资源](https://huggingface.co/datasets/lmms-lab/GQA) | 943,000 | 9,430 | 12,578 | 官方 balanced 配置；train / 有答案的 testdev |
| TextVQA | T2 · **多模态** | [Amanpreet Singh 等 · CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Singh_Towards_VQA_Models_That_Can_Read_CVPR_2019_paper.html) | [资源](https://huggingface.co/datasets/lmms-lab/textvqa) | 34,602 | 347 | 5,000 | 官方 train / validation 的格式镜像；真实图像中的文字问答 |
| MedQuAD | T3 · 文本 | [Asma Ben Abacha、Dina Demner-Fushman · BMC Bioinformatics 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6805558/) | [资源](https://github.com/abachaa/MedQuAD) | 13,037 | 131 | 3,370 | 公开有答案的 XML；本研究按原始文档分组划分 |
| VQA_RAD | T4 · **多模态** | [Jason J. Lau 等 · Scientific Data 2018](https://www.nature.com/articles/sdata2018251) | [资源](https://huggingface.co/datasets/flaviagiammarino/vqa-rad) | 1,793 | 18 | 451 | 所下载镜像的 train / test；不是声称原论文所有记录，保留 yes/no |
| SLAKE | T4 · **多模态** | [Bo Liu 等 · ISBI 2021](https://research.polyu.edu.hk/en/publications/slake-a-semantically-labeled-knowledge-enhanced-dataset-for-medic/) | [资源](https://www.med-vqa.com/slake/) | 9,835 | 99 | 2,094 | 作者 train / test；中英双语、OPEN/CLOSED 均保留 |
| LLaVA-Med | T4 · **多模态** | [Chunyuan Li、Cliff Wong、Sheng Zhang 等 · NeurIPS 2023 Datasets and Benchmarks](https://proceedings.neurips.cc/paper_files/paper/2023/file/5abcdf8ecdcacba028c6662789194572-Paper-Datasets_and_Benchmarks.pdf) | [资源](https://github.com/microsoft/LLaVA-Med) | 162,640 | 1,627 | 193 | 作者 instruct60k / 独立 qa50；替换 PathVQA；含作者 GPT-4 标注 |
| GeoQuestions1089 | T5 · 文本 | [Sergios-Anestis Kefalidis 等 · ISWC 2023](https://iswc2023.semanticweb.org/wp-content/uploads/2023/11/142660264.pdf) | [资源](https://github.com/AI-team-UoA/GeoQuestions1089) | 871 | 9 | 214 | 原始地理问答与结果表；本研究按查询组划分 |
| GeoChat | T6 · **多模态** | [Kartik Kuckreja 等 · CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Kuckreja_GeoChat_Grounded_Large_Vision-Language_Model_for_Remote_Sensing_CVPR_2024_paper.html) | [资源](https://github.com/mbzuai-oryx/GeoChat) | 547,873 | 5,479 | 137,978 | 原始 GeoChat_Instruct 全部对话题型；本研究按图像划分 |
| FinQA | T7 · 文本＋原表格 | [Zhiyu Chen 等 · EMNLP 2021](https://aclanthology.org/2021.emnlp-main.300/) | [资源](https://github.com/czyssrs/FinQA) | 6,251 | 63 | 1,147 | 作者 train / test；表格数值推理，保留 yes/no |
| MME-Finance | T8 · **多模态** | [Ziliang Gan 等 · ACM MM 2025](https://doi.org/10.1145/3746027.3758230) | [资源](https://hithink-research.github.io/MME-Finance/) | 935 | 10 | 236 | 原始英文 MMfin.tsv 全部 1,171 QA；本研究按底层图表组划分 |
| AQUA | T9／T10 · 文本／**多模态** | [Noa Garcia 等 · ECCV VISART Workshop 2020](https://github.com/noagarcia/ArtVQA) | [资源](https://github.com/noagarcia/ArtVQA/tree/master/AQUA) | 69,812 | 699 | 4,912 | 作者 train / test；按原标记区分知识、视觉；同一数据集只列一次 |

**原始规模与使用范围需要区分：**

- **LLaVA-Med**：文件名为 instruct60k，当前作者文件实有 **56,702 段训练对话／图像**，展开为 **162,640 对 QA**；独立评测为 **50 图像、193 QA**。训练和评测参考含作者 GPT-4 标注，不能称全人工临床金标准。原图未补齐，不能称该项训练就绪。
- **GeoChat**：原始文件含 **308,861 段对话**，展开为 **685,851 对 QA**；不再只选旧版开放问答或计数题。多轮历史只作为该轮原始上下文，不把整段对话误算成一条 QA。
- **MedQuAD**：原始集合 **47,457 QA**；当前公开源有 **16,407 条非空参考答案**，另 **31,034** 条公开答案缺失。缺失标签无法监督训练／评分，数量和原因单列，不补造答案，也不是按题型删除。
- **GeoQuestions1089**：原始 **1,089** 问，公开文件 **1,085** 条有答案，**4** 条为 null；本轮保留所有可评分题型，包括布尔结果、列表与原始 CSV，不只取数值题。
- **VQA-RAD**：本表是所下载第三方格式镜像的 **1,793／451** 划分，不能直接冒称原论文完整原始集合；来源已单独标明。
- **GQA**：选择作者 balanced 配置，不冒称全部生成问题；另有 val **132,062**、隐藏 test **95,336**，本轮使用有公开答案的 testdev **12,578**。SQuAD、VQAv2、TextVQA 同样明确使用公开 dev／validation 作完整评测池，不能冒称官方隐藏 test 成绩。
- **MME-Finance**：选择英文 MMfin.tsv **1,171 QA**，包含原始图表、表格、文档及其他图像类别；不再只选 PC K 线。它不等于论文所有语言／扩展版本总量。
- **AQUA**：官方训练 **69,812**、val **5,124**、测试 **4,912**；本轮不使用 val。作者知识标记对应 T9：训练 **40,244**、1% **403**、测试 **3,642**；视觉标记对应 T10：训练 **29,568**、1% **296**、测试 **1,270**。T9 使用问题文本视图，是本研究的模态设置，不是另一个原生纯文本数据集；缺失画作上下文可能影响可回答性。SLAKE 的官方 validation **2,099** 同样不用。

官方 train／test（或公开带答案的 dev／validation／testdev）优先；无官方划分时，本研究按原始文档／查询／图像／底层图表分组，以 `SHA256(42:group)` 排序，取前 `ceil(20% × 组数)` 作测试，其余训练。组大小不同，QA 比例不必恰好为 80/20。四项研究划分为 MedQuAD、GeoQuestions1089、GeoChat、MME-Finance，不能称作者官方划分。

1% 组在每个来源训练池取 `ceil(0.01 × N)`，按原始题型比例以最大余数分配名额，再按 `SHA256(42:QA_ID)` 固定选取；AQUA 两个任务分别取整。所有方法共享同一清单和训练顺序；两组完整测试文件的 SHA256 必须相同。图像缺失不会通过静默丢样本处理，资产核验通过后再冻结运行清单。

### 十任务数量汇总

| 任务 | 领域／模态 | 全量训练 QA | 1% 训练 QA | 两组共同完整测试 QA |
|---|---|---:|---:|---:|
| T1 | 自然／通用 · 文本 | 87,599 | 876 | 10,570 |
| T2 | 自然／通用 · 图像 | 1,421,359 | 14,215 | 231,932 |
| T3 | 医学 · 文本 | 13,037 | 131 | 3,370 |
| T4 | 医学 · 图像 | 174,268 | 1,744 | 2,738 |
| T5 | 地理／遥感 · 文本 | 871 | 9 | 214 |
| T6 | 地理／遥感 · 图像 | 547,873 | 5,479 | 137,978 |
| T7 | 金融 · 文本＋表格 | 6,251 | 63 | 1,147 |
| T8 | 金融 · 图像 | 935 | 10 | 236 |
| T9 | 艺术文化 · 文本 | 40,244 | 403 | 3,642 |
| T10 | 艺术文化 · 图像 | 29,568 | 296 | 1,270 |
| **合计** | **十任务** | **2,322,005** | **23,226** | **393,097** |

### 后续论文使用依据

既有来源与后续使用记录见 [数据溯源与引用](data/DATASET_PROVENANCE.md) 和 [BibTeX](data/dataset_references.bib)；其中旧候选状态和旧筛选描述属于历史记录，本轮范围以上表和本轮审计为准。例如 SQuAD 被 BERT／MobileBERT 使用，VQAv2／GQA／TextVQA 被 SMoLoRA 使用；AQUA 被 [ArtQuest（WACV 2024）](https://openaccess.thecvf.com/content/WACV2024/papers/Bleidt_ArtQuest_Countering_Hidden_Language_Biases_in_ArtVQA_WACV_2024_paper.pdf) 评测，GeoChat 被 [TEOChat（ICLR 2025）](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ac3af725ae398b6184faae0828bdbd6c-Abstract-Conference.html) 用于指令训练。新增 LLaVA-Med 的独立复用为 [STLLaVA-Med（NeurIPS 2024 Datasets and Benchmarks）](https://papers.neurips.cc/paper_files/paper/2024/file/aec33ab89b5986605cd7c331396e7e5c-Paper-Datasets_and_Benchmarks_Track.pdf)。论文使用这些数据不等于使用本研究完全相同的分组、模态或评价方式。

## 原始问答与图像示例

以下均为源数据实际记录，不是模型生成结果；长文本在表格中仅展示节选，完整问题、所有参考、上下文和源 ID 见 [完整样例页](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md)。图像为原始字节复制，页面仅缩放展示。样例用于说明数据内容，不声称已经选入本轮 1% 清单。尚无原图的 LLaVA-Med 明确留空，避免用其他图片替代。

| 数据集／任务 | 原始问题 | 原始参考答案 | 原始图像／输入 |
|---|---|---|---|
| [SQuAD11 · T1](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#squad11) | To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France? | Saint Bernadette Soubirous | 文本；完整上下文见样例页 |
| [VQAv2 · T2](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#vqav2) | What is this photo taken looking through? | net | <img src="sample/ten_task_cl_20261015_full_1pct/VQAv2.jpg" width="180" alt="VQAv2 原始输入"> |
| [GQA · T2](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#gqa) | What animal is on the table? | kitten | <img src="sample/ten_task_cl_20261015_full_1pct/GQA.jpg" width="180" alt="GQA 原始输入"> |
| [TextVQA · T2](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#textvqa) | what is the brand of phone? | nokia | <img src="sample/ten_task_cl_20261015_full_1pct/TextVQA.jpg" width="180" alt="TextVQA 原始输入"> |
| [MedQuAD · T3](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#medquad) | What is (are) Pemphigus vulgaris ? | Pemphigus vulgaris is an autoimmune disorder that involves blistering of the skin and mucous membranes. It occurs almost exclusively in middle-aged or older people. Many cases begi…（展示节选） | 文本；完整上下文见样例页 |
| [VQA_RAD · T4](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#vqa-rad) | are regions of the brain infarcted? | yes | <img src="sample/ten_task_cl_20261015_full_1pct/VQA_RAD.jpg" width="180" alt="VQA_RAD 原始输入"> |
| [SLAKE · T4](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#slake) | What modality is used to take this image? | MRI | <img src="sample/ten_task_cl_20261015_full_1pct/SLAKE.jpg" width="180" alt="SLAKE 原始输入"> |
| [LLaVA-Med · T4](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#llava-med) | What is the purpose of the flow diagram? | The purpose of the flow diagram is to illustrate the lung cancer screening process implemented by the local government of Yokohama City. It shows the steps involved in the screenin…（展示节选） | 原图尚未就绪 |
| [GeoQuestions1089 · T5](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#geoquestions1089) | Which are the non-metropolitan counties of England? | county<br>http://yago-knowledge.org/resource/Derbyshire<br>http://yago-knowledge.org/resource/Devon<br>http://yago-knowledge.org/resource/Warwickshire<br>http://yago-knowledge.org/resource…（展示节选） | 文本；完整上下文见样例页 |
| [GeoChat · T6](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#geochat) | Can you count the number of buildings and trees in the image? | The image contains a total of 1 building and 1 tree. | <img src="sample/ten_task_cl_20261015_full_1pct/GeoChat.png" width="180" alt="GeoChat 原始输入"> |
| [FinQA · T7](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#finqa) | what is the the interest expense in 2009? | 380 | 文本；完整上下文见样例页 |
| [MME-Finance · T8](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#mme-finance) | What content does this image display? | This image displays a table of financial stock information under the category &quot;&quot;Most Popular Stocks,&quot;&quot; specifically focusing on the &quot;&quot;24H Hot&quot;&quot; stocks. The table includes columns f…（展示节选） | <img src="sample/ten_task_cl_20261015_full_1pct/MME-Finance.jpg" width="180" alt="MME-Finance 原始输入"> |
| [AQUA · T9／T10](data/TenTaskCL/ten_task_cl_20261015_full_1pct/samples.md#aqua) | what is a large clock displayed in | building | <img src="sample/ten_task_cl_20261015_full_1pct/AQUA.jpg" width="180" alt="AQUA 原始输入"> |

## 训练记录、结果与入口

每个方法训练前及每完成 T1–T10 的一个阶段，都完整评测十个任务，得到 **11×10** 矩阵。保留逐题预测／参考、每任务和每来源的 EM 正确数／总数、Token F1、ROUGE-L，最后计算 AP、ForgetRate、FWT、BWT。上述统一词面指标不冒称各基准的原生语义、定位或数值指标。

全量和 1% 组均记录每步 loss、裁剪前后梯度范数、学习率、耗时；每 5 秒和阶段边界记录 CPU RSS／PSS／USS 与 GPU allocated／reserved／进程占用。`memory_summary.json` 单列**训练前、最后阶段训练后释放优化器与梯度后的具体内存／显存、差值与峰值**，区分临时开销和跨任务持久增长。

- [实验说明](exp/ten_task_cl_20261015_full_1pct/README.md) · [执行策略](exp/ten_task_cl_20261015_full_1pct/execution_policy.json) · [新协议](rules/013_native_full_and_one_percent.md)
- [当前进度](summary/ten_task_cl_20261015_full_1pct/progress.md)：新日志与逐题结果在 `exp/ten_task_cl_20261015_full_1pct/runs/`，新报告在 `summary/ten_task_cl_20261015_full_1pct/`。
- [论文](paper/README.md) 与 [作者源码](code/README.md)；`data/` 存原始资产和划分，`model/` 存基座。
- 历史代码及结果统一存于 `exp/ago/`、`summary/ago/`。旧版小数据样例仍是历史资料，不作为本轮数据清单。

论文表更新器只更新“方法 × 方向二维表”段，不覆盖上述实验与数据集说明：

```bash
python3 paper/update_repository_readme.py
```
