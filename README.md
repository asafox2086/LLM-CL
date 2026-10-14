# LLMCL：小模型、小数据的多模态持续学习

本仓库整理生成式大语言模型／多模态大模型的持续学习论文、作者源码和数据依据，并开展五领域十任务的轻量化持续微调实验。当前协议为 **Qwen2-VL-2B-Instruct、每任务训练 100 条／测试 50 条、无验证集、seed 42、每任务训练 1 轮、不保存 checkpoint**。

## 当前服务器与迁移状态

主仓库已迁移至 `/data1/liyapeng/program/LLMCL`，通过 Tailscale `liyapeng@100.87.141.22:8666` 工作。资源内容与环境检查已通过；约 177 GiB 的旧重复仓库已清理。未整包迁移的 GeoChat 原始数据及维持旧端连接的 Tailscale 环境暂时保留。详见 [迁移与清理报告](summary/ten_task_cl_20261012/migration_report.md)。

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

## 本轮实验

新候选资源已下载：GeoQuestions1089 官方 1089 条问题／答案、GeoChat 150 张遥感图像及原始问答、FinChart-Bench 150 张图表及原始 QA。正式任务子集与论文使用口径仍在核验，详见 [下载进度](data/DOWNLOAD_PROGRESS_20261014.md) 和 [真实候选样例](data/DATASET_CANDIDATE_SAMPLES_20261014.md)。

**任务修订中，训练暂不启动。** 用户要求图像任务体现真正视觉理解：T5/T6 法学整体允许替换，优先核验地理／遥感；T8 金融合同／报表页问答需重新筛选图表任务。艺术文化保留，其他领域及 100／50、seed 42、无验证集预算不变。下表相关旧数据已明确标记，尚未确定替代数据时不冒充最终方案。

| 任务 | 领域 | 模态 | 数据集 | 训练／测试 |
|---|---|---|---|---:|
| T1 | 自然／通用 | 文本 | SQuAD 1.1 | 100／50 |
| T2 | 自然／通用 | 图像 | VQAv2、GQA、TextVQA | 100／50 |
| T3 | 医学 | 文本 | MedQuAD | 100／50 |
| T4 | 医学 | 图像 | VQA-RAD、PathVQA、SLAKE | 100／50 |
| T5 | 待定，优先地理 | 文本 | 待核验替代数据；旧 CUAD 停用 | 100／50 |
| T6 | 待定，优先遥感 | 图像 | 待核验；旧 CUAD 合同页停用 | 100／50 |
| T7 | 金融 | 文本 | FinQA | 100／50 |
| T8 | 金融 | 图像 | 重新筛选真正图表问答；旧 TAT-DQA 停用 | 100／50 |
| T9 | 艺术文化 | 文本 | AQUA 知识问答子集 | 100／50 |
| T10 | 艺术文化 | 图像 | AQUA 视觉问答子集 | 100／50 |

均使用原始问题、答案和标注；图像任务使用真实图像／原始文档页，不将分类标签包装成开放问答。CUAD 图像视图和 AQUA 文本视图的适配口径、采样及检索细节见 [固定协议](rules/012_small_data_results_only.md) 与 [数据溯源](data/DATASET_PROVENANCE.md)。不同方法共享相同的训练／测试样本 ID。

每个方法从同一基座开始，先评测全部十任务，再按 **T1→T10** 连续训练；每训练完一个任务，评测全部十任务，各 50 条，形成 **11×10** 测试矩阵。验证集已取消，不进行验证推理或最佳 checkpoint 选择。阶段间模型和方法必需的历史参考留在内存中；进程中断需从基座重跑，旧预测保留但不与新运行混算。

保存逐题预测与参考答案、训练日志、每任务及每源数据集的 EM 正确数／总数、Token F1、ROUGE-L。完整运行后分别计算 **AP、ForgetRate、FWT、BWT**，输出热图和性能曲线（PNG／PDF／绘图数据）。统一 EM 与词面分数不冒称各数据集的官方指标。

计划比较二维表中的 **SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER** 八种非回放论文方法。不运行 SeqLoRA。必须保留作者核心机制，并公开 Qwen2-VL 与小数据适配差异；下载源码不等于完成适配或开始实验。

当前状态以 [对比进度](summary/ten_task_cl_20261012/comparison_report.md) 为准；方法依据和固定源码版本见 [code/README.md](code/README.md)。多卡按空闲状态分配独立方法。

## 本轮数据集 sample 示例

每个任务取固定训练清单的第一条，展示真实问题和参考答案；这是数据样例，不是模型回答。表中长文本仅作展示节选，原始问题／答案未修改。点击任务编号查看完整问题、所有参考答案、上下文及图像。

| 任务／来源 | 问题（原文节选） | 参考答案（原文节选） | 图像输入 |
|---|---|---|---|
| [T01](data/TenTaskCL/samples.md#t01) · SQuAD11 | When was the Canada First Defence Strategy introduced? | 2008 | 文本输入，完整上下文见样例页 |
| [T02](data/TenTaskCL/samples.md#t02) · GQA | What animal is on the table? | kitten | <img src="sample/ten_task_cl_20261012/T02_image_1.jpg" width="180" alt="T02 原始输入图像"> |
| [T03](data/TenTaskCL/samples.md#t03) · MedQuAD | What are the symptoms of Hairy nose tip ? | What are the signs and symptoms of Hairy nose tip? The Human Phenotype Ontology provides the following list of signs and symptoms for Hairy …（节选） | 文本输入，完整上下文见样例页 |
| [T04](data/TenTaskCL/samples.md#t04) · PathVQA | what appears to have changed into a blast crisis? | total effacement case | <img src="sample/ten_task_cl_20261012/T04_image_1.jpg" width="180" alt="T04 原始输入图像"> |
| T05 | 数据集修订中 | 暂不展示已停用的旧样例 | 等待核验新样例 |
| T06 | 数据集修订中 | 暂不展示已停用的旧样例 | 等待核验新样例 |
| [T07](data/TenTaskCL/samples.md#t07) · FinQA | what was the percentage change in pro forma revenue from 2001 to 2002? | 44% | 文本输入，完整上下文见样例页 |
| T08 | 数据集修订中 | 暂不展示已停用的旧样例 | 等待核验新样例 |
| [T09](data/TenTaskCL/samples.md#t09) · AQUA | what are part of a polyptych executed for the church of the convent of san francesco at borgo san sepolcro | these panels | 文本输入，完整上下文见样例页 |
| [T10](data/TenTaskCL/samples.md#t10) · AQUA | what is the location of the picture | street | <img src="sample/ten_task_cl_20261012/T10_image_1.jpg" width="180" alt="T10 原始输入图像"> |

## 目录与入口

```text
LLMCL/
├── README.md                         # 当前实验与论文二维表
├── paper/                            # 论文、PDF、发表核验
│   ├── pre_training/
│   ├── pre_finetuning/
│   │   ├── method_replay/
│   │   ├── method_regularization/
│   │   └── method_architecture/
│   └── pre_alignment/
├── code/                             # 各方法作者源码与适配依据
├── data/                             # 数据资产、来源依据与固定样本清单
├── model/                            # 固定模型资产
├── exp/
│   ├── ten_task_cl_20261012/          # 当前入口、运行日志和逐题结果
│   └── ago/                         # 历史实验与旧脚本
├── summary/
│   ├── ten_task_cl_20261012/          # 当前汇总、指标和图
│   └── ago/                         # 历史报告（含旧 CV 与医学实验）
└── rules/                            # 实验协议与调整记录
```

[当前运行说明](exp/ten_task_cl_20261012/README.md) · [执行策略](exp/ten_task_cl_20261012/execution_policy.json) · [固定样本 ID 与校验值](data/TenTaskCL/current_small_manifest.json) · [历史报告](summary/ago/README.md) · [历史实验](exp/ago/README.md)

服务器仓库迁移目标：`/data1/liyapeng/program/LLMCL`。确认没有现有队列后启动：

```bash
.vision-env/bin/python exp/ten_task_cl_20261012/comparison_queue.py
```

论文表依据已核验的清单生成；只更新根 README、无需重新下载：

```bash
python3 paper/update_repository_readme.py
```

根目录不再保留旧医学报告的兼容符号链接。旧实验保存在 `exp/ago/`，旧报告保存在 `summary/ago/`；历史实验的模型、配额和结论与本轮分别解读。`sample/` 只展示本轮十任务数据；旧示例资产归档至 `summary/ago/sample_assets/`。`CV_data/`、`source_code/` 保留历史数据与源码。
