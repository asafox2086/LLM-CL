# LLMCL：小模型的六领域多模态持续学习

**六个领域、六个图文问答任务、每领域一个数据集**。金融替换为FinMME；第六领域新增交通／自动驾驶CODA-LM。

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

## 当前六任务协议

独立实验：`six_domain_cl_20261016_finmme_coda`；代码／参数在`exp/six_domain_cl_20261016_finmme_coda/`，日志报告在`summary/six_domain_cl_20261016_finmme_coda/`，QA清单在`data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/`（父目录名保留，内容是六任务）。旧五／十任务服务停用，不复用其预测。
当前状态：六任务清单已生成；GPU预检和正式运行以进度报告为准。
基座Qwen2-VL-2B-Instruct；seed42；一轮；无验证。九方法：SeqFT-LoRA、SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。GPU0保留；GPU3跑全量；GPU1/2并行跑1%组，均完整测试。
每个方法独立从基座开始，按T1→T6训练；初始及每阶段结束测全部六任务，输出**7×6矩阵**。保留逐题结果、EM正确数/总数、Token F1、ROUGE-L及AP、ForgetRate、FWT、BWT。统一词面指标不等于原生评分：FinMME涉及数值容差和多选；CODA-LM原文采用LLM评审，EM不是它的官方准确率。

## 数据集：来源、数量与复用依据

单位均为QA，每个来源训练QA低于十万。全部来源已按完整文件计数并冻结清单。保留原始选择、是非、计数等构成。

| 任务／领域 | 数据集（图文） | 提出者／首发 | 全量训练 | 1%训练 | 完整测试 | 划分 | 独立实际使用 |
|---|---|---|---:|---:|---:|---|---|
| T1 · 自然／通用 | TextVQA | [Amanpreet Singh等，CVPR 2019](https://textvqa.org/) | 34,602 | 347 | 5,000 | 官方train/validation；公开validation作测试 | CoIN（NeurIPS 2024）、SMoLoRA（ICCV 2025） |
| T2 · 医学 | SLAKE | [Bo Liu等，ISBI 2021](https://www.med-vqa.com/slake/) | 9,835 | 99 | 2,094 | 官方train/test；双语；val2099不用 | Uncertainty-Driven Expert Control（ICCV 2025） |
| T3 · 地理／遥感 | RSVQA-LR | [Sylvain Lobry等，TGRS 2020](https://rsvqa.sylvainlobry.com/) | 57,223 | 573 | 10,004 | 官方active掩码train/test；val10005不用；不自行分箱 | GeoChat（CVPR 2024）、LHRS-Bot（ECCV 2024）、RSAdapter（TGRS 2024） |
| T4 · 金融 | FinMME | [Junyu Luo等，ACL 2025主会](https://aclanthology.org/2025.acl-long.1426/) | 8,864 | 89 | 2,233 | 源池11099；2条缺答案单列；11097有标注QA按原图SHA256分组seed42研究80/20，非官方划分 | Beyond Real Weights（WACV 2026）实际评测；AgentFinVQA（2026预印本）实际评测 |
| T5 · 艺术／文化 | AQUA | [Noa Garcia等，ECCV VISART Workshop 2020](https://github.com/noagarcia/ArtVQA) | 69,812 | 699 | 4,912 | 官方train/test；知识和视觉题都提供原画作；val5124不用 | CL-CrossVQA（WACV 2025）、ArtQuest（WACV 2024）；具体子集见调研 |
| T6 · 交通／自动驾驶 | CODA-LM | [Kai Chen等，WACV 2025](https://github.com/DLUT-LYZ/CODA-LM) | 20,495 | 205 | 2,123 | 作者English LLaVA Train/Test；源卡20495/2123；Val18455、Mini193不用 | MPDrive（CVPR 2025）实际评测；RoboTron-Drive（ICCV 2025）实际训练和评测 |

六任务合计（当前源文件／源卡口径，冻结后复核）：全量训练 **200,831**，1%训练 **2,012**，共同完整测试 **26,366**。
FinMME的ID3698、9806金答案为空，原记录单独保留审计，不伪造、不以空串监督。作者仅发布名为train的评测池；本研究按图像字节SHA256分组，组哈希seed42排序，20%组留作测试。原选项作为输入，verified_caption/related_sentences只作来源元数据，不输入模型，避免泄漏答案。
1%组按源题型分层，最大余数分配ceil(1%×N)，seed42哈希选ID。所有方法共享ID、顺序及测试SHA256；不自行删掉选择、分类、是非题。CODA-LM用作者原问题、答案和区域红框图，不自行造图或改写问答。

## 多模态依据

六个来源都有真实图像和问题文本，模型同时接收两种输入。**图文任务身份不等于每一道题都必须看图**：SLAKE知识题、AQUA知识题及语言偏置可能允许捷径。原始构成照常保留；若主张模型实际利用图像，需要额外的固定测试集遮图／错配图消融，当前未声称已做。

| 数据集 | 视觉来源与任务 |
|---|---|
| TextVQA | 自然照片＋问题：读取场景文字并结合图像作答 |
| SLAKE | 医学影像＋中英问题；保留视觉题、知识题及OPEN/CLOSED |
| RSVQA-LR | 遥感图像＋问题：计数、存在性、比较与区域判断 |
| FinMME | 真实金融图表＋问题、选项；单选、多选和数值题全保留 |
| AQUA | 画作＋问题：视觉内容和艺术知识问答 |
| CODA-LM | 真实道路图像＋作者指令：场景理解、区域理解、驾驶建议；区域题保留作者红框图 |

原始发表、独立复用正文与限制见[六域多模态核验](data/MULTIMODAL_SIX_DOMAIN_AUDIT_20261016.md)。医学首发ISBI、遥感首发TGRS、艺术首发workshop，不把后续顶会复用写成首发顶会。

## 原始数据 sample

[完整问题、选项、答案和原图](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md) · [机器审计](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/dataset_audit.json)。长答案在下表节选；旧样例不在本页展示。

| 数据集 | 原始问题 | 参考答案 | 原图 |
|---|---|---|---|
| [T1 · TextVQA](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#textvqa) | what is the brand of phone? | nokia | <img src="sample/six_domain_cl_20261016_finmme_coda/TextVQA.jpg" width="180" alt="TextVQA原始图像"> |
| [T2 · SLAKE](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#slake) | What modality is used to take this image? | MRI | <img src="sample/six_domain_cl_20261016_finmme_coda/SLAKE.jpg" width="180" alt="SLAKE原始图像"> |
| [T3 · RSVQA-LR](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#rsvqa-lr) | Is it a rural or an urban area | urban | <img src="sample/six_domain_cl_20261016_finmme_coda/RSVQA-LR.png" width="180" alt="RSVQA-LR原始图像"> |
| [T4 · FinMME](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#finmme) | Which loan category had the highest growth rate in 1Q23? | A | <img src="sample/six_domain_cl_20261016_finmme_coda/FinMME.png" width="180" alt="FinMME原始图像"> |
| [T5 · AQUA](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#aqua) | what was exhibited at the 1861 salon | this painting | <img src="sample/six_domain_cl_20261016_finmme_coda/AQUA.jpg" width="180" alt="AQUA原始图像"> |
| [T6 · CODA-LM](data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/samples.md#coda-lm) | &lt;image&gt;<br>There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car&#x27;s driving behavior: vehicles (cars, …（节选） | In the traffic image under observation, several road users and elements are noticeable:<br><br>1. Vehicles:<br>   - A green bus is directly occupying the lane in front o…（节选） | <img src="sample/six_domain_cl_20261016_finmme_coda/CODA-LM.png" width="180" alt="CODA-LM原始图像"> |

## 参数与资源记录

各方法在全部六阶段结束后仅保存一次`exp/six_domain_cl_20261016_finmme_coda/runs/<condition>/<method>/<run_id>/final_parameters/`：完整最终模型（含冻结专家、合并权重）、方法与路由状态、配置、处理器、代码和SHA256。不存阶段、优化器、恢复checkpoint；参数不进Git。下游评定前必须新进程重载并验证输出等价。
两组每个方法记录loss、裁剪前后真实梯度范数、CPU RSS/PSS/USS、GPU allocated/reserved/NVML；单列训练前、训练后具体占用、变化与峰值。训练后测量在释放优化器后、最终参数导出前。
[执行策略](exp/six_domain_cl_20261016_finmme_coda/execution_policy.json) · [协议](rules/016_six_multimodal_finmme_coda_20261016.md) · [进度](summary/six_domain_cl_20261016_finmme_coda/progress.md)。
