# LLMCL：小模型的五领域多模态持续学习

## 当前研究：LLaVA-OneVision-0.5B诊断

**最新研究要求：先解释痛点，暂停新方法设计。** 优先检验更多训练反而退步、事实正确性不带来特异收益、聚合准确率掩盖旧答案丢失等现象；不能预设“LoRA无效”。[痛点与因果核验清单](summary/onevision_diagnostics_20261016/PAINPOINT_FIRST.md)。

当前采用语言骨干0.5B、完整模型约0.9B的LLaVA-OneVision，独立使用新服务器GPU1。冻结1%训练1313条、诊断500条，完整测试21454条；Qwen2-VL基线在两机后台独立运行，不跨基座排序。

**受限LoRA容量诊断已完成，但尚未证明自动找到最优数量。** Kmax=1/2/3/5四组实际保留1/2/3/5个专家，155题原生任务覆盖生成集答对47/47/53/55题。分配由当前与旧TRAIN探针NLL决定，输入路由只读取问题与图片，不使用测试答案。四组均达到上限；目前只是有上限的训练驱动复用／新增，尚无成本驱动的主动停止结论。新增收益主要来自遥感，金融未同步提高。[完整控制与配对区间](summary/onevision_diagnostics_20261016/COMPLETED_CONTROLS.md)。

**基底／更新预算9条轨迹、18个终点已完成。** 随机A+零B、旧自然A+零B及完整旧A+B，在三个原生任务上做共享前缀的1／4更新控制。旧函数的收益区别于仅继承A；医学任务可随更新改善，但旧A+B遥感EM由76/100降至44/100，不能用“更新不足”解释全部低分。额外4遍是机制干预，不冒充正式1epoch结果。[设置](analysis/ONEVISION_BASIS_BUDGET_PROTOCOL.md)。

**完整21454题确认正在独立重测。** 原cap1在FinMME:5082出现2052个图像占位token与2106行视觉特征不匹配；失败证据和旧部分结果保留，不计为完成。新脚本只校正连续占位token展开长度；五域TRAIN首题及故障题与官方完整forward的NLL误差均为0，原像素／问题／答案／权重不变。nohup服务仅用GPU1，按预先指定cap1、cap3从头全量测；500题已看诊断与其余20954题分别报告，最终评测不推导完整FWT/BWT。[修复脚本](analysis/onevision_full_input_repair.py)。

知识干预17/17已完成：rank8正确／低相关／错配知识医学EM为16/31/23（各100题）；正确事实并未稳定胜过负对照。事实池字面关系及tail匹配有限，16事实编码的原模板回忆也未验证成功，因此不能据负结果断言专业知识无用。[知识阶段结果](summary/onevision_diagnostics_20261016/KNOWLEDGE_INTERIM.md)、[内容与参数审计](summary/onevision_diagnostics_20261016/KNOWLEDGE_MECHANISM_AUDIT.md)。医学100与原生KG267重叠12题，不合并为独立367题。

训练后视觉4模型×3视图×300题已完成。最终顺序模型医学原图／无图／错图NLL为1.870／2.356／2.136，说明图像输入不能简单视为无效；这些是NLL诊断，不证明视觉能力充分或专业知识是主因。原基座医学正确图与错图差的区间仍跨0。[原基座视觉证据](summary/onevision_diagnostics_20261016/VISUAL_EVIDENCE.md)。

默认后端任务／领域4×6对照已完成。同域最终锚EM9/23，跨域4/23；跨五域虽然总EM高于S1的2/23，S1答对的两题已全部丢失、另新增四题，必须区分旧答案遗忘与新答对题。[配对变化、梯度和预算审计](summary/onevision_diagnostics_20261016/TASK_DOMAIN_MECHANISM_AUDIT.md)。单一23题锚、默认数值差及内容／token预算混杂限制因果解释。

新确定性19阶段顺序／独立任务控制已完成，完整T1权重逐元素复现通过。旧默认后端配对失败记录保留，不能用新结果回填旧运行。9个独立单更新任务A不变、B改变符合零B初始化的首步梯度规律，不是零学习率故障。[因子审计](summary/onevision_diagnostics_20261016/FACTOR_UPDATE_EVIDENCE.md)、[短数值复现](summary/onevision_diagnostics_20261016/REPRODUCIBILITY_EVIDENCE.md)。

每阶段参数／RNG／SHA、loss、真实lr、梯度及CPU/GPU内存前后／峰值保留。固定TRAIN探针和基底预算控制保存原始梯度；不能声称知识训练保存了每一步原始梯度。旧Qwen九个单更新阶段存在ceil-warmup导致lr=0的混杂，活动源码不热改。基线与研究均通过nohup后台服务运行；研究尚未完成，不以小诊断集结果替代全量确认。

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
<tr><td><a href="https://proceedings.mlr.press/v306/padmanabhan26a.html" title="Updating Parametric Knowledge with Context Distillation Retains Post-Training Capabilities"><strong>DiSC</strong></a><br>ICML 2026<br><a href="paper/pre_training/2026_ICML_DiSC.pdf">PDF</a></td><td></td><td><a href="https://proceedings.mlr.press/v306/xia26j.html" title="Adversarial Latent Embedding Repair for LLM Continual Learning"><strong>ALER</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_regularization/2026_ICML_ALER.pdf">PDF</a></td><td><a href="https://proceedings.mlr.press/v306/xu26cj.html" title="Hyper-LLaVA: Hyperbolic Uncertainty-aware Modality-Balanced Routing for Multimodal Continual Instruction Tuning"><strong>Hyper-LLaVA（本轮移出）</strong></a><br>ICML 2026<br><a href="paper/pre_finetuning/method_architecture/2026_ICML_Hyper_LLaVA.pdf">PDF</a></td><td></td></tr>
<tr><td><a href="https://proceedings.mlr.press/v306/borodin26a.html" title="Diffract: Spectral View of LLM Domain Adaptation"><strong>Diffract</strong></a><br>ICML 2026<br><a href="paper/pre_training/2026_ICML_Diffract.pdf">PDF</a></td><td></td><td><a href="https://proceedings.iclr.cc/paper_files/paper/2025/hash/74fc5575632191d96881d8015f79dde3-Abstract-Conference.html" title="Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning"><strong>FV-guided Training</strong></a><br>ICLR 2025<br><a href="paper/pre_finetuning/method_regularization/2025_ICLR_FV_guided_Training.pdf">PDF</a></td><td><a href="https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html" title="CL-MoE: Enhancing Multimodal Large Language Model with Dual Momentum Mixture-of-Experts for Continual Visual Question Answering"><strong>CL-MoE（本轮新增，多模态）</strong></a><br>CVPR 2025<br><a href="paper/pre_finetuning/method_architecture/2025_CVPR_CL_MoE.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Tong_Any-SSR_How_Recursive_Least_Squares_Works_in_Continual_Learning_of_ICCV_2025_paper.html" title="Any-SSR: How Recursive Least Squares Works in Continual Learning of Large Language Model"><strong>Any-SSR</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_Any_SSR.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Guo_Federated_Continual_Instruction_Tuning_ICCV_2025_paper.html" title="Federated Continual Instruction Tuning"><strong>FCIT</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_FCIT.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Instruction-Grounded_Visual_Projectors_for_Continual_Learning_of_Generative_Vision-Language_Models_ICCV_2025_paper.html" title="Instruction-Grounded Visual Projectors for Continual Learning of Generative Vision-Language Models"><strong>MVP / Instruction-Grounded Visual Projectors</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_MVP_Instruction_Grounded_Visual_Projectors.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td></td><td></td><td><a href="https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html" title="SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning"><strong>SMoLoRA</strong></a><br>ICCV 2025<br><a href="paper/pre_finetuning/method_architecture/2025_ICCV_SMoLoRA.pdf">PDF</a></td><td></td></tr>
<tr><td></td><td><a href="https://arxiv.org/html/2503.00413#S5"><strong>ER-LoRA（本轮新增）</strong></a><br>标准经验回放对照；CL-MoE CVPR 2025评测，非该年新方法</td><td></td><td></td><td></td></tr>
</tbody>
</table>

归类按主要机制，混合机制、原论文模型及生成任务见 [论文完整索引](paper/README.md)。预训练条目包含语言／知识／上下文扩展型 CPT，不全部具有序贯抗遗忘实验；HPA 归入持续安全对齐保持。**本轮加入ER-LoRA作为回放对照；DGG仅保留在文献表，不参与本轮实验。**

## 当前五数据集／19原生任务协议

独立实验：`five_domain_native_cl_20261016_sparse`；代码／参数在`exp/five_domain_native_cl_20261016_sparse/`，日志报告在`summary/five_domain_native_cl_20261016_sparse/`，冻结清单在`data/FiveDomainCL/five_domain_native_cl_20261016_sparse/`。旧六领域实验已按用户要求停止，不复用旧预测。
基座Qwen2-VL-2B-Instruct；seed42；一轮；无验证。十方法：SeqFT-LoRA、ER-LoRA、CL-MoE、SMoLoRA、PASs-MoE、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。SeqFT-LoRA是标准顺序微调基础对照的描述性名称，不作为独立论文算法。Hyper-LLaVA保留在文献表，移出本轮实验。

本轮新增：**ER-LoRA → 持续微调／基于回放**；**CL-MoE → 持续微调／基于架构／多模态，CVPR 2025主会**。ER是标准经验回放对照，也由CL-MoE论文评测，不计作新增2025论文。CL-MoE按论文公式5–12适配Qwen2-VL：8专家、总rank64、alpha128、top2、gamma0.7、beta0.5；保留实例路由、任务路由和按专家关系的动量更新。Qwen没有encoder CLS，任务中心使用冻结解码器最后一个问题token的特征；推理不传入测试任务ID。作者公开脚本中任务推断被注释、参数合并前缀不匹配，本轮按论文公式实现并披露差异，不能称原架构精确复现。ER容量5000，只访问过去训练样本；固定哈希选取和回放ID，当前／回放损失1:1。详见[适配说明](exp/five_domain_native_cl_20261016_sparse/METHOD_ADAPTATIONS.md)。

两台服务器Python 3.10.21、55项已安装分发版本、23个执行源码文件及策略哈希已逐项匹配。新服务器GPU0保留、GPU1留给自研方法，GPU2跑1%、GPU3跑全量；旧服务器GPU0–3跑1%。SMoLoRA遵循作者BF16配置，仅安排支持BF16的3090，其他方法基础权重FP16；分主机记录硬件和耗时。
新服务器GPU0保留、GPU1留给自研，GPU3全量，GPU2抽样；旧服务器GPU0–3均用于抽样。每卡一个进程；分工互斥，显存预检不足的旧机方法回到新机，不改算法或测试数据。
原生任务顺序：TextVQA → SLAKE的Modality/Position/Organ/Size/Abnormality/Quantity/Plane/Shape/Color/KG → RSVQA-LR的rural_urban/presence/count/comp → FinMME → CODA-LM的general/regional/suggestion。标签是原标注，顺序是本研究预先固定，不声称是官方课程；金融答案格式、医学OPEN/CLOSED另行分组统计，不重复训练。
初始全测19任务；中间只测刚学任务和下一任务；最终全测。共**74组测试／80,316条预测**；未评测格为null，不伪造完整20×19矩阵。不报告ForgetRate；保留原生任务等权AP/FWT/BWT、逐题EM正确数/总数、Token F1及ROUGE-L。领域最终分数按该域测试QA数加权，另报五领域宏平均，避免与19任务等权AP混淆。统一词面指标不等于原生评分：FinMME涉及数值容差和多选；CODA-LM原文采用LLM评审，EM不是它的官方准确率。

## 实际运行快照与诊断安排

2026-10-16 16:08（北京时间）核查：两机共六个正式运行进程，均处于初始完整评测，训练阶段完成数为0，不能把队列中的`training`状态解释为已经完成训练。CL-MoE已通过19任务真实2B反向传播，以及独立进程重载的损失等价／有限梯度检查，重载损失误差为0；诊断权重已丢弃，正式运行从原基座开始。

| 主机 | GPU | 当前正式进程 | 条件 | 实际阶段 |
|---|---:|---|---|---|
| 新3090 | 2 | CL-MoE | 1%训练／完整测试 | 初始评测 |
| 新3090 | 3 | SeqFT-LoRA | 全量训练／完整测试 | 初始评测 |
| 旧2080Ti | 0 | SeqFT-LoRA | 1%训练／完整测试 | 初始评测 |
| 旧2080Ti | 1 | ER-LoRA | 1%训练／完整测试 | 初始评测 |
| 旧2080Ti | 2 | PASs-MoE | 1%训练／完整测试 | 初始评测 |
| 旧2080Ti | 3 | PaRSP | 1%训练／完整测试 | 初始评测 |

其余任务排队接续；新服务器GPU1保持空闲，供后续诊断使用。当前不预设自研算法：先从已有实验分析单任务能否学会、跨任务遗忘、视觉依赖和资源开销，再决定要解决的问题。T5基座核查是独立调研，不切换正在运行的基座。[启动报告](summary/five_domain_native_cl_20261016_sparse/LAUNCH_REPORT.md) · [实际检查证据](summary/five_domain_native_cl_20261016_sparse/launch_proofs.json)。本节是带时间戳的静态快照，实时进度以服务器状态文件为准。

## T5／VL-T5基座核查

T5是文本encoder-decoder模型家族；T5-Small为60M、Base为220M、Large为770M，另有3B和11B，因此不能只凭“T5”判断大小。[Google官方模型列表](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints)。T5-Base语言模型的FP16纯权重约0.44GB（220M×2字节推算），不包括视觉模块、激活、梯度和优化器，也不是实测训练显存。

图文持续问答论文常见的是**VL-T5**，它在T5-Base基础上增加视觉输入；作者原始流程使用每图36个区域的预提取特征。需要把视觉特征提取的时间、模型和存储一起计算。[VL-T5官方代码](https://github.com/j-min/VL-T5) · [视觉特征说明](https://github.com/j-min/VL-T5/blob/main/feature_extraction/README.md)。

不能据某张结果表的多行VL-T5认定大部分新方法都用T5：VQACL（CVPR2023）采用VL-T5；CL-MoE（CVPR2025）表1同时对照VL-T5和LLaVA-7B，其提出的方法使用LLaVA-7B；当前参考的Continual LLaVA采用Vicuna语言模型。[VQACL源码](https://github.com/zhangxi1997/VQACL) · [CL-MoE正文](https://arxiv.org/html/2503.00413) · [Continual LLaVA正文](https://arxiv.org/html/2411.02564)。

VL-T5适合作为小模型诊断候选，但当前五域是否可用尚未实测：医学、遥感、金融图表与原始自然图像区域特征可能存在差异；TextVQA需考察读字能力，SLAKE需考察原生双语，CODA-LM需考察长答案。不能靠删除这些题、翻译原文或丢弃图像来适配。先核验预训练权重和图像特征管线，再做单任务可学习性／速度／显存测量；是否替换主基座须据测量决定。[完整核查](data/T5_BACKBONE_AUDIT_20261016.md)。

## 数据集：来源、数量与复用依据

单位均为QA，每个来源训练QA低于十万。全部来源已按完整文件计数并冻结清单。保留原始选择、是非、计数等构成。

| 领域 | 数据集（图文） | 提出者／首发 | 全量训练 | 1%训练 | 完整测试 | 划分 | 独立实际使用 |
|---|---|---|---:|---:|---:|---|---|
| D1 · 自然／通用 | TextVQA | [Amanpreet Singh等，CVPR 2019](https://textvqa.org/) | 34,602 | 347 | 5,000 | 官方train/validation；公开validation作测试 | CoIN（NeurIPS 2024）、SMoLoRA（ICCV 2025） |
| D2 · 医学 | SLAKE | [Bo Liu等，ISBI 2021](https://www.med-vqa.com/slake/) | 9,835 | 99 | 2,094 | 官方train/test；双语；val2099不用 | Uncertainty-Driven Expert Control（ICCV 2025） |
| D3 · 地理／遥感 | RSVQA-LR | [Sylvain Lobry等，TGRS 2020](https://rsvqa.sylvainlobry.com/) | 57,223 | 573 | 10,004 | 官方active掩码train/test；val10005不用；不自行分箱 | GeoChat（CVPR 2024）、LHRS-Bot（ECCV 2024）、RSAdapter（TGRS 2024） |
| D4 · 金融 | FinMME | [Junyu Luo等，ACL 2025主会](https://aclanthology.org/2025.acl-long.1426/) | 8,864 | 89 | 2,233 | 源池11099；2条缺答案单列；11097有标注QA按原图SHA256分组seed42研究80/20，非官方划分 | Beyond Real Weights（WACV 2026）实际评测；AgentFinVQA（2026预印本）实际评测 |
| D5 · 交通／自动驾驶 | CODA-LM | [Kai Chen等，WACV 2025](https://github.com/DLUT-LYZ/CODA-LM) | 20,495 | 205 | 2,123 | 作者English LLaVA Train/Test；源卡20495/2123；Val18455、Mini193不用 | MPDrive（CVPR 2025）实际评测；RoboTron-Drive（ICCV 2025）实际训练和评测 |

五数据集合计：全量训练 **131,019**，1%训练 **1,313**，共同完整测试 **21,454**。拆成19阶段不增加样本或重复训练。SLAKE的9,835/99训练QA是十类合计，不是每类数量；其1%最小类仅1条，细分类结果须结合测试样本数解释。
FinMME的ID3698、9806金答案为空，原记录单独保留审计，不伪造、不以空串监督。作者仅发布名为train的评测池；本研究按图像字节SHA256分组，组哈希seed42排序，20%组留作测试。原选项作为输入，verified_caption/related_sentences只作来源元数据，不输入模型，避免泄漏答案。
1%组按源题型分层，最大余数分配ceil(1%×各数据集训练量)，seed42哈希选ID；沿用父实验已冻结的ID，再按原生任务互斥拆分，不重新抽样。每阶段ID、顺序和测试SHA256事先冻结在[split_lock.json](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/split_lock.json)；两服务器、十方法共用且启动核验，文件只读。[19阶段实际数量](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/partition_audit.json)。不自行删掉选择、分类、是非题。CODA-LM保留作者原问题、答案和区域红框图。

## 多模态依据

五个来源都有真实图像和问题文本，模型同时接收两种输入。**图文任务身份不等于每一道题都必须看图**：SLAKE知识题及语言偏置可能允许捷径。原始构成照常保留；若主张模型实际利用图像，需要额外的固定测试集遮图／错配图消融，当前未声称已做。

| 数据集 | 视觉来源与任务 |
|---|---|
| TextVQA | 自然照片＋问题：读取场景文字并结合图像作答 |
| SLAKE | 医学影像＋中英问题；保留视觉题、知识题及OPEN/CLOSED |
| RSVQA-LR | 遥感图像＋问题：计数、存在性、比较与区域判断 |
| FinMME | 真实金融图表＋问题、选项；单选、多选和数值题全保留 |
| CODA-LM | 真实道路图像＋作者指令：场景理解、区域理解、驾驶建议；区域题保留作者红框图 |

原始发表、独立复用正文与限制见[六域多模态核验](data/MULTIMODAL_SIX_DOMAIN_AUDIT_20261016.md)。医学首发ISBI、遥感首发TGRS，不把后续顶会复用写成首发顶会。

## 原始数据 sample

[完整问题、选项、答案和原图](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md) · [机器审计](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/partition_audit.json)。长答案在下表节选；旧样例不在本页展示。

| 数据集 | 原始问题 | 参考答案 | 原图 |
|---|---|---|---|
| [D1 · TextVQA](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md#textvqa) | what is the brand of phone? | nokia | <img src="sample/five_domain_native_cl_20261016_sparse/TextVQA.jpg" width="180" alt="TextVQA原始图像"> |
| [D2 · SLAKE](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md#slake) | What modality is used to take this image? | MRI | <img src="sample/five_domain_native_cl_20261016_sparse/SLAKE.jpg" width="180" alt="SLAKE原始图像"> |
| [D3 · RSVQA-LR](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md#rsvqa-lr) | Is it a rural or an urban area | urban | <img src="sample/five_domain_native_cl_20261016_sparse/RSVQA-LR.png" width="180" alt="RSVQA-LR原始图像"> |
| [D4 · FinMME](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md#finmme) | Which loan category had the highest growth rate in 1Q23? | A | <img src="sample/five_domain_native_cl_20261016_sparse/FinMME.png" width="180" alt="FinMME原始图像"> |
| [D5 · CODA-LM](data/FiveDomainCL/five_domain_native_cl_20261016_sparse/samples.md#coda-lm) | &lt;image&gt;<br>There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car&#x27;s driving behavior: vehicles (cars, …（节选） | In the traffic image under observation, several road users and elements are noticeable:<br><br>1. Vehicles:<br>   - A green bus is directly occupying the lane in front o…（节选） | <img src="sample/five_domain_native_cl_20261016_sparse/CODA-LM.png" width="180" alt="CODA-LM原始图像"> |

## 参数与资源记录

参数保存在`exp/five_domain_native_cl_20261016_sparse/runs/<condition>/<method>/<run_id>/checkpoints/`。SeqFT-LoRA在两组均保存全部19阶段checkpoint；其他九方法每组只存最终第19阶段一份。保存完整模型及方法图（含冻结专家、合并权重、路由运行状态）、处理器、RNG、配置和SHA256，最终不重复导出。每个方法正式启动前必须在独立进程重载真实2B诊断checkpoint，检查前向损失等价和有限反向梯度。阶段边界重建优化器，故不保存旧优化器；SeqFT支持完整阶段边界恢复，其他方法中断后需新建run从基座开始。权重不进Git。
两组每个方法记录loss、裁剪前后真实梯度范数、CPU RSS/PSS/USS、GPU allocated/reserved/NVML；单列训练前、训练后具体占用、变化与峰值。训练后测量在释放优化器后、最终参数导出前。
[执行策略](exp/five_domain_native_cl_20261016_sparse/execution_policy.json) · [协议](rules/017_five_domain_native_sparse_two_hosts.md) · [进度](summary/five_domain_native_cl_20261016_sparse/progress.md)。
