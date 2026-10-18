> **核心背景：轻量、小数据、低内存增长。允许多个LoRA，单LoRA不是硬限制。** 先检验少量数据为何导致既有方法退化，再针对性提升；目标内存增长低于具体对比LoRA方法，须同设置实测，并报告额外计算成本。两台服务器基线持续后台运行。[核心规则](rules/021_lightweight_data_limited_core_20261017.md)。

新[TRAIN调用控制](summary/onevision_diagnostics_20261016/functional_router/calibration_audit/RESEARCH_FINDINGS.md)：同一专家池43条训练记忆，16参数校准将平均NLL3.313454降至1.852542；仅头逐图排除后调用不变。这是调用质量的TRAIN证据，尚不是测试F1收益，未增加LoRA或知识。

评价口径：以各领域总Answer F1、最终AP/FWT/BWT及资源成本评价方法，不要求所有题型、语言或样本全面提升；细分诊断不否定整体增益，也不是必须报告的主结果。
2026-10-18实际运行更新：GPU1 worker348381/startticks14466517已自动启动，原500NLL/155预测/路由精确重载门槛通过；43条TRAIN、172次NLL前向、400次CPU校准更新完成，16参数确有改变。四个完整诊断视图仍未完成，不能宣称TEST收益。外来GPU1进程350639在本作业准入之后进入，保留双方；共同运行墙钟时间不能当独占性能，下一作业仍等待实际空闲。两机基线不变。
# LLMCL：小模型的五领域多模态持续学习

> 当前正式1%对照已改用[双骨干共享框架](exp/two_backbone_native_cl_20261017_v2/README.md)，旧Qwen单步零学习率轨迹已隔离，不能进入主表。旧机OneVision、新机Qwen；新GPU0保留、GPU1研究。

> 双骨干1%结果采用[分骨干主表](summary/two_backbone_pct01_20261017/MAIN_TABLE.md)：五领域最终成绩、AP、FWT、BWT；新正式实验与历史零学习率／小规模诊断分开。

> 全量训练原131019条中仅SLAKE:1622为空答案，经用户授权排除后为**131018条**；原始文件保留。1%训练1313条、完整测试21454条不变。[版本与两机核验](summary/two_backbone_native_cl_20261017_v2/README.md)。

> 唯一主指标现为 **Answer F1（原始Token F1，0–100）**；不做选项映射。两骨干/全部方法及AP/FWT/BWT均用同一口径。[冻结规则](rules/020_answer_f1_primary_20261017.md)。

## 研究成果：按问题阅读

新增[从多领域论文改造的两个轻量模块](summary/onevision_diagnostics_20261016/functional_router/PROTOCOL.md)：借鉴D-MoLE的识别与MVP的相关性聚合，固定已有专家池，用43条原TRAIN的172次NLL前向校准16参数质量头，再比较硬调用／连续LoRA残差。15项CPU机制、10项调度、3项接续安全检查通过，已接入[后台队列](summary/onevision_diagnostics_20261016/functional_router_queue/README.md)；GPU当前归因作业自然结束后执行，不宣称已获收益。两个预定候选均完整21454题确认，不按诊断挑赢家；专家池和额外计算成本仍计入。

[相似度复用方案训练结果](summary/onevision_diagnostics_20261016/learned_reuse/INTERIM_RESULT_20261018.md)已出：同155题/19任务AP固定22.391、可学习31.025、顺序LoRA43.363。可学习版比固定版高8.634pp，仍比顺序LoRA低12.338pp；两组均前3阶段用满3个专家，尚未达到性能和自主少量模块目标。原两组完整测试0/2待测，继续保留。

正式OneVision1%已有SeqFT／ER两方法完成，AP40.286／39.275；另4个在跑、4个待跑。[主表](summary/two_backbone_pct01_20261017/MAIN_TABLE.md)与上述研究AnyRes结果分开。新机曾发生磁盘写满，已保全故障证据、核验写盘及真实S9参数后续跑Qwen SeqFT，ER另建run重跑；[恢复记录](summary/storage_recovery_20261018/README.md)。

新增[别人发现了哪些多领域问题](summary/onevision_diagnostics_20261016/LITERATURE_PROBLEMS.md)：12篇正式会议论文加COAST版本，逐篇列问题、对照证据、解决思路与适用边界；他人成果与本实验结论分开。

**统一阅读入口：[问题发现总报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md)。** 14项均按「探究的问题／做的实验／结果／证明了什么／改进的方法／改进的效果」记录，首页区分已证实、被反例否定和未解决；细节按需进入[证据目录](summary/onevision_diagnostics_20261016/README.md)。

| 重点发现 | 实际结果 | 结论边界 |
| --- | --- | --- |
| 当前任务内部就能损伤开放功能 | CL-MoE Organ当前学习OPEN−5.707，CLOSED+6.935；后续领域还未训练 | 不是全由后来跨域遗忘；当前适配实现，尚未隔离路由/保护/数据原因，[全对照](summary/onevision_diagnostics_20261016/native_acquisition_20261018_v3/RESEARCH_FINDINGS.md) |
| 少更新关联会在领域内反转 | 7方法共同13任务总体一次更新增量较低，医学内CL-MoE/PaRSP/OSFT反向 | 不将相关性当步数因果，不泛化原同任务预算控制 |
| 回放均值稳定可以掩盖开放题退化 | 正式ER Organ全均值差区间含0，OPEN184题却35.661→23.533，−12.129pp区间为负；19个池无容量淘汰 | 排除本例旧TRAIN被容量挤掉，未证抽样覆盖是掉分原因，[全部证据](summary/onevision_diagnostics_20261016/baseline_language_followup_20261018/RESEARCH_FINDINGS.md) |
| 总分与开放题能力、实际成本需分开 | Qwen S4 Organ总分45.209高于OneVision37.314，OPEN却20.985低于35.118；PAS中文强增主要闭式、参数约Seq260倍 | 单任务时点、架构/容量等混杂，不推出纯规模/知识因果或轻量优势 |
| 共同历史后四域换五域，F1与loss可不同步 | Organ507题18.626→17.196，F1差−1.430区间含0，NLL+0.157908区间为正 | 前3阶段权重/首批梯度精确相同，后续内容/顺序仍不同；未证领域数因果，[完整对照](summary/onevision_diagnostics_20261016/matched_task_domain/common_prefix/VERIFIED_RESULTS.md) |
| 跨域差距不全是遗忘 | Organ507共同习得点20.361，同域末34.168、跨域末18.626；跨域净变−1.735区间含0 | 同域存在+13.808pp后续收益；单锚/序列，不能把同域−跨域差全算遗忘 |
| 固定权重改指令能改变语言输出质量 | 跨域双指令中文1.234→13.225，英文36.087→30.632；同域中文22.336→13.819 | 有恢复也有损害；不能按TEST选指令，未证明知识充分 |
| 领域替换混入语言分布变化 | 中文F1同域22.336/跨域1.234；中文预测含汉字251/25，TRAIN后续中文曝光17/0 | 分层差异不是语言因果；不能断言知识完整或差距全为知识遗忘 |
| 正式基线中文可少量习得 | 原Organ26QA/2更新，SeqFT/ER中文F1增量+17.559/+17.362pp | 完整507题学习前→刚学完；最终未完成，不与AnyRes混算 |
| 当前任务尚未学好，不全是后来忘了 | Modality初始/1/4步F1为29.600/29.263/51.956；KG为5.511/7.240/17.918 | 多训练局部有效，遥感却76→44；额外计算另计 |
| 同答仍可保留图像信号 | 遥感全urban，原图仍改善正确答案概率；平衡损失44→50 | 只缓解坍缩，未证明视觉完好或唯一成因 |
| EWC最新中心首步梯度为0 | 32/32首批；增大λ不能改变该性质 | 后续步/更早中心可生效，未证最终准确率因果 |
| 平均旧loss改善不保证子组稳定 | 4条医学TRAIN的11个已学转移中6次均值改善仍有组退化 | 两题一组、内容混杂；不是语言因果/F1或方法收益 |
| 记住事实不保证迁移 | 16事实可回忆，目标医学/KG无普遍稳定改善 | 未排除知识不足、覆盖与视觉实体绑定 |
| 总分掩盖异质性 | 155题22改善/18退化；任务宏平均与领域宏平均方向相反 | 诊断不能替代完整测试 |
| 两模块是否有效 | 12/12原型完成，三任务四组均返回原更新，无新增收益 | 严格48组已启动，完整CL仍待运行，未宣称算法提升 |
| 低显存增长不等于总资源低 | CPU专家池随容量增加；预热改变前后差 | 常驻总量、纯TRAIN峰值和额外计算均需比较 |

新增[回放实际保护了多少旧题](summary/onevision_diagnostics_20261016/matched_task_domain/replay_exposure/VERIFIED_RESULTS.md)：T04两个已完成控制同样回放32次，旧锚曝光却为32/15；预定跨域覆盖与实际完成分列，不拿容量或计划代替性能因果。

新增[同一任务重复训练六控制](summary/onevision_diagnostics_20261016/matched_task_domain/repeat_control_evidence/T04_VERIFIED_RESULTS.md)：投影未触发时5状态/输出精确等于SeqFT；EWC后续会改变参数，λ100的辅助NLL更差，但F1差异区间包含0。不把无实际干预解释为方法失败，不把无域切换结果解释成跨域成因。

新增[真实参数位移核验](summary/onevision_diagnostics_20261016/first_adam_geometry/VERIFIED_RESULTS.md)：两个同起点医学任务，完整LoRA矩阵余弦0.977，扣除共同旧功能后新增位移余弦0.205；高总相似度不证明新学方向重复。梯度范数相差1.919倍，实际参数位移范数近乎相同，不能直接把梯度大小当学习量或遗忘因果。已知优化器性质与本设置观察分开报告。

原生AnyRes单LoRA控制已完成**21454题**：自然75.917、医学40.943、遥感49.010、金融12.991、交通31.916，19任务AP=41.361；完整FWT/BWT尚缺阶段评测。[完整结果及诊断偏差](summary/onevision_diagnostics_20261016/full_confirmation_evidence/CAP1_COMPLETION.md)。此研究与正式双骨干V2输入协议不同，不混排名。

新增[严格同起点首批梯度](summary/onevision_diagnostics_20261016/matched_task_domain/first_transition_geometry/VERIFIED_RESULTS.md)：重复/同域/跨域新旧TRAIN余弦均为正；不以首批负冲突解释最终跨域差，也不把局部Taylor预测当有限F1结果。

共同S1完整507/228推理已[接入新CPU队列](summary/onevision_diagnostics_20261016/causal_measurement_queue/README.md)，用于分开遗忘与同域正迁移缺失；实际GPU评测尚待，不拿准备当结果。

原48组内部[优先两锚的SeqFT同域/跨域共同控制](summary/onevision_diagnostics_20261016/causal_anchor_first_queue/SCHEDULING.md)，再补所有机制。全部48保留，GPU源码/ID/设置不变。

最新[语言混杂因果接续](summary/onevision_diagnostics_20261016/language_causal_queue/README.md)：同nohup服务CPU277043先补可用共同S1，再运行固定两权重×四既有指令、0次训练的所有完整507题格，之后接回原48与后续队列。实际CPU门槛通过，GPU效果待测；原265251仅CPU退役，基线不停，GPU1他人进程存活时等待。[已核验观察](summary/onevision_diagnostics_20261016/matched_task_domain/language_profile/T04_VERIFIED_RESULTS.md)、[干预协议](summary/onevision_diagnostics_20261016/language_cue/PROTOCOL.md)。

新增[正式V2基线学习前/刚学完审计](summary/onevision_diagnostics_20261016/baseline_language_profile/VERIFIED_RESULTS.md)：同一507题上，OneVision SeqFT/ER中文F1约5.118→22.677/22.480，均仅2个实际更新；后续最终点待测。两机6运行10个封存格逐题复算，未改任何基线，也不把初始分数误当习得参照。

GPU1当前及接续：12原型已结束→[严格48组归因](analysis/ONEVISION_MATCHED_TASK_DOMAIN_PROTOCOL.md)→[四组完整19阶段及全测试](summary/onevision_diagnostics_20261016/two_module_fullstream/PROTOCOL.md)→cap3完整确认→稀疏转移。正式基线持续nohup；“最优LoRA数量”“显著方法提升”“优于具体LoRA方法的内存增长”均尚未证明。[自动原型结果](summary/onevision_diagnostics_20261016/two_module_prototype/VERIFIED_PILOT_RESULTS.md)按完成证明更新，不用排队状态冒充结果。

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
