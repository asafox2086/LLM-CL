> 最新接续2026-10-18：[可学习向量空间试验](../../summary/onevision_diagnostics_20261016/learned_reuse/PROTOCOL.md)已接入新[CPU290886队列](../../summary/onevision_diagnostics_20261016/learned_reuse_queue/README.md)。同Kmax3的固定/学习两组，1313TRAIN/19阶段；先补可用共同S1、再两组训练、全格语言对照与原48，随后两组完整21454确认及原完整方法。原277043仅CPU计划退役，GPU/基线/旧观察器保留；13CPU机制+23策略检查通过，GPU1他人占用时健康等待，尚无GPU试验收益。以下为历史接续与原研究协议。

> [语言分布审计](../../summary/onevision_diagnostics_20261016/matched_task_domain/language_profile/T04_VERIFIED_RESULTS.md)显示跨域差距在中文更大，TRAIN语言也变，不能全归因知识遗忘。原[语言队列](../../summary/onevision_diagnostics_20261016/language_causal_queue/README.md)已被上述新入口继承，固定两权重×四既有指令推理全部保留。

> 共同S1完整旧锚补测原入[CPU测量队列](../../summary/onevision_diagnostics_20261016/causal_measurement_queue/README.md)，现由上述新入口继承，观察器265271不变，原48训练全保留。实际GPU重载/735题尚待。

> GPU1现优先原48组中的两个锚、四序列SeqFT共同控制，随后补齐所有机制；数据/训练/评测未改，当前作业自然结束。新[排队规则与保全过程](../../summary/onevision_diagnostics_20261016/causal_anchor_first_queue/SCHEDULING.md)，实时`causal_anchor_first_queue/queue_state.json`；旧CPU220246已计划退役，不恢复。

> 成果统一看[问题发现总报告](../../summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md)：12个问题、每项六栏；首页列已证实、被反例否定与未解决。补入T02/T11初始/1/4步习得和两模块12/12结束快照，不把无实际改动解释成机制失败。

> 原生AnyRes SeqFT/cap1已完成21454题，19任务AP41.361；[完整表](../../summary/onevision_diagnostics_20261016/full_confirmation_evidence/CAP1_COMPLETION.md)。原型已结束，GPU1当前严格48归因→四组完整19阶段/最终全测→cap3→完整稀疏转移。实际进度看[原型自动报告](../../summary/onevision_diagnostics_20261016/two_module_prototype/VERIFIED_PILOT_RESULTS.md)。基线不停，不热改活动/排队源码。

> 核心背景为轻量、1%小数据、低内存增长。允许多个LoRA，当前单LoRA仅为既定原型；比较须计入CPU/GPU常驻总量、统一预热的训练前后/峰值及额外计算。[规则021](../../rules/021_lightweight_data_limited_core_20261017.md)。唯一主指标为Answer F1，不做选项映射；[规则020](../../rules/020_answer_f1_primary_20261017.md)。
# OneVision 0.5B：先诊断，再设计方法

独立实验ID：`onevision_diagnostics_20261016`。用户选择LLaVA-OneVision-Qwen2-0.5B-OV（语言0.5B，完整模型约0.9B），仅使用新机GPU1。Qwen基线继续；跨骨干结果不能混成同基座方法排名。

模型固定官方HF版本`74dd0bf867a4cda7950c17663794267c60cf4b40`，直接镜像下载、不使用代理；逐文件校验官方LFS SHA256或Git blob。运行库复用已冻结的4.46.3环境，不升级影响现有作业。

## 诊断链与可证伪的问题

1. **输入与评价**：图文真实前向／反向、有限梯度、生成格式、原生类别／语言分组、视觉token数与耗时。先查代码／答案格式问题。
2. **可学习性**：未训练基座 vs 五域分别独立训练LoRA，使用原冻结1%训练池，共1313QA；保存训练前后NLL、准确匹配／F1／ROUGE-L和输出。若单域也学不好，不能直接归因为遗忘或专业知识不足；还需容量、学习率、训练预算及视觉信息控制。
3. **顺序干扰**：同一LoRA按原19任务顺序学习，增加19个逐任务独立LoRA作为严格匹配控制（相同QA、更新预算、调度器和测试ID）；五域独立训练另作领域能力诊断。记录训练样本梯度余弦、各阶段参数更新及旧任务损失；负余弦是局部梯度冲突证据，不等于已证明最终因果。
4. **冗余与复用**：构建五个独立LoRA的5×5迁移矩阵；计算实际更新`deltaW=scale*B*A`的精确余弦、奇异谱／95%能量秩、输入输出子空间重合。不能直接比较A/B元素后就称语义重复（分解不唯一）。复用候选只从已学领域中挑选，选择依据来自当前TRAIN探针，禁止按测试成绩选路由或容量。
5. **视觉依赖**：固定诊断测试题，比较原图、确定性错配图、文字输入的NLL／生成差异。这是标注清楚的消融视图，不改原始文件，不把合成输入混入训练或正式成绩。图像增益不足可能来自数据偏置、视觉特征或模型能力，需按领域复核。

## 数据与阶段区分

原正式划分来自`data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json`；核验每个QA文件SHA256，不重采样训练池，不变更问题、答案、图像或原生题型。研究诊断使用五域独立训练，以及相同19原生任务顺序训练，两者分组标注，不能冒充完全相同的任务设置。

为快速定位问题，预先固定每域100个诊断test（按原生任务分层、至少一条、seed42哈希选ID），其中固定每域10条生成探针；其余可先用teacher-forced NLL。诊断样本是原测试集的子集，**不能称完整测试或正式主表结果**；原完整测试21454QA不变，最终候选需完整测试确认。诊断目标答案可计算评价，不能用于路由、容量、超参数选择。训练路由／梯度探针从原训练池事先冻结，最多每域16条；不另建验证集。

rank8、alpha32、dropout0.1、q/v投影、AdamW、seed42、1epoch、有效batch16；warmup使用floor(0.03×更新数)且小于更新数，单更新阶段不warmup。已确认旧队列ceil方案会令单次更新任务学习率为0；该旧设置单列混杂因素，不热改运行中的基线或将其解释为能力／遗忘。初始使用官方原生AnyRes图像预处理，不静默缩小图像以制造加速。BF16在3090上运行。保存每阶段LoRA参数、固定基座哈希、所有诊断输出、loss／真实梯度、CPU RSS/PSS/USS和GPU训练前后／峰值。LoRA快照仅描述冻结基座上的更新；不声称存了独立全模型副本。

## 根据证据再做的容量实验

用户提出的研究假设：给定LoRA容量上限Kmax，模型决定复用已有LoRA或增设模块。先测固定单LoRA、每域独立LoRA、只复用旧LoRA、复用后继续训练，明确新数据／额外计算量。若迁移矩阵和子空间分析支持复用，再比较Kmax=1/2/3/5的容量与性能曲线；路由只用训练信息，拒绝测试oracle选最佳模块。报告实际模块数、总参数／可训练参数、存储／显存和结果，不能只报上限。

这些是待检验假设，不预先命名为已发表算法，不承诺大幅提升。不把小数据单seed初步趋势当成机制证明。

日志与参数：`exp/onevision_diagnostics_20261016/runs/`；报告：`summary/onevision_diagnostics_20261016/`。服务先核查模型完整性和GPU1空闲，再启动诊断；不抢占新机GPU0/2/3或旧机GPU。

## 接续研究优先级

已有初步格式／语言混杂线索，见[INTERIM_FINDINGS](../../summary/onevision_diagnostics_20261016/INTERIM_FINDINGS.md)。接续时先做独立的固定输出格式／问题语言控制，再将知识不足、梯度冲突与复用增益归因。当前活动runner不热改，另起可追溯诊断。
