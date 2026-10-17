> 当前唯一主指标与结论先看[Answer F1诊断重审](PRIMARY_F1_REASSESSMENT.md)。[冻结规则](../../rules/020_answer_f1_primary_20261017.md)明确无选项映射。旧原始EM/ROUGE记录仅追溯。

# OneVision痛点研究：阅读入口

**先读[探究总报告](RESEARCH_REPORT.md)。** 按12个问题统一写为：探究的问题 → 做的实验 → 结果 → 证明了什么 → 改进的方法 → 改进的效果。未完成、负结果、局部有效均明确标记。

本文导航于2026-10-17整理。旧快照保留用于追溯，不再作为当前成果入口；不移动固定路径，避免破坏脚本与证据引用。

| 想看什么 | 当前入口 | 内容与状态 |
| --- | --- | --- |
| 已做什么、证明／证伪了什么、改进有没有用 | [RESEARCH_REPORT](RESEARCH_REPORT.md) | 主报告，12项统一六栏 |
| 低GPU增长能否代表总容量不变 | [容量资源核验](capacity_resource_evidence/VERIFIED_RESULTS.md) | 固定GPU计数仍可对应1/2/3/5个CPU专家；成本与峰值范围全报，无内存优势声明 |
| 更多LoRA是否也改善选择 | [ROUTING_MECHANISM_EVIDENCE](ROUTING_MECHANISM_EVIDENCE.md) | 同池路线NLL对照；19原型/500选择跨容量相同，不声称F1提升 |
| 完整确认/小诊断能否外推 | [cap1完成与衔接](full_confirmation_evidence/CAP1_COMPLETION.md) | 全量表/155题队列差异/原型启动；同宏平均可掩盖域间误差 |
| 完整21454确认是否完成 | [自动结果入口](full_confirmation_evidence/VERIFIED_RESULTS.md) | cap1完整21454已核验，AP41.361；cap3待测，完整FWT/BWT为空 |
| 遥感全部同答：是否没看图 | [RURAL_COLLAPSE_EVIDENCE](RURAL_COLLAPSE_EVIDENCE.md) | 5权重×3视图×TRAIN／TEST全部完成 |
| 类别平衡有没有修复坍缩 | [RURAL_WEIGHT_EVIDENCE](RURAL_WEIGHT_EVIDENCE.md) | 三条轨迹全部完成；44→50，仍94题urban |
| 知识是否真正写入、是否帮助问答 | [KNOWLEDGE_FIT_EVIDENCE.json](KNOWLEDGE_FIT_EVIDENCE.json) | 正确／错配×rank1／8、0／8／64步全部完成 |
| 早期知识对照的格式／内容混杂 | [KNOWLEDGE_FORMAT_AUDIT](KNOWLEDGE_FORMAT_AUDIT.md)、[KNOWLEDGE_MECHANISM_AUDIT](KNOWLEDGE_MECHANISM_AUDIT.md) | 原17条件与事实池边界；不与新64步结果混算 |
| 多训与旧LoRA复用 | [COMPLETED_CONTROLS](COMPLETED_CONTROLS.md)、[FACTOR_UPDATE_EVIDENCE](FACTOR_UPDATE_EVIDENCE.md) | 9轨迹18终点与单步A不变证据 |
| 正则化什么时候生效 | [EWC_ACTIVATION_AUDIT](EWC_ACTIVATION_AUDIT.md)、[FIRST_STEP_CAUSALITY_AUDIT](FIRST_STEP_CAUSALITY_AUDIT.md) | 首步最新项0；不等于已证明首步有害 |
| 补齐多域／少任务因果缺口 | [确定性双锚协议](../../analysis/ONEVISION_MATCHED_TASK_DOMAIN_PROTOCOL.md) | CPU验证通过并部署等待；GPU门槛及48轨迹结果未产生 |
| 同域／跨域及不同机制 | [TASK_DOMAIN_INTERIM](TASK_DOMAIN_INTERIM.md)、[TASK_DOMAIN_MECHANISM_AUDIT](TASK_DOMAIN_MECHANISM_AUDIT.md) | 24组合完成，数值／单锚限制未消除 |
| 总分掩盖了什么 | [ACQUISITION_COLLAPSE_AUDIT](ACQUISITION_COLLAPSE_AUDIT.md) | 47→47中有9丢失、9新增 |
| 参数、梯度与实际训练占用 | [KNOWLEDGE_TRAINING_AUDIT.json](KNOWLEDGE_TRAINING_AUDIT.json)、[MEMORY_WINDOW_AUDIT](MEMORY_WINDOW_AUDIT.md) | 4病例64步原始梯度及8个训练内存窗口 |
| 原始协议、运行恢复和失败 | [实验README](../../exp/onevision_diagnostics_20261016/README.md)、[修复队列](full_input_repair/queue_state.json) | 统一队列为cap1→核心48→cap3，失败保留 |

`INTERIM_FINDINGS.md`、`MECHANISM_PROGRESS.md`等保留为过程快照，不能覆盖最新完成证明；`research_status.json`已同步本次核验状态。当前机器状态需同时核对真实PID与日志；单独状态文件不证明进程活着。

`KEY_FINDINGS.md`保留机制台账；其数学性质、实际观察和性能因果分别标注。所有诊断集均标明范围，不冒充完整21454题结果。

- [知识写入后的迁移核查](KNOWLEDGE_TRANSFER_AUDIT.md)：包含全部11个反向问答案例及正确／错配对照；词形差异仅待核查，不另造准确率。

- [完整确认已结束四域](FULL_COMPLETED_DOMAINS_AUDIT.md)：19331题，原生任务细项及既有400题／其余18931题分开；交通仍运行，不称全量全部完成。

- [确定性19阶段同域／跨域拆分](DOMAIN_BOUNDARY_AUDIT.md)：医学NLL同域-0.367、后续跨域+0.319；任务内容/预算未匹配，只作为待复核线索。中间未生成的原固定队列格保持null。

- [原答案预算与选项格式](ANSWER_BUDGET_AND_OPTION_AUDIT.md)：完整金融129条正确选项内容却原EM=0；1%训练及500诊断未发生答案截断。辅助审计，不是训练提升，不替换主表。

- [梯度冲突的阶段内时间位置](GRADIENT_TIMING_EVIDENCE.md)：128更新31次负点积中26次在后续批次，原向量仅32首批可重算；非最终质量因果证明。总报告已按Answer F1完整重写，旧版归档。

- [训练后的视觉输入作用](TRAINED_VISION_F1_EVIDENCE.md)：同一最终权重医学100题原图43.022、错图30.889、无图24.310；配对区间/同token反边界与300错图像素核验全部保留。无新训练、非完整测试。

- [当前统一调度](causal_before_full_method_queue/ACTIVATION.json)：用户已委托GPU1方案并授权两模块；原cap1后先12组原型、原48组归因，再四组完整19阶段/最终确认，cap3和稀疏FWT/BWT保留。只停旧CPU协调器，原GPU推理与基线保持；旧协调器PID失效为计划停用。

- [核心对照图像结构预检](matched_task_domain/INPUT_GEOMETRY_AUDIT.md)：879固定QA/35尺寸（含队列外故障检查），实际CPU processor与官方打包形状核对；仅原故障检查2052/2106，固定队列0异常。初次Python整数简化路径漏检已作废重做，实际GPU门槛仍待执行。

- [从问题到两模块原型](two_module_prototype/PROTOCOL.md)：用户最新授权已覆盖旧暂停设计要求；已有CPU检查与GPU排队，实际提升待测。全19阶段/完整测试仍必须完成，不能以12组独立小试验冒充主实验。
- [完整两模块19阶段协议](two_module_fullstream/PROTOCOL.md)：训练/完整最终/稀疏转移评测及恢复代码已实现并排队，[结果入口](two_module_fullstream/VERIFIED_RESULTS.md)按155诊断与21454完整分表，当前无GPU效果；成本和失败均记录。
- [知识迁移的字面相关性复核](KNOWLEDGE_LOCALITY_EVIDENCE.md)：四组初始目标输出/NLL逐题完全相同，31/40/196原生KG分组及8/64预算全部审计；“优于错误知识”不能代替“优于不加”。事后字面组不是语义/图像实体覆盖证明，未识别唯一原因。
- [原目标完成审计](CAUSAL_STATUS.md)：明确哪些要求由真实证据满足、哪些仍待严格GPU控制/全量确认；方法代码准备不能代替原因归属，不缩减目标。
