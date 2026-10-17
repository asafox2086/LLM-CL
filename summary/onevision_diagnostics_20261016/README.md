# OneVision痛点研究：阅读入口

**先读[探究总报告](RESEARCH_REPORT.md)。** 按8个问题统一写为：探究的问题 → 做的实验 → 结果 → 证明了什么 → 改进的方法 → 改进的效果。未完成、负结果、局部有效均明确标记。

本文导航于2026-10-17整理。旧快照保留用于追溯，不再作为当前成果入口；不移动固定路径，避免破坏脚本与证据引用。

| 想看什么 | 当前入口 | 内容与状态 |
| --- | --- | --- |
| 已做什么、证明／证伪了什么、改进有没有用 | [RESEARCH_REPORT](RESEARCH_REPORT.md) | 主报告，8项统一六栏 |
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
| 原始协议、运行恢复和失败 | [实验README](../../exp/onevision_diagnostics_20261016/README.md)、[修复队列](full_input_repair/queue_state.json) | 全量cap1／cap3仍待完成，失败保留 |

`INTERIM_FINDINGS.md`、`MECHANISM_PROGRESS.md`等保留为过程快照，不能覆盖最新完成证明；`research_status.json`已同步本次核验状态。当前机器状态需同时核对真实PID与日志；单独状态文件不证明进程活着。

`KEY_FINDINGS.md`保留机制台账；其数学性质、实际观察和性能因果分别标注。所有诊断集均标明范围，不冒充完整21454题结果。

- [知识写入后的迁移核查](KNOWLEDGE_TRANSFER_AUDIT.md)：包含全部11个反向问答案例及正确／错配对照；词形差异仅待核查，不另造准确率。

- [完整确认已结束四域](FULL_COMPLETED_DOMAINS_AUDIT.md)：19331题，原生任务细项及既有400题／其余18931题分开；交通仍运行，不称全量全部完成。
