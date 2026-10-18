# 当前研究入口：先区分冷启动与跨域复用

当前以[五格初始化来源控制](initialization_source_controls_20261018/v1/PROTOCOL.md)为主线：同一Organ目标26条TRAIN／2次更新，比较cold、同域／跨域完整继承、同域／跨域A-only，五组最终均测原507题。源历史已实际完成，冷启动目标对照已开始；五组最终507结果仍待，不新增路由或扩容模块。

| 入口 | 用途 |
| --- | --- |
| [唯一研究调度](initialization_source_controls_20261018/v1/queue/README.md) | 实际状态读该目录queue_state.json及进程；以下历史PID不作为当前调度依据 |
| [五格结果](initialization_source_controls_20261018/v1/VERIFIED_RESULTS.md) | 只接受封存证明；未完成留空，不表示零收益 |
| [问题发现总报告](RESEARCH_REPORT.md) | 问题、实验、结果、证明、改进、效果及局限 |
| [不依赖跨域旧LoRA的初始化文献](expert_cold_start_20261018/CURRENT_TASK_INITIALIZATION_PRIOR_ART.md) | EVA NeurIPS2025、CorDA NeurIPS2024；已核验已有工作，未在本研究实施 |

正式V2公共完整确认已按最新用户要求后移，源码和原作业保留；两机正式基线继续后台运行。原生AnyRes／BF16研究与正式V2结果不混排。原生连续19阶段/93更新及四视图现已完整核验，缓存几何AP45.700、顺序43.363、质量42.860，域宏区间均含0；[六栏、同题NLL反例及全部区间](functional_state_cache_20261018/v1/first_four_view_outcome/RESEARCH_FINDINGS.md)。新五格仍须各自完成507题，不把诊断当完整提升。

## 历史更新与证据导航

以下保留原始更新。带“最新”“当前”的旧文字及PID只反映当时快照；现行入口以上表为准，旧已退役调度不得恢复。

# 2026-10-18最新入口

新冷启动精确桥接：三专家训练量1214/11/88；e1只训练一步，随后17阶段权重完全没变，不是后来被覆盖。对应同11TRAIN/一步/228TEST控制，从基座新建F1 19.876，继承完整旧自然LoRA29.263（差+9.388，图像区间[4.559,14.330]）。它不是nativeSeq权重，不能拼主表；仅定位初始化/旧功能未继承的局部问题，未证知识唯一原因或整体热启动方案收益。见[六栏和限制](expert_cold_start_20261018/RESEARCH_FINDINGS.md)。

最新完整四视图诊断已核验：同155题AP，原调用31.025→质量硬36.704／几何软35.996／质量软36.387，仍低于顺序43.363。完整21454候选已启动、尚未完成；正式V2主表仍只填已完成SeqFT/ER。见[五域主表格式与边界](functional_router/MAIN_TABLE_DIAGNOSTIC.md)。

2026-10-18最新调度：同nohup服务新CPU356616接续GPU348381自然完成，两个预定完整21454QA候选提前到四视图之后；原48/旧完整测试/其余全部任务保留，基线不中断，不按TEST挑赢家。18策略/4继承门槛通过，尚不代表收益。实时入口见[functional_confirmation_queue](functional_confirmation_queue/README.md)。

新增[固定池TRAIN调用与仅头逐图排除](functional_router/calibration_audit/RESEARCH_FINDINGS.md)：43条记忆、16参数实际拟合、44次CPU分析；与完整测试收益严格分开。

[总报告第13项：从文献机制到小数据轻量改造](RESEARCH_REPORT.md#13-借鉴多领域方法能否修复相似却未必会答的调用)。原学习相似度两组训练已完成但未胜顺序参考；[真实结果](learned_reuse/INTERIM_RESULT_20261018.md)。新固定池质量校准＋连续残差已接入[当前队列](functional_router_queue/README.md)，GPU效果待测；[完整协议](functional_router/PROTOCOL.md)。共同T04起点和固定指令全格已有新结果，见K22，不沿用历史“未测”状态。新机磁盘事故及基线续跑见[恢复报告](../storage_recovery_20261018/README.md)。下文为保留的原证据索引。

> 当前唯一主指标与结论先看[Answer F1诊断重审](PRIMARY_F1_REASSESSMENT.md)。[冻结规则](../../rules/020_answer_f1_primary_20261017.md)明确无选项映射。旧原始EM/ROUGE记录仅追溯。

# OneVision痛点研究：阅读入口

**先读[问题发现总报告](RESEARCH_REPORT.md)。** 按14个问题统一写为：探究的问题 → 做的实验 → 结果 → 证明了什么 → 改进的方法 → 改进的效果。首页区分已证实、被反例否定与未解决；未完成、负结果、局部有效均明确标记。

本文导航于2026-10-17整理。旧快照保留用于追溯，不再作为当前成果入口；不移动固定路径，避免破坏脚本与证据引用。

| 想看什么 | 当前入口 | 内容与状态 |
| --- | --- | --- |
| 少更新关联是否就是原因、当前任务能否伤害开放功能 | [7方法共同13任务／六栏](native_acquisition_20261018_v3/RESEARCH_FINDINGS.md) | CL-MoE当前OPEN−5.707，后续领域未发生；总体7/7关联在医学3方法反转，不作纯步数因果；V3排除损坏资源，不放宽评分 |
| 正式基线学会后丢了什么、回放存储是否不足 | [新8个封存格／六栏](baseline_language_followup_20261018/RESEARCH_FINDINGS.md) | ER OPEN−12.129区间为负但全均值区间含0，池0容量淘汰；PAS强增主要闭式且资源大，Qwen总分不代表OPEN |
| 同学习历史下四域换五域会怎样 | [前三阶段精确匹配及全507对照](matched_task_domain/common_prefix/VERIFIED_RESULTS.md) | F1差−1.430区间含0，NLL+0.157908区间为正；中文245/254题两边F1均0，不能据此推领域数因果或两模型等价 |
| 学习输入相似度、优先复用LoRA是否有效 | [同155题实际结果](learned_reuse/INTERIM_RESULT_20261018.md)、[核验结果](learned_reuse/VERIFIED_RESULTS.md) | 两组19阶段完成，学习AP31.025低于顺序43.363；均用满3模块，完整确认0/2 |
| 最新唯一研究队列 | [质量校准与软聚合CPU340143](functional_router_queue/README.md) | 旧336769仅CPU退役、原GPU338128自然结束后接续；全部旧任务保留，新四视图与2完整候选待测 |
| 文献机制如何适配小模型／1% | [固定池轻量改造](functional_router/PROTOCOL.md) | 43TRAIN/172NLL、16参数头＋单专家连续残差，零新增LoRA；CPU通过不是GPU提升 |
| 别人发现了什么问题、怎样找原因 | [文献问题综述](LITERATURE_PROBLEMS.md) | 12篇正式论文+COAST版本；先看7类问题，附证据/方案/适用边界 |
| 跨域差距是否混入语言变化 | [完整507题语言/题型审计](matched_task_domain/language_profile/T04_VERIFIED_RESULTS.md) | 中文差−21.102/英文−9.960pp；TRAIN中文曝光17/0，不能纯归因知识遗忘 |
| 固定权重能否靠输出指令恢复 | [四指令干预](language_cue/PROTOCOL.md)、[结果](language_cue/VERIFIED_RESULTS.md) | 所有完整8格已测，跨域中文部分恢复但英文/同域中文有损害；不按TEST选提示词 |
| 原语言干预接续 | [原语言因果CPU队列](language_causal_queue/README.md) | 已由新学习复用入口继承，所有4个固定推理进程保留；不恢复旧CPU277043 |
| 正式基线的低分是否已在习得时存在 | [最新学习前/后与保持](baseline_language_followup_20261018/RESEARCH_FINDINGS.md)、[全原生任务扩展](native_acquisition_20261018_v3/RESEARCH_FINDINGS.md) | SeqFT/ER最终已核验，另新增PaRSP/OSFT/CL/PAS即时学习；旧K19留历史，不与AnyRes拼接 |
| 平均旧loss改善是否保护全部旧功能 | [TRAIN子组梯度/有限变化](train_language_geometry/VERIFIED_RESULTS.md) | 4条固定TRAIN、11个已学转移中6次均值改善掩盖组退化；反方向/同域例也保留，不作F1或语言因果 |
| 已做什么、证明／证伪了什么、改进有没有用 | [RESEARCH_REPORT](RESEARCH_REPORT.md) | 主报告，14项统一六栏 |
| 首批跨域梯度是否已负冲突 | [精确共同起点TRAIN审计](matched_task_domain/first_transition_geometry/VERIFIED_RESULTS.md) | 三序列余弦均正，不能推断后续无冲突或有限F1因果 |
| 跨域差距是遗忘还是少了同域正迁移 | [完整共同起点/净变化](shared_anchor_initial/VERIFIED_RESULTS.md) | T04全507习得点20.361；同域后续+13.808，跨域−1.735区间含0；T02仍待 |
| 名义预算相同，跨域是否影响完整旧锚 | [首个跨域完整对照](matched_task_domain/FIRST_CROSS_CONTRAST.md) | 507题18.626/34.168、差−15.542；全锚S1缺失，不能全算遗忘 |
| 当前原48组怎样接续 | [共同起点补测与控制优先](causal_measurement_queue/README.md) | 补规范S1、两锚各4条SeqFT优先，全部48保留；新CPU265251，旧所有者退役 |
| 同样回放32次是否保护了同样旧题 | [TRAIN覆盖审计](matched_task_domain/replay_exposure/VERIFIED_RESULTS.md) | 已完成重复/同域旧锚曝光32/15；跨域仅计划数，不作性能归因 |
| 同域增加任务是否已经明确伤害旧任务 | [首个同域/重复对照](matched_task_domain/FIRST_WITHIN_CONTRAST.md) | 最终507题34.168/35.102，差区间含0；旧锚曝光不同，不纯归因任务数 |
| 没有跨域时保护会不会也出现限制 | [重复任务六控制](matched_task_domain/repeat_control_evidence/T04_VERIFIED_RESULTS.md) | 投影未触发/5状态完全相同；EWC后续真实改变，NLL差而F1差含0 |
| 严格48组中机制是否启用、是否有效 | [机制启用/性能并列](matched_task_domain/mechanism_activation/VERIFIED_RESULTS.md) | 新CPU观察器只接受完整轨迹，旧/新F1与EWC项、回放、投影和实际位移并列；已读取真实完成轨迹，数量随完成证明更新 |
| 完整LoRA相似是否代表新学习重复 | [真实Adam/BA增量](first_adam_geometry/VERIFIED_RESULTS.md) | 两任务总矩阵余弦0.977、增量0.205；原始梯度范数不是实际学习量，无F1因果声明 |
| 低GPU增长能否代表总容量不变 | [容量资源核验](capacity_resource_evidence/VERIFIED_RESULTS.md) | 固定GPU计数仍可对应1/2/3/5个CPU专家；成本与峰值范围全报，无内存优势声明 |
| 更多LoRA是否也改善选择 | [ROUTING_MECHANISM_EVIDENCE](ROUTING_MECHANISM_EVIDENCE.md) | 同池路线NLL对照；19原型/500选择跨容量相同，不声称F1提升 |
| 完整确认/小诊断能否外推 | [cap1完成与衔接](full_confirmation_evidence/CAP1_COMPLETION.md) | 全量表/155题队列差异/原型启动；同宏平均可掩盖域间误差 |
| 完整21454确认是否完成 | [自动结果入口](full_confirmation_evidence/VERIFIED_RESULTS.md) | cap1完整21454已核验，AP41.361；cap3待测，完整FWT/BWT为空 |
| 遥感全部同答：是否没看图 | [RURAL_COLLAPSE_EVIDENCE](RURAL_COLLAPSE_EVIDENCE.md) | 5权重×3视图×TRAIN／TEST全部完成 |
| 类别平衡有没有修复坍缩 | [RURAL_WEIGHT_EVIDENCE](RURAL_WEIGHT_EVIDENCE.md) | 三条轨迹全部完成；44→50，仍94题urban |
| 知识是否真正写入、是否帮助问答 | [KNOWLEDGE_FIT_EVIDENCE.json](KNOWLEDGE_FIT_EVIDENCE.json) | 正确／错配×rank1／8、0／8／64步全部完成 |
| 早期知识对照的格式／内容混杂 | [KNOWLEDGE_FORMAT_AUDIT](KNOWLEDGE_FORMAT_AUDIT.md)、[KNOWLEDGE_MECHANISM_AUDIT](KNOWLEDGE_MECHANISM_AUDIT.md) | 原17条件与事实池边界；不与新64步结果混算 |
| 当前任务先没学好，还是后来忘掉 | [初始/1/4步桥接](acquisition_budget_bridge/VERIFIED_RESULTS.md) | 同228题，首步NLL降但F1不见同步收益；多训局部改善并保留反例 |
| 新模块实际改变什么 | [T02四组](two_module_prototype/FIRST_T02_ACTIVATION.md) | 强度/步长均1，状态/结果相同，无新增收益 |
| 真实训练内存峰值 | [TRAIN窗口](two_module_resource_windows/VERIFIED_RESULTS.md) | bytes/重置峰值/预热差异，不据增长数直接比算法 |
| 多训与旧LoRA复用 | [COMPLETED_CONTROLS](COMPLETED_CONTROLS.md)、[FACTOR_UPDATE_EVIDENCE](FACTOR_UPDATE_EVIDENCE.md) | 9轨迹18终点与单步A不变证据 |
| 正则化什么时候生效 | [EWC_ACTIVATION_AUDIT](EWC_ACTIVATION_AUDIT.md)、[FIRST_STEP_CAUSALITY_AUDIT](FIRST_STEP_CAUSALITY_AUDIT.md) | 首步最新项0；不等于已证明首步有害 |
| 补齐多域／少任务因果缺口 | [确定性双锚协议](../../analysis/ONEVISION_MATCHED_TASK_DOMAIN_PROTOCOL.md) | 已有12完整轨迹，case18磁盘故障后独立重跑；全部48尚未完成 |
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
