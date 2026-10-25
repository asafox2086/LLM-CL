# 补充说明：证据索引、实验历史与新会话交接

更新：2026-10-25。主结论与所有成果摘要集中在[README](README.md)。这里保存研究口径、原始来源、已关闭分支、实现故障和费用边界。本文不是新的实验或质量结果；原[README历史归档](README_HISTORY_20261024.md)保留时间线（不复制QA／图像样例，原件私有保留），历史的“下一步／正在运行”不是当前授权队列。

## 2026-10-25接手续况

人类已重新授权科研，旧小时自动跟进仍暂停，未自行恢复。最新直接FFN删除已完整结束，实际减重／固定GPU峰改善但五域质量下降；完整原始产物和CP已从OLD scratch转存NEW private，并核验pruned SHA。见[结果及全部实施／修复成本](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md)和[归档收据](summary/ffn_direct_ablation_20261025/v1/ARCHIVE_RECEIPT.json)。

OLD账户701GiB／硬配额700GiB，即使整盘2.5TiB空闲也不可继续写入；未删除旧数据。stage-refresh原进程已因配额故障退出，无完整VERIFIED，旧queue运行标签失真。后续不要等待该PID或消费partial质量。新机Qwen两个基线的实际PID和预测文件仍推进；未新增完整正式证明，因此主表保持8/16。正式表原行和权重列未改。

2026-10-25补证：37次同FIT联合删除NLL完成，确认本固定集合损伤超出独立单删累计；没有新F1或压缩方案，原构造仍关闭。原始标量、两次0模型准备与Git换行检查失败后误激活的实施偏离均见[完整记录](summary/ffn_joint_effect_20261025/v1/FINDINGS.md)。

2026-10-25小视觉接口已完整核验；全部质量/资源及两项CPU检查、资产环境失败、权重/原回答归档和成本边界见[功能承接结果](summary/vision_interface_transfer_20261025/v1/FINDINGS.md)。不恢复旧Smol整模/视觉尾/半宽或语言删层分支。

2026-10-25独立词面证据诊断完成：自然域两图均受益，父子差距扩大，固定路线关闭。两次预检查失误、首次真实加载后QA前路径故障、经两agent接受且保原期限的盲态实施补正与全部费用见[新完整记录](summary/lexical_evidence_probe_20261025/v1/FINDINGS.md)。这不是恢复原小视觉图训练或真实OCR部署优势。

2026-10-25紧凑表示先行完整对照结束：原87步能学习但仍未保质，五域宏37.47低于父43.79，四域低父。图权重−34.96%、实际文件−19.73%、5GPU峰均低；CPU RSS高水上升和CUDA逐key导出可能重复共享载荷均公开，未热修源或重跑。固定表示＋87步配方关闭，不加步／解冻／改桥；详见[完整结果与成本](summary/compact_first_learning_20261025/v1/INTERPRETATION.md)。普通低位骨干＋高精度更新提议经文献和本地证据评判未立项，不重开旧量化分支。当前无新已准入模型，科研目标仍未完成。

2026-10-25固定域末功能读数完整完成：S19零生成恢复精确匹配封存图后，S1／11／15／16各20DEV唯一新评分。四域末均低父末态目标；自然+8.69／遥感+20为后续净收益，医学−0.67／金融−5局部损失。不能把现缺口全归域后遗忘，域末不等于严格刚习得，不自动选择记忆方法；详见[全部证据与边界](summary/domain_end_function_20261025/v1/INTERPRETATION.md)。900秒预算内实际226.63模型秒，0训练／NLL／TEST；旧固定训练点不恢复，尚无新候选。正式基线推进但无新封存，8/16不变。

2026-10-25紧凑载体更新分配联合对照完成：新all宏39.262042，旧紧凑wide 37.467196，完整父43.791987；未通过父各域质量底线／资源联合标准，关闭固定all-r8配方。285新生成／280唯一评分、1213FIT／87更新／1213次backward，0TEST／NLL；四域末与最终全保留，模型总1534.471821秒，CPU提取25.877585秒及历史成本另计。跨表示有历史训练硬件混杂，原协议和新质量前边界修订并存；不称纯交互或既有all-r8原创。[完整证据](summary/compact_allocation_20261025/v1/INTERPRETATION.md)。

2026-10-25 GPU回答监督pair未形成质量结果：仅原CE完成320backward／20更新、278.787001秒；原KD构图erfinv报错、唯一盲态修订在all_buffers_digest匹配处失败，0新生成／评分／TEST。原KD与修订CE起止观察代理7.704807＋15.914976秒另计，保守三段和302.406783秒；GPU算子检查3.446713秒及元数据I/O／运输／实现另计。不把缺测称科学阴性，不复用旧CE或再修加载。[完整终态](summary/poststream_response_transfer_20261025/compatibility/FAILURE_FINDINGS.md)。

下面为2026-10-24交接原记录，原“当前／活动／下一步”均按历史时点理解。

## 新会话运行交接（2026-10-24历史）

### 阶段刷新原运行（现已故障终止）

- 题目：OSFT阶段刷新整包是否贡献有限步数下的习得；不是继续压矩阵或优化一条强基线作为主方法。
- 代码：`exp/osft_stage_refresh_mechanism_20261024/v1/{core,common,worker,freeze,dispatch}.py`。
- 权威记录：[协议](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/PROTOCOL.json)、[READY](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/READY.json)、[准入](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/ADMISSION_REVIEW.json)、[ACTIVATION](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/ACTIVATION.json)、[启动快照](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/PHASE_OBSERVED.json)。
- READY SHA：`d276508a82e51855b3c7e4ac94f8fa3329167c7bec6cf190368c09f00271d4a5`。五源／协议冻结，不热改。
- OLD隔离目录：`/data2/liyapeng_grp/program/LLMCL/.deployment/osft_stage_refresh_mechanism_20261024_view`；unit `llmcl-onevision-osft-stage-refresh-mechanism-20261024.service`。
- 最新[交接快照](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/HANDOFF_OBSERVED.json)：2026-10-24 15:30:05 UTC，同boot `5b259bd9-d0cb-4a1d-8b90-6e6107080488`，controller3728445/start755358934、Fixed worker3731901/start755415692、UID1084、PGID各自同PID、物理GPU0，实际身份匹配。prefix已COMPLETE且EXIT，Fixed已STARTED运行，Refresh未STARTED，无VERIFIED／故障；未读partial质量。原prefix worker3728466/start755358964仅启动历史。后续核真实/proc/startticks/queue/proof，不照抄永久live。
- 实际首次STARTED unix1792855169.5632505。prefix最多3600秒，Fixed／Refresh各7200，总18000模型秒；包括加载、重复模板SVD再覆盖共同状态、全部训练／480生成／观察／保存重载。不是用户时限或1%定义。CPU准入、运输、核验、历史另列。
- controller保持单流同卡，仅真正空卡可首次启动。Restart=no／KillMode=process／不enable；看不到worker字段不等于故障。只自身UID/PID/startticks/PGID完全匹配才可按冻结watchdog回收自身隔离进程，禁止向他人或正式GPU进程信号。
- 三case COMPLETE且实际成功退出后原dispatch唯一verify：480新原token_f1、153真实optimizer见证、输入／参考／状态绑定。不得读partial质量、手动再verify／评分／bootstrap。之后只消费证明一次报告与解释审阅。
- 首次真实STARTED后的故障／超预算／中断关闭固定点，不续权重、不加时、不自动版本／rankLR／额外QA。停止某固定点不等于停止整个科研，也不能因困难把goal标完成／blocked。
- 当前比较不减少完整权重、不证明全部OSFT优越性、没有自动紧凑方法或1313／21454训练授权。正负结果先改变是否需要阶段刷新功能这一设计判断。

### 本次实现准入中发生了什么

首次tiny检查将OSFT高／低分解切片误当独占连续存储，计数断言失败；是统计实现问题，不是模型质量失败。原5源、旧review、stdout／stderr／失败和费用原字节保存在[attempt1](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/cpu_admission_attempt1/SOURCE_REVIEW.json)，原私有tiny权重留OLD，不入Git。

仅修计数：按底层storage上的逻辑区间求并集；逻辑必需权重、backing存储和未用backing分列。重载严格值／digest不变，逻辑布局相同，允许保存重载前后backing空余不同。训练、数据、刷新、投影、评分和预算未改。[修复记录](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/CPU_COUNT_FIX_INTERVENTION.json)连接最初运输旧SHA与新准入源。

首检查监督4.321477秒；必要完成检查3.891517秒／监督4.721693秒重叠，不称首试通过。两次均无真实模型／QA。小图两臂共享S1权重／RNG／method.stage1精确，低因子继续更新，S2实际refactor0／14，完整重载通过。这些只是准入，不是科学发现。

准入发布`b061fe2145ad0cb71e014b76de66b4c080631e86`（30文件），启动发布`528cd6dd9d359e5138436260d2eefd8c647e5025`（4文件），均NEW main已推送并逐字节镜像。活动入口不会因新会话自动取消，既有小时跟进已更新到本实验；无实质变化保持安静。

### 权威位置和基本操作边界

- NEW Git main：`/data1/liyapeng/program/LLMCL`；本地`D:/锦/大学/科研/LLMCL`非Git镜像，勿在本地伪建权威历史。只白名单提交／推送／逐字节同步，保留无关修改。
- 已验证NEW SSH由`.deployment/confirmation_ssh_20261022.py`调用；OLD经`.deployment/old_sparse_exec_20261024.py`中转，后者共用relay文件，不并发调用。密钥位置沿既有私有配置，不复制到文档或Git。
- 当前自身观测脚本：`.deployment/stage_refresh_observe_old_20261024.py`首次启动已写不可覆盖快照；以后需要新名称保存新观察，勿重跑带`assert not PHASE_OBSERVED.exists()`的首次发布脚本。科学dispatch才是完整结果唯一入口。
- 与已授权`/root/review_next_probe`可实质讨论；不是把子agent的暂时不确定当禁止执行，亦不重复盖章替代探究。
- 人类要求“是什么就写什么”：成果表不按CPU／GPU拆成假实验；硬件、dtype、费用放此补充。训练状态下降不冒充最终模型缩小。

## 关键结果的证据入口

| 范围 | 原始完整入口 | 用途 |
| --- | --- | --- |
| 正式双骨干 | [MAIN_TABLE](summary/two_backbone_pct01_20261017/MAIN_TABLE.md)、[七范围索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json)、[FOREVER独立证明](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md) | 原1313／21454正式结果；74个必要评测格只适用于正式完整轨迹 |
| OSFT／PASs功能与成本比较 | [完整分析](summary/edge_deployment_20261019/sealed_baseline_tradeoffs_20261021/FINDINGS.md)、[机制账本](summary/final_weight_reduction_20261023/mechanism_to_design_v1/RESEARCH_LEDGER.md) | 刚习得／保持与任务差异，不构造未知因果 |
| 最终权重8组 | [分类／参数／字节](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)、[完整EVIDENCE](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/EVIDENCE.json) | 只读已保存对象图元数据；未进行新推理或新折叠 |
| 探索四组 | [qv／OSFT／all／extra](summary/edge_deployment_20261019/osft_increment_screen/v1/RESULTS_20261022.md) | 1213 FIT／100 DEV，87更新；不与正式分数混排 |
| all完整确认 | [FINDINGS](summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1/FINDINGS.md) | 正式数据完整质量阴性，禁止看TEST追参 |
| 宽qv／保护／ShortFixed | [同坐标](summary/edge_deployment_20261019/position_coordinate_control_20261022/v1/FINDINGS.md)、[保护包](summary/edge_deployment_20261019/osft_protection_control_20261022/v1/FINDINGS.md)、[固定短侧](summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2/FINDINGS.md) | 不更新、前向删除、完整轻量三类区别 |
| 冷启动完整证明链 | [COLD_START_PROOF](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/COLD_START_PROOF.md)、[同域／跨域五格](summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/CONTROL_OUTCOME.md) | 仅局部新专家，不是所有方法清零 |
| 首步梯度控制 | [函数中性](summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md)、[首步A](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/first_factor_gradient/RESEARCH_FINDINGS.md) | 开放梯度不等于完整继承 |
| 低分回答行为 | [FINDINGS](summary/final_weight_reduction_20261024/response_failure_profile_v1/FINDINGS.md) | 已见正式TEST仅分析，不据此挑新结构／提示 |
| 金融输入合同 | [FINDINGS](summary/final_weight_reduction_20261024/finance_input_contract_v1/FINDINGS.md) | 所查字段／选项未错位，4项strict解析仍未知；不补称历史逐token一致 |
| PAS保护机会 | [FINDINGS](summary/final_weight_reduction_20261023/pas_rs_coverage_v1/FINDINGS.md) | 40结构零／53资格上界的条件重建，非实际有效保护／F1因果 |
| S0分差核对 | [FINDINGS](summary/final_weight_reduction_20261024/osft_initial_gap_check_v1/FINDINGS.md) | 排除初态常数分差叙述，不是因果效应 |
| 图文支持干预 | [RESULT_FINDINGS](summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/RESULT_FINDINGS.md) | 同OLD四状态488NLL，未支持OSFT整体支持增幅更高 |
| r200双完整流 | [FINDINGS](summary/final_weight_reduction_20261023/answer_guided_pilot_v1/FINDINGS.md) | 实际少25.83%权重，质量失败 |
| gate／up保留两流 | [README](summary/final_weight_reduction_20261024/ffn_role_retention_v1/README.md) | 同大小、同适应，起点已损伤，不能归为单纯遗忘 |
| up复用／拟合 | [OUTCOME](summary/final_weight_reduction_20261024/gate_response_reuse_v1/OUTCOME.json) | 局部误差下降未保完整回答 |
| D功能读出 | [README](summary/final_weight_reduction_20261024/functional_readout_carrier_v1/README.md) | 保G/U也未使r200 D替代保质 |

## 已关闭、缺测及未执行分支

| 分支 | 实际做到哪里／终态 | 后续边界 |
| --- | --- | --- |
| PASs FIT v1 | 外部重启中断46条更新，13完整CP；早期optimizer RAM见证丢失 | 不拼轨迹或补造历史，不续旧CP |
| PASs FIT v2 | 4小时超时，77／87更新、17CP、1074已记录FIT，0DEV／VERIFIED；最后inflight未知 | 质量缺测，不判ShortFixed胜败；不延时／新v3。[封存](summary/edge_deployment_20261019/pas_fit_reference_20261022/v2/BUDGET_FAILURE_FINDINGS.md) |
| Shared fullspan | 教师特征捕获触发8GiB载荷门，尚未分解／恢复／压缩质量评价 | 资源估计失败，不是共享稀疏功能阴性；不扩cache救跑。[故障](summary/final_weight_reduction_20261024/shared_fullspan_carrier_v1/FAILURE_FINDINGS.md) |
| 固定稀疏连接 | 首个剪后打包有非有限值，无初态质量门或Frozen／Plastic结果 | 旧入口关闭；统计上下文修复仅人类决策提案，**未获明确例外**，不能借一般授权执行。[故障](summary/final_weight_reduction_20261024/sparse_connection_plasticity_launch_repair_v1/INITIAL_FAILURE_FINDINGS.md) |
| Smol原生起点 | 已加载权重，但首题processor的2048与1820尺寸契约冲突，0生成 | 不能算小载体质量阴性；不换配置／型号自动救跑。[故障](summary/final_weight_reduction_20261023/native_initial_answer_baseline_v1/FAILURE_FINDINGS.md) |
| native E/Q/R | v1视觉后端构造失败；v2小图精确梯度比较失败，真实模型未运行 | 不放宽容差或自动v3；不是动态选择方法质量结论 |
| 视觉尾13→1 | 真实模型0 STARTED，方向讨论后关闭 | “基线冻结视觉”不能推出尾层无用，不恢复 |
| 普通factor r540+qv | 仅小图准入，0真实STARTED，按人类框架目标关闭 | 不是质量阴性，不自动恢复 |
| 全局输出偏置 | WIP实现／旧必要性讨论后方向关闭，0新模型 | 保全部父权重，缺紧凑承接，不因代码写了就启动 |
| 其它旧固定点 | NormOnly、宽qv、去保护、半宽、K69、implicit QR/core、能量／内存/header、共享输入投影、gauge／beta等均已封存 | 不重复谱／计数／梯度／旧评分或轮流换分解工具维持活跃 |

有先例不是关闭理由；实验质量失败、没有用途的干预、或已经明确的固定出口才是。关闭固定条件不表示否定整个技术家族。没有模型运行和没读质量的情况分别标缺测／未执行，不写0分。

## 成本、存储和解释边界

### 已完成主要模型工作的费用摘要

| 工作 | 已记录模型运行秒 | 已包含的主要额外工作 |
| --- | ---: | --- |
| ShortFixed v2 | 1887.158 | 1213 FIT／87步、19保存、38NLL+38短生成门、100DEV；更新869.379／保存492.165为重叠子区间 |
| PAS FIT v1 | 启动到重启约6231.327，非完整RUN | 46已记录更新／605FIT加未知inflight、13CP；更新5243.445／保存448.149重叠 |
| PAS FIT v2 | STARTED到FAILURE文件mtime14404.668，非完整RUN | 77更新／17CP、门槛，保存1075.757；原seal误记save字段已由独立correction更正 |
| r200两完整流 | 2927.04 | 两流各1213FIT／87步／100DEV、19归并、额外414门曝光 |
| FFN角色两完整流含共同初态 | 3067.093059 | 2426FIT／174更新／500生成、完整保存重载 |
| gate-response复用 | 1072.496211 | 48教师请求／200DEV、两臂各1536block更新／196608样本状态呈现 |
| D-only功能读出 | 1238.165081 | 48教师请求／200DEV、统计分解与保存重载，0训练更新 |

这些不是同计算预算的速度比较。实现、准入、运输、等待、SSH、唯一核验、报告／审阅和历史失败另外计费；监督／退出观察与内部RUN重叠，不能相加。费用细分仅消费原封存，不重跑或重复全文件hash。

### 完整权重与运行内存

完整权重必须包含冻结基、专家、路由和必需权重buffer；共享storage去重。固定模型的P_total(t)是水平线，构造时跳变与阶段增长分开。可训练参数不含冻结基，不能用`nn.Parameter`数量漏掉OSFT高基buffer；普通buffers、可重建cache、教师和训练历史另列。

保存态字节不是checkpoint文件大小，也不是RSS／allocated／reserved／NVML／推理峰。模型dtype不同会改变字节但不改变参数数，禁止以精度／量化／算子卡常作为创新。所有参照允许的常规fold公平适用；仅将OSFT分解折回同稠密形状不能称结构压缩。

ShortFixed的完整USV不减；梯度／Adam少21.34%仅训练辅助。它相对普通LoRA_all仍有约20.34倍训练坐标，不等于整机内存倍数。已有阶段末采样不是推理峰，不能从小增长判断长期有界／泄漏或分配器因果。

## 已完成理论／源码／文献判断的边界

- 单阶段完整补基下ShortFixed固定短侧与原允许矩阵集合相同，不代表Adam有限步、FP16路径或后续高基轨迹相同；数学不需重证明。
- 梯度能量小、当前Taylor小、低秩谱集中、路由少或参数变化小，均不足以认证永久删除后仍保功能和可学性。局部输出MSE也不能代替全图Answer F1。
- 同层gate/up共享输入投影有SkipCat等先例；输入基共享、独立输出系数并不自动冲突。两套独立输入基与共享基不能天然同时满足同功能起点和同完整预算，旧代数／预算已结案。
- 合并两专家的共同项仍可能保留2r因子量；一般gauge保持BA不保持A能量路由，补metric须计存储。native两组动态缩放／beta不直接减父权重，旧主方法依据已关闭。
- 共享词表缩宽需计输入／输出投影及桥；只一份完整词表加两端映射会增加2dr，不能当词表压缩。SliceGPT函数结论限其架构和理论算术，不称FP16逐值相同。
- HOFT／LoRA-FA／SVFT／RandLoRA／FiPS／ZipLM／SkipCat／动态秩等可作为工具；我们没有因为读了论文就获得方法贡献。NIRVANA当前版本有条件限制，旧解析小量族不应包装成推翻论文或证明本模型权重必要。
- 论文中5%恢复通常是约3.3万样本，不保证本1213 FIT有限持续更新足够恢复。标准预训练小模型的作者能力不算我们的增益。

## 正式基线和不可重复工作

最后一次正式证明检查2026-10-24 14:08:04 UTC，七范围索引仍7／14，加独立FOREVER一组共8／16。仅新sealed证明才运行`analysis/v2_forever_main_table_20261020.py --write`，必须保留FOREVER与8组已测保存态存储列。无新proof不重复pending发布、不热改publisher、不从旧PID猜当前任务完成。

19任务final-only研究不是正式74格；没有阶段矩阵不能补FWT／BWT。历史AnyRes单LoRA21454题AP41.361等结果也不与V2公共图像协议混排。旧155／507诊断、旧数据量轨迹和队列名称不代表当前准入；全部保留在原证据与历史README，不恢复自动队列。

## 文献分类与数据组成

以下从旧README整理保留原文献、数据来源与核查索引，仅作为资料；日期和“当前”措辞属于其历史背景。论文发表与数据量来自既有记录，本次没有重新检索、评分或模型实验。

### 方法 × 方向二维表


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
</tbody>
</table>

归类按主要机制，混合机制、原论文模型及生成任务见 [论文完整索引](paper/README.md)。预训练条目包含语言／知识／上下文扩展型 CPT，不全部具有序贯抗遗忘实验；HPA 归入持续安全对齐保持。**正式回放比较保留FOREVER（ACL2026）；标准ER-LoRA已移出，DGG仅保留文献。**

## 当前五数据集／19原生任务协议

当前正式框架为`exp/two_backbone_native_cl_20261017_v2/`，结果在`summary/two_backbone_pct01_20261017/`；冻结1%清单为`data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json`。旧`exp/five_domain_native_cl_20261016_sparse/`及其2026-10-16启动快照是历史版本，不作为当前运行状态。磁盘写满后的故障保全、真实S9恢复与ER新run重跑见[恢复记录](summary/storage_recovery_20261018/README.md)。

两骨干为Qwen2-VL-2B-Instruct与LLaVA-OneVision-Qwen2-0.5B-OV。相同1313条TRAIN、21454条TEST、19阶段、seed42、一轮、有效batch16/microbatch1，无验证集；公共图像预算401408像素及128MiB CPU缓存。各方法保留已披露的作者差异，同方法跨骨干共享实现。这些硬件与调度描述来自旧方案；当前实际任务和资源须重新核真实身份，不能由历史分工推断。全量版本已就绪，不把数据就绪写成全量实验已经完成。

当前7项已准入方法：SeqFT-LoRA、CL-MoE、SMoLoRA、PASs-MoE、PaRSP、OSFT、SpaRTA。用户移出尚未正式训练的FV-guided Training与ALER，历史预检和文献仍保留。FOREVER已独立适配准入，其中OneVision完整结果已入正式表；Qwen尚无完整核验结果。共8项方法、两骨干16个目标，不表示全部完成。[方法发表信息与状态](summary/two_backbone_pct01_20261017/BASELINES.md)。SeqFT-LoRA是标准顺序微调对照的描述名；标准ER-LoRA已移出当前比较，CL-MoE为CVPR2025多模态方法。Hyper-LLaVA移出本轮，DGG仅保留文献。主要机制归类与LoRA参数化不互斥，例如PASs既有专家架构，也有LoRA参数正则。[V2适配与准入](summary/two_backbone_native_cl_20261017_v2/README.md)。

顺序为TextVQA → SLAKE的Modality/Position/Organ/Size/Abnormality/Quantity/Plane/Shape/Color/KG → RSVQA-LR的rural_urban/presence/count/comp → FinMME → CODA-LM的general/regional/suggestion。原生标签保持，顺序是研究预先冻结课程，不称官方课程。1%划分两机、所有方法完全共享，不重新抽样。

唯一Acc为**原始Answer F1（0—100）**，不翻译、不映射选项、不额外抽取答案；不并列其他指标作为主Acc。[评分锁](rules/020_answer_f1_primary_20261017.md)。初始全19、中间当前+下一任务、最终全19，共74格/80316条预测；不报告历史峰值ForgetRate，缺格保持null。五领域最终分按各域QA加权，AP按19原生任务等权，二者不可混用。统一Answer F1不是FinMME/CODA-LM的原生官方评分。

正式V2公共输入与研究原生AnyRes/BF16分开报告；155生成题/500诊断不能冒充完整21454题或填补完整FWT/BWT。[正式执行策略](exp/two_backbone_native_cl_20261017_v2/execution_policy.json)。

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
| D2 · 医学 | SLAKE | [Bo Liu等，ISBI 2021](https://www.med-vqa.com/slake/) | 9,834 | 99 | 2,094 | 官方train/test；仅排除用户授权的空答案SLAKE:1622；双语；val2099不用 | Uncertainty-Driven Expert Control（ICCV 2025） |
| D3 · 地理／遥感 | RSVQA-LR | [Sylvain Lobry等，TGRS 2020](https://rsvqa.sylvainlobry.com/) | 57,223 | 573 | 10,004 | 官方active掩码train/test；val10005不用；不自行分箱 | GeoChat（CVPR 2024）、LHRS-Bot（ECCV 2024）、RSAdapter（TGRS 2024） |
| D4 · 金融 | FinMME | [Junyu Luo等，ACL 2025主会](https://aclanthology.org/2025.acl-long.1426/) | 8,864 | 89 | 2,233 | 源池11099；2条缺答案单列；11097有标注QA按原图SHA256分组seed42研究80/20，非官方划分 | Beyond Real Weights（WACV 2026）实际评测；AgentFinVQA（2026预印本）实际评测 |
| D5 · 交通／自动驾驶 | CODA-LM | [Kai Chen等，WACV 2025](https://github.com/DLUT-LYZ/CODA-LM) | 20,495 | 205 | 2,123 | 作者English LLaVA Train/Test；源卡20495/2123；Val18455、Mini193不用 | MPDrive（CVPR 2025）实际评测；RoboTron-Drive（ICCV 2025）实际训练和评测 |

五数据集合计：有效全量训练 **131,018**（原131,019条仅排除空答案SLAKE:1622），1%训练 **1,313**、完整测试 **21,454**均不变。[排除授权与逐行核验](summary/two_backbone_native_cl_20261017_v2/FULL_MISSING_LABEL_EXCLUSION.json)。拆成19阶段不增加或重复训练样本；SLAKE原全量9,835条、有效9,834条/1%99条是十类合计，最小1%类仅1条。原始源文件不改。
FinMME的ID3698、9806金答案为空，原记录单独保留审计，不伪造、不以空串监督。作者仅发布名为train的评测池；本研究按图像字节SHA256分组，组哈希seed42排序，20%组留作测试。原选项作为输入，verified_caption/related_sentences只作来源元数据，不输入模型，避免泄漏答案。
1%组按源题型分层，最大余数分配ceil(1%×各数据集训练量)，seed42哈希选ID；沿用父实验已冻结的ID，再按原生任务互斥拆分，不重新抽样。每阶段ID、顺序和测试SHA256事先冻结在私有`split_lock.json`；两服务器、所有方法共用且启动核验，文件只读。私有`partition_audit.json`中的19阶段数量。不自行删掉选择、分类、是非题。CODA-LM保留作者原问题、答案和区域红框图。

## 多模态依据

五个来源都有真实图像和问题文本，模型同时接收两种输入。**图文任务身份不等于每一道题都必须看图**：SLAKE知识题及语言偏置可能允许捷径。原始构成照常保留；研究诊断包含固定的原图/遮图/错配图视图，证据见[问题发现总报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md)，小诊断不能证明所有任务/每道题均依赖图像。

| 数据集 | 视觉来源与任务 |
|---|---|
| TextVQA | 自然照片＋问题：读取场景文字并结合图像作答 |
| SLAKE | 医学影像＋中英问题；保留视觉题、知识题及OPEN/CLOSED |
| RSVQA-LR | 遥感图像＋问题：计数、存在性、比较与区域判断 |
| FinMME | 真实金融图表＋问题、选项；单选、多选和数值题全保留 |
| CODA-LM | 真实道路图像＋作者指令：场景理解、区域理解、驾驶建议；区域题保留作者红框图 |

原始发表、独立复用正文与限制见[六域多模态核验](data/MULTIMODAL_SIX_DOMAIN_AUDIT_20261016.md)。医学首发ISBI、遥感首发TGRS，不把后续顶会复用写成首发顶会。

## 数据正文的保存边界

原QA、原图与数据划分正文保存在服务器私有数据树，本补充不复制样例或制作无效下载链接。旧README完整原件仅私有留存；公开历史副本省略其样例段。相对路径`data/FiveDomainCL/five_domain_native_cl_20261016_sparse/`只说明冻结数据位置，不表示这些私有文件已进入Git。
