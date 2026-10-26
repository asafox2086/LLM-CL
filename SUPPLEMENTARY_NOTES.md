# 补充说明：证据索引、实验历史与新会话交接

2026-10-26最新：10%双骨干SeqFT均完整核验，总2/16，Qwen宏52.853667、完整参数2210075136，两组最终模型已耐久保存；[第二组证据](summary/two_backbone_pct10_20261025/SECOND_COMPLETE_FINDINGS.md)。[位置诊断](summary/train_answer_prior_20261025/POSITION_FINDINGS.md)已结束、唯一核验并归档，不能由有限训练前有效样本推断学习导致位置敏感性变化。固定预算三臂已[接入原分配器](summary/conditional_shared_budget_20261026/ACTIVATION.json)，尚未STARTED；下文较早的1/16及等待位置诊断描述均为历史快照。临时清理仅197个可重建pyc/1364966bytes，主表记录完整保留。

更新：2026-10-25。主结论与所有成果摘要集中在[README](README.md)。这里保存研究口径、原始来源、已关闭分支、实现故障和费用边界。本文不是新的实验或质量结果；原[README历史归档](README_HISTORY_20261024.md)保留时间线（不复制QA／图像样例，原件私有保留），历史的“下一步／正在运行”不是当前授权队列。

## 10%第二主表：实际启动与归档边界

2026-10-26补齐两组FOREVER的[独立完整结果核验入口](analysis/pct10_forever_complete_verify_20261026.py)：复用原10%评分／日程核验，另核实际dynamic／final回放、来源、gate和rolling／S19完整图，缓存与native14分开。只在真实完整训练exit0后手动调用，不新增自动扫描或GPU任务；目前仅源码审阅，未实际消费／评分，正式仍1/16。旧1%“零dynamic”与固定memory数不能沿用；请求epochs与实际HF曝光分列，训练峰不冒充受控推理峰。[源码核对与执行边界](summary/two_backbone_pct10_20261025/FOREVER_COMPLETE_VERIFIER_SOURCE_REVIEW.md)保留首次执行前两处真实接口修正，不把实现补齐当方法效果。

实际接续已完成：先发布d4806d8b，29科学源／policy／BUILD及2监督源逐字节运输后，unix1792988577.2441113核新OLD CPU318427/start768699750/UID1084、unit `llmcl-pct10-baselines-dynamic-old-20261026.service`存在；仅旧CPU4049826退出，原Qwen Seq4046298、双PAS4049854/4048289、LLaVA CL236681及各wrapper身份前后全同，0 GPU信号。两FOREVER已由defer进入原空卡分配器waiting，尚无其GPU门或正式质量。原三个入口SHA不变，完整证明仍1/16。详见[实际接续](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_ACTIVATION.json)与[首次handoff](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_HANDOFF_old.json)。

原1% Qwen FOREVER属于独立nohup队列，不在native pending5中；unix1792989120.7491376实查10498/start131514/UID1004、boot c8bddf24…、原launcher命令与OWNER匹配，state为healthy_wait_for_free_baseline_slot且worker=null，没有NEEDS_ATTENTION／TRAINING_PROCESS_COMPLETE或训练日志。保留其原Qwen队列先清空再等GPU2的条件，没有新增副本／改调度。[实际独立waiter](summary/two_backbone_pct10_20261025/FOREVER_PCT01_WAITER_OBSERVED.json)是时点核实，不是永久live或正式结果。

首次运输已成功，外层wrapper却只接受nanosleep的wchan，20秒后在所有CPU／GPU信号前拒绝；只读复核原CPU仍同身份、实际为`poll_schedule_timeout.constprop.0`、无HANDOFF_INTENT／INPUT。随后仅补认该stdlib coordinator等待子进程的内核状态，原运输收据校验后复用，不重运科学源／启动模型；第二次完成实际CPU接管。该实施错误不是模型失败或科学结果，原[拒绝事实](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_PREHANDOFF_FAILURE.json)保留。新FOREVER完整结果消费者仍待实现，不由native14组合观察器冒认。

2026-10-26接续准备：独立FOREVER动态runner已构建的29项科学源保持原字节，新增两个coordinator目录内的stdlib监督源，把唯一2400秒GPU预检／独立重载接入原同一分配器。静态READY只封源／policy／协议；停止旧CPU后另生成真实queue摘要及controller身份输入，不因queue变化重写科学READY。只允许两项FOREVER从defer进入既定门，保全native；正式仍须从fresh训练，门失败不救跑，原collector已包含动态目录无需重建。当前此段只记录实施／准入，不冒称实际交接、GPU通过或质量；实际后续看收据。[具体调用与剩余完整核验缺口](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_GATE_INTEGRATION.md)。

2026-10-26更新：人类明确第二主表需两个backbone全部8类基线、与1%同样全部指标并保留最终模型；两处主表已补齐23列。首组LLaVA SeqFT已完整核验，五域宏47.462／AP45.874，真实参数／保存态存储已填；NEW最终pin保存完整CP，当前1/16。完整正负、原唯一评分／元数据费用与[首组证据](summary/two_backbone_pct10_20261025/FIRST_COMPLETE_FINDINGS.md)保留，1%原表不改。

最终模型留存包装经plain_r8_control源码／AST审阅接受后先发布b3056c64再实际启动NEW CPU服务；只替换旧归档CPU，0模型／新评分／GPU信号。混合cgroup路径检查首次失败在任何停止／启动前，修wrapper查实际mount后已成功启动；即时观察断言失败，未持久化具体false子条件，不臆测原因。之后只读同unit/PID1538724/start32618866实际active及无错误16组index／1组已存，补事实收据，不重启。见[归档规则与实际状态](summary/two_backbone_pct10_20261025/FINAL_MODEL_RETENTION.md)。下方0结果／旧collector PID均保留为历史快照。

人类已明确授权10% TRAIN＋完整TEST；实际冻结13105 TRAIN／21454 TEST／19任务／828次主任务更新，其余沿用1%设置。[正式协议](summary/two_backbone_pct10_20261025/PROTOCOL.md)与[独立主表](summary/two_backbone_pct10_20261025/MAIN_TABLE.md)保留采样、runner及参数／资源口径。

事前发布453f93a93a21d8042d27f472d2c1a1f6438d6901后已[首次激活](summary/two_backbone_pct10_20261025/ACTIVATION.json)。最新[实际快照](summary/two_backbone_pct10_20261025/LATEST_OLD_OBSERVATION.json)为unix1792938818.253909：OLD新controller4049826／UID1084 active/running，四组SeqFT／PAS均通过preflight及reload（return0），正式training phase处于initial_evaluation／S0，四卡利用率50–69%。其余8个native组合OLD排队、两SMoLoRA待NEW；0完整10%正式结果，不把初始评价当训练完成或新F1。

[控制器交接](summary/two_backbone_pct10_20261025/CONTROLLER_HANDOFF.json)仅替换CPU调度4044638→4049826，0 GPU信号；OneVision PAS原reload worker4049282已在unix1792938739.866自然exit0，早于交接，没有被中断，原三源与冻结PROTOCOL／READY不改。[FOREVER兼容发现](summary/two_backbone_pct10_20261025/FOREVER_COMPATIBILITY_FINDING.md)覆盖原初期14组合计划：两组明确等待10%动态回放适配核验，尚未启动FOREVER模型。旧core限制main_updates≤24而S1需217次，旧S2预检固定1次而当前需7次；不能仅删断言或复制1%零动态触发证明。

[NEW归档collector](summary/two_backbone_pct10_20261025/ARCHIVE_ACTIVATION.json) PID1318977 active，已收第一轮小证据，尚无CP。OLD RAM tmpfs不是耐久存储；NEW归档全量小证据及各run latest／final，不代表全部SeqFT 19阶段CP均已耐久保存。OLD原保留策略不变：SeqFT保19CP，其他native保S19，FOREVER原rolling；不以collector存活或一次同步代替具体CP归档证明。

## 最新研究重点与刚习得观测

2026-10-26按人类“探索需要高效”收敛：只保留固定总r8预算的共享、均匀、A响应soft三臂。独立评判淘汰需要额外W0前向的Hard聚类提议；均匀对照用于排除静态缩放/优化差异。新增独立GPU训练/重载worker，沿原1213FIT/100DEV/87更新探索条件，不改两正式表，不读取旧模型续训，不重建split；当前尚未排队或运行。每臂before/acquired/final各100及5固定资源窗，完整部署参数另核。见[协议](summary/conditional_shared_budget_20261026/PROTOCOL.md)、[评判](summary/conditional_shared_budget_20261026/DECISION_REVIEW.md)、[实现映射](summary/conditional_shared_budget_20261026/IMPLEMENTATION_MAP.md)、[文献边界](summary/conditional_shared_budget_20261026/LITERATURE_NOTE.md)。医学审计已闭合并推送bc72a12；不再扩展或重评分。

2026-10-26人类把目标明确为允许性能略低、显著内存优势优先，并要求从强方法抽象功能后用简洁机制实现，或从已证明共同问题设计自有方法；不能只选旧方法／折叠／删模块，不限既有方法。旧失败点保留终态，不因放宽偏好自动恢复；可重新陈述其已有质量—资源折中，但不把大域损失擅称轻微。全表最高正式锚是Qwen SeqFT48.972351，本身单LoRA；同骨干PAS46.619976／OSFT45.937438不代表全表最高。待检验的新方向是固定总预算内的条件分工，现无该方法新质量。

10%两个医学阶段下降的[有限配对含义审计](summary/train_answer_prior_20261025/MEDICAL_PAIRED_FINDINGS.md)已完整结束：491题／283原样相同／208变化；两agent盲标198同类10分歧，全部保留。Plane F1−0.718而原EXACT+0.862，明确同内容不同表达组贡献−1.005747pp；Position未知79项占总失分77.7485%，主因仍未识别，不据此开统一格式／知识模块。0新模型／评分／tokenize，私有原文及理由留工作区，公开只hash／标量。两个模型agent不是医学正确性真值，不因未知扩样／重判或重开旧语言提示实验。

2026-10-26全19任务的两比例比较已唯一执行，0新评分／QA／预测／模型／CP读取：同LLaVA／SeqFT刚习得19高0低0平，五域宏42.026058→49.996420，阶段内增量13高6低。10%医学Position／Plane阶段自身为−4.859471／−0.718391pp，但刚习得仍都高于1%；不能从更高末值称当前更新都更有效。两组q/v-r8同源容量不变，93→828是optimizer调用，10%822正LR／6合法零LR；不识别纯数据因果或同数值子空间，不扩为全部方法结论。[全结果、参数空间、来源及费用](summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION_FINDINGS.md)。金融多选零token F1非集合正确率，286单字母输出不等286语义全错；[已补明确解释](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md)，评分设置不改。

2026-10-26定向核对PIF目标：其错误内容负例混合了绑定正确／错配，作者本也说明两种反馈，未来整包收益不能单独归因绑定；纯一致性允许稳定选错，不替代正确性锚。原文Eq.(7)与Eq.(8)的负例方向不一致已目视核对，未取得可裁定实际实现的代码，不推作者实现错误。[完整推导、原始来源与设计影响](summary/train_answer_prior_20261025/PIF_OBJECTIVE_REVIEW.md)只排除不充分目标／归因，不是新的模型因果或成功框架。0训练／QA／评分，不改360诊断及两主表条件；独立审阅另存同目录。

既定360位置读出已准备实际单次执行接入：只在原空卡分配器第一次发现真空卡时占一个slot，四态串行／各1200秒，其他GPU仍原基线；不另起争卡allocator。源和现有输入合同不改，控制／释放与一次完整消费者代码经独立审阅，未运行模型或实际消费者；准入不冒称功能结果。接续前发现source_dir定位、子phase admission继承、last_attempted_worker退出核查，以及消费者startticks类型／每phase proof ledger范围，均在任何执行前更正，不改变科学条件。见[执行接口、预算与全部边界](summary/train_answer_prior_20261025/POSITION_EXECUTION.md)。

接入已实际完成：事前发布79b5260e，新OLD单一CPU342620/start768961721接管318427，原四GPU及各wrapper身份前后全同、0GPU信号；首快照slot=null，没有模型STARTED／响应或预算消耗，只是队列从计划变为实际可执行。四源／科学配置不变，未来首张真空卡由原分配器启动唯一controller，同卡结束再交回baseline；全证据[见实际接入](summary/train_answer_prior_20261025/POSITION_QUEUE_ACTIVATION.json)。冻结controller注释误称本report会被现有归档器镜像，实际白名单仅baseline10% report；公开／私有结果均待结束后显式NEW转存核SHA，不据此claim耐久，也不为注释改动热更排队科学源。

2026-10-26后续功能问题经research_review接受：45条原资格共同TRAIN、固定循环、四个已完成S15／S16状态，360新生成将检验内容选择对位置重排是否稳定。四个真实完整CP已只读stat确认存在，尚未载入；最长原context308，不触发原6000检索。没有新的split／训练／评分，worker及实际绑定待完成，不把准备当结果。[必要性与正负决策](summary/train_answer_prior_20261025/POSITION_DIAGNOSTIC_DECISION_REVIEW.md) · [具体协议](summary/train_answer_prior_20261025/POSITION_DIAGNOSTIC_PROTOCOL.md)。

实施现已备妥：原完整图恢复／90原生成的独立worker、45张图片实际SHA及两池共同对象、固定45视图严格逆变换、完整调用绑定均形成；源码及真实producer经root与baseline_review核对，0模型／生成／评分。尚未协调GPU执行或自动排队，没有功能结果／新训练准入。原metadata.options多余假设在首次执行前删除；root漏看run连续赋值造成的错误身份建议也在首次执行前撤回，原正确断言保留。[实施与全部费用边界](summary/train_answer_prior_20261025/POSITION_IMPLEMENTATION.md)。

实际运输／输入合同阶段另暴露两处元数据错误：Windows Path写Linux路径在任何写入／worker导入前被拒；修为POSIX后，stdlib真实输入检查发现45ID摘要编码应沿原compact JSON而非默认带空格JSON。两次都没有worker.main／STARTED／Torch／权重，原字节及失败全留存；一行编码修正经作者确认，不改变ID或科学条件。最终worker6dc74f36／binding4c49f6e5在OLD完成0.058247秒真实输入绑定，45题／两根90图身份与可逆视图全部通过，0模型／生成／评分／CP读取。没有GPU准入／排队或质量结论，见[全部修正和收据](summary/train_answer_prior_20261025/POSITION_IMPLEMENTATION.md#真实输入绑定的两处模型前修正)。
2026-10-26首个完整10%结果后的同ID金融比较已唯一完成：2233题=1307单选＋546多选＋380数值，S15／S16／S19六套原预测全部绑定两条正式proof；复制13398条已有分数，0模型／生成／训练／新评分／权重读取。金融刚习得F1为16.405→25.414，但单选回答不同值174→9、原满分331→495、原零分966→812，撤回“集中本身就说明习得失败”的推断。10%多选仍286条严格单字母且原F1全零，数值S15→S16为8.640→6.974；本固定配方的题型习得不均匀，不据此识别统一位置偏置或PIF有效。完整子进程1.357466秒，内部1.048091秒重叠不加；源码执行前相对路径围栏修正、正负、来源及独立review均见[完整观测与设计影响](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md)。没有据TEST分析新增训练方案或修改设置，两个完整主表及最终模型留存任务不缩减。

2026-10-26人类进一步明确：设置一旦确定就不再更改；研究固定条件下的方法局限，不为救结果调整条件。已写入根README、AGENTS和[重点原话](summary/acquisition_readout_20261025/v1/HUMAN_RESEARCH_FOCUS.md)。两个正式主实验设置未改。

一次共同TRAIN输入资格核对结束，主／子独立53项状态全同，45接受／原7排除／1隐含指代不确定，固定循环与范围封定，0模型／新评分／训练；[结果与全部边界](summary/train_answer_prior_20261025/OPTION_ELIGIBILITY_FINDINGS.md)。另一次[PIF定向阅读及独立评判](summary/train_answer_prior_20261025/PIF_READING.md)仅保留可借的训练约束，未据通用论文启动训练。两者均不是10%质量、位置根因或新方法成功；仍先解释固定比例正式结果。

10%后续比较已固定为同骨干／同方法的训练前、刚习得与原始错答分层，见[比较口径](summary/train_answer_prior_20261025/ACQUISITION_COMPARISON_PLAN.md)。[功能诊断评判](summary/train_answer_prior_20261025/FUNCTION_DIAGNOSTIC_ASSESSMENT.md)优先等待完整正式证明；没有准入新的1%／10%训练fork，也没有执行位置干预。TRAIN标签偏斜与模型错误之间的因果尚未识别。

FOREVER的独立10%目录已完成[纯元数据构建](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_BUILD_OBSERVED.json)，BUILD_RECEIPT SHA ceaaf231a06be45997a7354af36695f304393d75309c550a29ea2a70a8513b5b：29个Python源中原公共文件逐字节复制，run仅修改条件／比例／主指标元数据，新增core／adapter和构建器；未改1%或正在运行的10%源。初审曾漏读原完整preflight入口，root发现`begin_stage1`后19单题诊断不推进core，已在模型前修正严格丢弃未推进调度器的衔接；另修观察hash对0维Adam step的读取，0模型／评分。独立[集成审阅](summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_INTEGRATION_REVIEW.md)与[唯一GPU检查要求](summary/two_backbone_pct10_20261025/GATE_REQUIREMENTS.md)保留更正和边界。两FOREVER仍待核验，未运输OLD或启动，不将构建／审阅称GPU通过或质量成果。

人类最新明确“以后只要考虑新习得就行，不考虑遗忘了”，已写入根AGENTS.md与[研究重点记录](summary/acquisition_readout_20261025/v1/HUMAN_RESEARCH_FOCUS.md)。后续只检验现有方法为什么没有学好新任务及改善新习得的机制，不继续保持／抗遗忘／历史保护。正式五域最终F1、完整参数及GPU指标仍报告，不把工程实现或少数据计数当原因结论。

新的80条已有快照读出已完整核验：原共同配方与朴素r8的医学刚习得均36.857143、遥感均65，40条逐ID F1全同；共同配方的终态收益在这两个域没有对应新习得提升。两S19全状态重构及20阶段视图、80生成／唯一80评分、0训练／NLL／TEST完整；983.880617模型秒。verifier ROOT入口的真实0评分故障保留，独立单AST修复不动源／READY／回答，原unit仍failed；1542.919735原激活累计秒内完成。全部正负、范围、修复成本与[独立评判见完整结论](summary/acquisition_readout_20261025/v1/FINDINGS.md)。原预测和22新输出已完整转存NEW private，父包不再重复hash；无新的正式TEST结果，8/16及原表不变。

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

2026-10-25原生紧凑Q完整终态：初34.230333→37.991536<43.791987，完整权重−43.214033%，GPU五窗峰却全部更高；关闭固定点。仅配置标准加载对象恢复，保原失败、87更新不重做、原deadline不延；210生成／200唯一评分／0TEST，原起点到完成2176.544260秒含修订等待。完整CP已存NEW private并逐字节绑定，OLD源保留；两研究单位已退出。[全部证据与分项费用](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md)。

2026-10-25预训练空间读出对应pair首次真实fit失败：首组raw／teacher的FP16合同不成立，实际dtype未留存，不臆测具体类型或算子。已实际处理首FIT视觉，0新回答／评分／反向／优化器更新；后五阶段未STARTED，unit failed／原worker已退出。STARTED→EXIT15.663594秒仅重叠墙钟代理；准入4.483917秒、准备／运输／实现／审阅另计。原错误／协议／READY保留，非纯I/O例外，不加cast／改autocast救跑。本次质量缺测、非读出科学阴性；[证据与费用](summary/spatial_readout_transfer_20261025/v1/FAILURE_FINDINGS.md)。

2026-10-25原生响应选择诊断完整exit0：32组仅S1 FIT、32原answer_loss decoder前向及768抽样Down，全24层d非恒定且d*h_diff非零，RMS0.012322–0.090742。31.973244模型秒／准入5.810416秒，0反向／更新／生成／F1／TEST；诊断峰2173.629MiB非受控部署比较。原893675552系数和全部注册buffers不变，旧失败点未恢复。规则033没有要求候选必须小于未适应base；公平fold后的OSFT权重不可用原保存态放大差异。下一问题为同显式表示的动态／静态净作用，其事后首次启动见下段；[证据、全部24层与先例边界](summary/native_response_selection_20261025/v1/DIAGNOSTIC_FINDINGS.md)。

2026-10-25原生响应动态／固定pair首次启动：8源与协议及事前问题账本先发布223c4f89f8ee408f46ba4d95b2a26d3f64677487，PAIR_READY 14c9d38e11608d5582f9fe401606e28e4a6c0480ce33f86c5af26fa14e171444；OLD /tmp/llmcl-response-pair-1084-geic2vef。1792911986.427816快照unit active/running，controller3945071，dynamic3945072/start761038816/UID1084/隔离PGID同PID/GPU0与static3945073/start761038817/GPU1均实际身份匹配；两臂首次STARTED1792911968.094／1792911968.072。每臂3600秒训练含初100盲生成，最终同卡串行各600秒含5资源窗与100盲生成；总8400模型秒。计划2426反向／174更新／410生成／400唯一F1，0TEST；尚未全部发生。首次故障不恢复／延时／搜配置；四phase完整成功退出后才唯一核验质量。证据账本原字节冻结，新增支持／反证与逐域结果另写结果文件，不事后倒推问题或把动态胜静态当已胜普通QV。完整权重与同卡峰待测，未证明强参照优势；[事前账本](summary/native_response_selection_20261025/v1/PROBLEM_METHOD_EVIDENCE.md)、[启动](summary/native_response_selection_20261025/v1/PAIR_ACTIVATION.json)、[时点元数据](summary/native_response_selection_20261025/v1/PAIR_LATEST_METADATA_OBSERVATION.json)。

2026-10-25原生响应pair完整终态：唯一VERIFIED 482d531edc4ab26455986b2f722204b9ac350368f3ef71bdd70da8de85beca76，四phase实际exit0、unit inactive/MainPID0，原四worker身份均无；不再等待或恢复此入口。动态宏44.757602698／固定44.681200950，医学和金融均低旧各域底线，按事前协议关闭。两图base+24β，最终893675576系数、普通buffer242504bytes；实际五窗峰约1857–2166MiB，两图接近但没有测强基线胜利。完整模型秒2507.544632246、controller墙钟1436.842244530重叠不加；唯一核验25.930428430、标量consumer0.625、归档97.583366430秒另计，32QA校准31.973244172秒及历史成本保留。NEW private_paired已核314个新产物字节SHA，包括初末完整图和19阶段adapter；OLD源保留。原事前问题账本不改，新正负／逐题／β轨迹与缺测解释均在新[完整结果](summary/native_response_selection_20261025/v1/PAIRED_FINDINGS.md)。结果评审纠正动态β上界不能称已测βd，文案修正0模型／0评分，不重新消费质量。两活基线元数据见[时点记录](summary/native_response_selection_20261025/v1/BASELINE_METADATA_OBSERVATION_DURING_PAIR.json)，无新增正式proof，8/16及原主表全部保留。

2026-10-25人类明确修订判据：“不用五领域都最强，总的稍微高一点就行。因为我们有个很显著的优势：内存占用小得多”。因此旧逐域hard floor不再适用，按总体宏保留动态候选进入必要比较；旧PAIR_VERIFIED／事前账本／旧关闭判决原字节保留。正式PAS原完整图GPU1五窗已独立exit0，VERIFIED／state与输入绑定通过，73.128928190模型秒、5×32新token、0F1／NLL／训练。动态峰低24.862500–26.798542%，完整系数少13.608393%，原dtype权重字节少23.956668%；保留原混合dtype／训练CPUhistory／非最优部署／五窗非全分布限制。准备时missing-proof路径修复与误hash整source map（含1次旧CP及TEST／预测文件SHA-only、无QA解析／质量读取）12.935504919秒均公开，不抹除或当科研收益。下一项为匹配GPUAdam／GPUfold的QV-only，已写独立部署／适应副本，尚未准入或模型启动；不重开旧worker，不根据TEST挑方案。[新决策](summary/native_response_selection_20261025/v1/CURRENT_DECISION.md)、[PAS全部资源证据](summary/strong_pas_resource_20261025/v1/FINDINGS.md)。

2026-10-25匹配QV-only已实际首次STARTED：7科学源／协议与独立跨审先发布4d19822edef8f819079b97d733516b9df0cbdd9e，READY 5e6f239da2cc7e572704554770ea22dda18aeaab12a69e1f5cecbe40c9e93369；OLD /tmp/llmcl-response-qv-control-1084-nkmx9g5k。1792915769.496908快照unit active/running，controller3960919，train worker3960920/start761408126/UID1084/PGID同PID/GPU0实际身份匹配，已记录4次clip梯度见证，eval未STARTED，无COMPLETE/FAILURE/VERIFIED。这仅是时点观测，不永久live或冒充完整更新。首次train3600／eval600秒固定，原数据序／48AB初始化／GPUAdam／fold绑定原pair，0新初DEV，最终105生成／100唯一F1，全部完成后核验。最初复制脚本常量出现次数断言3误而实际2，0模型、未写common，修正即复制；原verify源码ROOT新实验路径不存在的假设在首次模型前改为绑定的旧冻结RUN，0模型、source正审后冻结。两处是必要实施修正，不当科学收益／CPU模型门。原两基线于1792914964.992831元数据实查仍为同PID8260/start60876与8265/start60882，CPUticks25923339／26014584，索引SHA41b80a...不变，无新增正式proof；8/16不变。

2026-10-25匹配QV-only已结束并唯一核验：VERIFIED 1b0a8f8df3e4f1a4f3338a439398cd9c3197930002302dd8e8a72015fb9fa99c，两phase COMPLETE/EXIT0、ALL_EXIT与原身份gone已确认。探索宏45.085962854，动态／固定44.757602698／44.681200950仅承接旧分数，0重评；按总体标准现选QV，不再以响应调制作收益来源。完整权重893675552／1787351104bytes、buffers9032、CP1787622158bytes；GPU训练4399104坐标、梯度17596416bytes、两Adam35193216bytes另计。与原PAS同物理GPU1／同五窗QV peak低24.843322–26.806462%，原dtype权重bytes少23.956672%；仍是原混合dtype实现，不称同dtype最优／速度／正式F1胜。1213FIT／87更新／105生成／100唯一评分／0TEST；模型953.547774416秒、controller956.121209038秒重叠不加，核验13.770847608秒、归档45.653184157秒与实现／运输／审阅另计。155个新产物已完整归档NEW private_run，SHA核验，旧scratch保留，权重／原始QA不入Git。RESULT_REVIEW披露作者身份与复核范围，不冒称独立重跑。正式表及FOREVER／8组存储值原字节不变。

2026-10-25基线容量调度已执行：旧主CPU2497/start10169与hardware2496/start10168的KillMode=process／无ExecStop、独立GPU8260/start60876与8265/start60882均实查后，只停两个CPU coordinator。新主1221743/start26017836统一接管两lane，原GPU身份与CPUticks前后同一且继续增长；无GPU信号、不重启训练。只将v2策略两NEW host设备分配改reserved[0]／pct01[1,2,3]／full[]，全部25科学py SHA原样；FOREVER冻结frame及队列不动。旧hardware已disable避免回到全Qwen屏障及自动full尾段；新主Restart=no、原KillMode=process。发布838da45aa274d5958c752f4f5c636ec752d9aaf5。1792917321.335583新机快照：GPU0为UID1020占用，GPU1三个UID1024，GPU2/3两原基线及UID1024共享；wrapper真实active/running、pending五项，不能谎称全基线已同时满载。1792917330.575376旧机四卡ownerless；七OneVision原结果已sealed仅缺必须原生BF16的新机SMo，不能在2080Ti强改FP16救跑。新机CL实际completed16/active17，SMo初态评价仍前进；保原supervisor路径及ready_reload原门，未重复preflight。新sealed proof后再提取新增组合完整权重/必需状态/原dtype字节/容器文件及训练坐标，保留原8组不重复大hash；此为必要调度实施，不当科研方法收益。

2026-10-25人类新增明确“不能简单扩容”、同时鼓励已有多领域CL机制的具体场景适配。宽QV formal源/协议及静态ADMISSION_REVIEW在新指令前刚备妥，但无READY、未调用metadata准入/激活，0formal训练/TEST生成/评分；独立DIRECTION_CLOSED覆盖其计划，不改原源/草案/审阅或已完成QV结果。仅只读资产检查（原数据完整、旧view缺TESTmeta）及本地8锚标量汇总；Qwen宏由原五域均值重聚合48.972351176657504，与原proof宏差约1ULP，不是新QA评分。wide新增4399104、比朴素540672多3858432，fold后均0新增，不能包装方法创新。GPU模型未启动、未终止任何训练；基线统一调度继续。正式9/19阶段仅一更新，FIT10/19，不能混为正式多数；特定冷专家问题不推广为所有共享基线。新研究转向已有CL功能承接的具体适配，候选不自动准入，完整目标仍未实现。

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


## 探索过程与证据索引

2026-10-25按人类要求重构：最外README只保留主表、命题与结论；详细过程移至此处。探索按“原始观测→事前命题与必要对照→实际执行→正负结果→结论边界／后续决定”保存。原事前记录不因结果回写；结果不符合命题也保留；模型未到质量评价的故障标记为缺测，不能记为科学证伪。

| 探索链 | 事前依据／过程 | 实际结论／终态 |
| --- | --- | --- |
| 原生FFN响应调制 | [原始问题账本](summary/native_response_selection_20261025/v1/PROBLEM_METHOD_EVIDENCE.md)、[诊断](summary/native_response_selection_20261025/v1/DIAGNOSTIC_FINDINGS.md) | [动态／固定全结果](summary/native_response_selection_20261025/v1/PAIRED_FINDINGS.md)、[匹配QV反证](summary/native_response_selection_20261025/qv_control/FINDINGS.md)、[不把扩容当创新的关闭决定](summary/native_response_selection_20261025/formal_confirmation/DIRECTION_CLOSED.json) |
| 原完整强参照的端侧成本 | [PAS五窗实测与原dtype限制](summary/strong_pas_resource_20261025/v1/FINDINGS.md) | [同卡QV／PAS逐窗比较](summary/native_response_selection_20261025/qv_control/RESOURCE_COMPARISON.json)，仅证实该测量范围的资源优势，无正式F1胜利 |
| 小载体／真实删除与功能承接 | [FFN联合删除](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md)、[小视觉接口](summary/vision_interface_transfer_20261025/v1/FINDINGS.md) | [紧凑表示先学习](summary/compact_first_learning_20261025/v1/INTERPRETATION.md)、[更新分配](summary/compact_allocation_20261025/v1/INTERPRETATION.md)、[原生SmolVLM](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md)：对应固定点均未达保质目标 |
| 借鉴CL并选择一致性目标 | [旧slow-NLL草案](summary/answer_reliability_consolidation_20261025/v1/IDEA_REVIEW_DRAFT.md)、[子agent代数反例与修正](summary/answer_reliability_consolidation_20261025/v1/CURRENT_IDEA.md) | [实际target读出门方案](summary/answer_reliability_consolidation_20261025/v1/METHOD_PROPOSAL.md)，尚无新质量结果；新模型未激活 |
| 匹配朴素r8 GPU控制 | [固定协议](summary/plain_r8_gpu_control_20261025/v1/PROTOCOL.md)、[源码审阅](summary/plain_r8_gpu_control_20261025/v1/ROOT_SOURCE_REVIEW.json) | [最近元数据观察](summary/plain_r8_gpu_control_20261025/v1/LATEST_METADATA_OBSERVATION.json)，训练完成、评价中；完整退出后才唯一核验，未消费partial质量 |

### 朴素r8控制的实际启动与实施成本

READY 74815accb4dea83cedadef2a994f8162ccfb1502c84589b36b657b55ac8cfd7f；预注册发布7fbf015a263fe5458ecf92a40eb43e27e9b91f2a。OLD `/tmp/llmcl-plain-r8-gpu-1084-mlzh5u5e`，unit `llmcl-plain-r8-gpu-control-20261025.service`。首次训练STARTED unix1792922673.0999377，worker3984538/start762109322，GPU0；固定train3600/eval600、总4200模型秒，外层4300，不重置。训练1213FIT／19阶段／87正LR更新，评价计划105生成／100唯一F1，0初态DEV／TEST。

unix1792923545.81275的只读快照：训练COMPLETE/EXIT0、87grad见证／19阶段完成且原身份已无；评价worker3988735/start762193843在GPU1身份匹配，未COMPLETE，unit active/running。不是完整质量结果或永久live声明。匹配r8训练540672坐标、梯度2162688bytes、GPUAdam4325760bytes；最终图设计893675552系数／1787351104权重bytes，实际完整评价证明待完成。

准入元数据脚本曾因SHA助手收到str路径并调用read_bytes失败，发生于临时目录创建与模型之前；修为Path后首次成功准入0.241954804秒。0模型失败、0QA、科学源未变；该实施修正不计科研收益，不抹去。源/冻结和模型数学均在GPU，未做CPU神经模型门。

### 本次README重构前的完整研究叙述（历史快照）

以下保留重构前README中主表以外的全部叙述，避免删除探索过程；其中“当前／下一步／正在运行”只代表该文当时的时点，以本节新索引及最新完整proof为准。主表保持原字节留在最外README，不在此重复。旧叙述中的尚未执行比较可能已在后文完成，不能据此再启动一次。

<details>
<summary>展开：2026-10-25重构前研究叙述与证据链接</summary>

# LLMCL：少数据多模态持续学习与紧凑模型研究

> 更新：2026-10-25。本文集中记录已有成果、负结果、尚未解释的问题和新会话接续入口。细节见[补充说明](SUPPLEMENTARY_NOTES.md)；原README时间线保存在[历史归档](README_HISTORY_20261024.md)（数据样例省略，原件私有保留），其中“当前／下一步／正在运行”均为历史记录。

## 先看结论与当前状态

- **1%基线调度已纠偏：** 新机只保留GPU0不由我方分配，GPU1/2/3统一接续；已移除OneVision SMoLoRA必须等全部Qwen结束的屏障及旧100%自动开放尾段。两原GPU进程身份完整保留并继续推进：CL-MoE已完16／正在17阶段，SMoLoRA初态评价；PAS／PaRSP待原CP重载，OneVision SMoLoRA／OSFT／SpaRTA待空卡。新机三可用槽当时均有进程，GPU1是他人占用，不能称所有方法已同时开跑；空闲后按原guard自动接续。FOREVER原队列保留、已核验8组不重复。完整参数／原dtype字节／文件大小分列，新增正式结果核验后增补。[执行快照与参数记录规则](summary/baseline_capacity_20261025/v1/FINDINGS.md)。
- **已经取得：** 双骨干正式对照中的8组完整核验结果、对应最终checkpoint的完整权重统计、多项习得／冷启动／回答行为诊断，以及一系列完整减重但质量失败的对照。失败结果和费用均保留。
- **仍未做到：** 尚无最终完整权重更小、又保留强参照回答质量的自研框架。也尚未因果解释OSFT习得优势及ShortFixed点估计表现的原因，不能把性能表或局部误差直接当机制。
- **强基线GPU实测已补：** 原完整正式PAS直接重载后，与两候选使用同一物理GPU1／相同五题输入／固定32新token窗口；动态峰1857–2166MiB，PAS2490–2959MiB，候选低24.86–26.80%。完整系数较PAS少13.608%，原dtype权重bytes少23.957%。这是原FP32专家／FP16基座实现的实际比较，含dtype差异，非同dtype最优部署或全分布峰；不替代尚未证明的总体强基线F1优势。5生成／0评分／73.128928模型秒，原状态不变。[全部窗口、来源与实施成本](summary/strong_pas_resource_20261025/v1/FINDINGS.md)。
- **当前方向已纠偏：** 宽QV探索宏45.085963是容量对照，不作为Ours：其4399104训练参数比朴素r8多3858432，约8.136倍容量，合并后两者都回原基座。用户明确方法需简单且有具体场景创新，允许借鉴已有多领域CL再适配；宽QV正式确认未准入／未启动，已停止推进。当前评审I-LoRA功能可靠性门等不靠加宽的具体适配，尚无新效果。[关闭记录](summary/native_response_selection_20261025/formal_confirmation/DIRECTION_CLOSED.json) · [问题与候选改动](summary/answer_reliability_consolidation_20261025/v1/CURRENT_IDEA.md)。
- **骨干命名与比较范围：** 统一写LLaVA-OneVision（Qwen2-0.5B-Instruct语言骨干＋SigLIP视觉编码器），完整893675552系数；OneVision此前是该LLaVA子系列简称。OneVision内最强已sealed五域宏PAS46.619976，全体已sealed最强Qwen顺序LoRA48.972351；两范围同时报告，不能无条件把PAS称全体最强。现有显存实测锚只有原PAS，实现dtype差异保留。
- **匹配QV对照已完整核验：** 原48个A/B初始化／1213FIT输入与两调制臂逐项相同，87个正LR GPU更新、19阶段；105新生成／100唯一评分，两phase成功退出后才核验。完整图893675552系数／1787351104权重bytes，另9032buffers bytes，完整CP1787622158bytes；已私有归档155个新产物且逐SHA核对。0TEST，无rank／LR／尺度救跑。[参数、五域、逐题与费用](summary/native_response_selection_20261025/qv_control/FINDINGS.md)。
- **从问题到反证：** 强方法的专家成本与原生FFN响应非退化曾促成调制假说，但匹配QV胜动态0.328360pp／胜固定0.404762pp；此前高于旧wide不能归因新增调制。保留事前账本，不把小β或局部信号改写为有效机制，也不隐藏阴性对照。[原始问题账本](summary/native_response_selection_20261025/v1/PROBLEM_METHOD_EVIDENCE.md)。
- **GPU功能诊断的边界：** 原生FFN响应选择的32组S1 FIT／768次抽样Down为非退化，但随后的匹配学习未支持两个调制包的净质量收益；初态信号存在不等于回答价值或专家可删。[原诊断](summary/native_response_selection_20261025/v1/DIAGNOSTIC_FINDINGS.md)。
- **最新尝试（质量缺测，已退出）：** 预训练空间读出对应与打乱控制在首个真实FIT视觉特征的dtype合同检查失败，仅fit启动；0训练更新／新回答／评分，两臂及父资源均未启动。原错误和费用保留，本次实施关闭，不改精度或静默重启；不据此否定读出假说，也不算科研性能进展。[故障、费用与两agent评判](summary/spatial_readout_transfer_20261025/v1/FAILURE_FINDINGS.md)。
- **最新完整功能对照：** 原生SmolVLM完整图普通q/v学习，宏34.230333→37.991536，低于历史完整父43.791987；自然95，但医学21.885965／金融10／交通18.071713仍低父。完整权重少43.214%，实际包少43.212%，五窗同卡GPU峰却高14.09–57.43%。质量与GPU联合标准均失败，关闭固定native-FP16-Q；不是新框架成功。[全部域、资源、修订与费用](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md)。
- **先前尝试（质量缺测）：** GPU删图后CE／蒸馏对照仅原CE完成320次反向／20更新；KD构图故障，一次盲态加载修订随后在全部buffers一致性断言停止。0新DEV回答／评分，不能比较CE与KD；旧CE不评价、不续CP，费用与失败全保留。[记录与费用](summary/poststream_response_transfer_20261025/compatibility/FAILURE_FINDINGS.md)。

公平比较要求最终真少权重或结构，**没有要求每个候选必须删原基座W0**；保基座而避免永久新增专家也可研究。OSFT允许的常规fold必须同等实施，不能拿其较大的原USV保存态作独占优势。相对动态专家的减量也须由实际最终图与同卡峰证明。

最新人类评价目标为**总体原Answer F1略高于最强基线，同时完整模型／GPU内存显著更小**；允许域间取舍，五域结果都报告，不再要求逐域分别领先。沿用既有五域宏口径，AP19另列；不根据结果挑平均方式。[最新规则](rules/035_overall_f1_memory_tradeoff_20261025.md)。

研究目标是**更少的最终完整模型权重，同时尽量保留或提高原Answer F1**。可训练参数量和训练状态作为辅助指标；端侧是科研动机，不做设备迁移、部署交付或精度卡常。已有技术可以借用，贡献由具体功能问题、必要对照和证据决定；不能因有先例就否决，也不能因换背景就宣称原创。

## 实验口径：三类结果不能混排

| 类别 | 数据与评价 | 如何使用 |
| --- | --- | --- |
| 正式持续学习 | 原1313 TRAIN／19阶段／完整21454 TEST，seed42，原Answer F1 | 仅完整核验后填正式AP／FWT／BWT；TEST不能选方案 |
| FIT→DEV先导 | 原训练池分出的1213 FIT／100 DEV，19阶段87次更新 | DEV已反复探索，单seed、每域20题、89个已知组；只筛选，不当新留出或稳定收益 |
| 历史诊断 | 11／155／228／507题或S1合法FIT子集等各自固定范围 | 仅支持本诊断的局部结论，不与前两类直接排名 |

唯一指标为原始token Answer F1（0–100）。**AP为19任务等权，五域宏是五个域均值再等权；两者不同。** T7／8／9／12没有探索DEV。用户尚未给定可容许掉分范围，因此没有“非劣／等效”结论。正式1313 teacher见过探索DEV，即使只在FIT蒸馏，也不能拿来筛学生。

正式完成数最后一次证明检查仍为**8／16**：OneVision七组、Qwen顺序LoRA一组；这是已核验数量，不是其它作业当前存活状态。Ours没有正式完整结果。



## 为什么强参照学得好：已有证据与未解问题

| 问题 | 已有数据／干预 | 现在可以说什么 |
| --- | --- | --- |
| OSFT优势主要在习得还是保持？ | 正式19任务刚习得均分OSFT **47.887821**、Seq **38.604705**；S0为15.050963／15.098792，BWT为−1.616191／+1.774861。[证据](summary/final_weight_reduction_20261024/osft_initial_gap_check_v1/FINDINGS.md) | OSFT在这条轨迹上刚习得更高，但保持指标更差；简单“起点分数高了同样几分”不足以解释。位置、坐标量、LR、分解和保护混杂，尚未识别原因或具体知识 |
| 所有方法都每域从零、几乎没更新吗？ | 11条医学QA、一次更新的局部新LoRA：从头19.88，完整继承29.26；冷组48层A首步不更新，继承后均更新。[局部证据](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/RESEARCH_FINDINGS.md) | 局部新专家存在弱习得，继承有帮助；**Seq和OSFT跨阶段连续，不是每域清零**。不能推广为所有基线共有故障 |
| 只是首步A没有梯度吗？ | 独立507题控制：冷14.681，同域／跨域完整继承18.113／18.468；A-only为14.340／14.464；函数中性补偿使首步梯度开放后15.428／15.533。[完整控制](summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md) | 开放首步梯度没有复制完整继承收益；功能继承与参数化作用不能只用单一首步故事解释 |
| OSFT必须训练两侧因子吗？ | ShortFixed完成同1213 FIT／87更新，DEV宏42.301329，对原OSFT40.783801；训练梯度和两Adam状态少21.34%。[证据](summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2/FINDINGS.md) | 本配方中冻结短侧仍有竞争力；**前向权重未删、长侧仍学方向**，不证明短侧普遍无用或新轻量方法成立 |
| 多位置、大坐标量是必要原因吗？ | 同4399104坐标宽q/v **43.791987**，七位置all-r8 **41.952549**，差1.839438pp、探索区间跨0。[证据](summary/edge_deployment_20261019/position_coordinate_control_20261022/v1/FINDINGS.md) | 原位置扩展收益混有容量影响；少数金融题贡献集中，尚未证明额外位置必要或无用 |
| 逐步保护包一定有净益吗？ | 仅关闭原OSFT梯度／参数两投影，DEV **42.333991**对40.783801；金融25→15，区间跨0。[证据](summary/edge_deployment_20261019/osft_protection_control_20261022/v1/FINDINGS.md) | 净作用未解，不能按平均分宣布保护无用；两组都保留阶段重分解，尚未检验刷新功能 |
| OSFT只是更会利用图像吗？ | 同环境61题、四状态488次原图／错图NLL，配对支持改善q/v .455118248、OSFT .441384426，相对差 **−.013733822**。[证据](summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/RESULT_FINDINGS.md) | 未支持“OSFT整体图文配对支持增加更多”的预定解释；不能反推不需要视觉、识别能力无用或多学了某种知识 |

OSFT医学域刚习得相对Seq高15.201186pp，最终仅高3.136410pp；自然域最终反而低1.725122pp。不能把任务平均优势说成五域都由同一机制稳定受益。PASs与OSFT正式AP仅差0.049476pp，也没有分离路由、专家、保护与数值例外的各自因果。

### 非自然领域回答差：定位到了什么

[完整回答行为分析](summary/final_weight_reduction_20261024/response_failure_profile_v1/FINDINGS.md)消费三条正式运行的初态、刚习得与末态记录，没有新生成或重评分：

- **金融：** PASs最终2233题有1669题输出同一个单字母选项，其中1185题为零分；OSFT众数1079题，其中819题零分。大量零分在刚学完时已经存在，不能全部归因遗忘，也不能仅用“没按选项格式回答”解释。
- **遥感：** PASs计数任务2947题中1298题集中为同一数字，其中1213题零分。Seq比较任务4002题中3999题回答相同，任务F1仍约66.72；平均分会掩盖固定回答。
- **医学：** 最终2094题，Seq／OSFT／PASs分别1022／986／1001题零分，预定字面异常仅覆盖91／58／41题，主要失分仍未解释。
- **交通：** 2123题全部部分分，是词面重合分布，不能写成全部正确；原记录不足以由触及长度上限断言被截断。

已查金融89 TRAIN＋2233 TEST的所查字段、选项与当前原始投影一致，未发现本地复制错位或检索丢选项；4条非空选项不能严格解析，仍未知，且不认证全部历史逐token输入或作者金标语义。[金融输入合同](summary/final_weight_reduction_20261024/finance_input_contract_v1/FINDINGS.md)

**没有找到统一的致命输出bug。** 输入利用、答案先验、知识调用、有限步优化和参数表示的作用仍未隔离；这些行为现象不是参数可删的证据。

## FIT→DEV探索结果

以下均为同一100已探索DEV的五域宏，不能与正式21454分数混排。各实验完整域分、逐题正负和费用在链接中。

| 条件 | DEV五域宏 | 实际意义与终态 |
| --- | ---: | --- |
| q/v LoRA-r8 | 39.870439 | 原先导参照；固定条件完成 |
| 原OSFT | 40.783801 | 原FIT-only参照；机制尚未解释 |
| LoRA_all-r8 | 41.952549 | 小测曾入选；完整确认后来失败，见下 |
| LoRA_extra | 40.365016 | 四组筛选完成，未入选 |
| 同坐标宽q/v | 43.791987 | 同4.399M坐标对all高1.839438pp，区间跨0；固定对照关闭 |
| OSFT去逐步保护 | 42.333991 | 平均点高、金融下降，净作用未解；固定对照关闭 |
| ShortFixed | 42.301329 | 对OSFT高1.517529pp，9高7低84平，区间[−.126262,3.962820]；主要由自然和医学各1题贡献；仅必要性消融 |
| NormOnly | 23.242734 | 权重确更新但低于两锚；无同DEV零样本base，不能证明已有参数普遍不足；关闭 |
| fresh PASs FIT参照 | **缺测** | v1重启中断；v2冻结4小时内仅77／87更新，无DEV，不是质量阴性；禁止续跑 |

四组原筛选见[结果](summary/edge_deployment_20261019/osft_increment_screen/v1/RESULTS_20261022.md)。ShortFixed对q/v高2.430890pp，但医学低5.5pp，探索区间也跨0；它没有通过fresh PASs完整比较，所以未放行完整确认。不能把已有最强正式PASs当合法FIT-only teacher替代。

**LoRA_all完整确认：** 原1313训练／19阶段／93次更新→21454题，五域宏**44.894423**、AP19 **44.794146**，低于正式OSFT和PASs，五域均低OSFT。已关闭这套配方，不据TEST救rank／LR／层／路由。[完整负结果](summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1/FINDINGS.md)

## 真正减少完整权重的探索：哪些失败了

这些是实际完成的负结果，不能因为权重变小或局部误差下降就写成轻量高性能方法。表中0分是实际测得，缺测另标；F1均为对应100已探索DEV宏。

| 固定探索 | 完整最终权重系数 | 起点／末态F1 | 结论与证据 |
| --- | ---: | --- | --- |
| 小视觉先验＋S1仿射空间接口 | 581,213,056（−34.9638%） | 已学父43.791987 → 候选38.966256 | 293组拟合／32组观察，语言冻结；五GPU峰少261–511MiB，自然／医学／交通下降，整体保质失败。[证据](summary/vision_interface_transfer_20261025/v1/FINDINGS.md) |
| 强wide-q/v父，37FIT真实单删定位后联合删8个FFN | 789,080,096（−11.704%） | 已学父43.791987 → 删图19.620168 | 0训练／反向，真实保存重载；五域F1均降，GPU峰少193.0–200.2MiB，保质失败。[证据](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md) |
| 语言三矩阵r200，Spectral／Answer两完整流 | 662,833,184（−25.8307%） | 末态 .910394／.880147 | 两流各1213 FIT／87更新；19次回答候选均未采用。真实减重但质量崩坏，关闭。[证据](summary/final_weight_reduction_20261023/answer_guided_pilot_v1/FINDINGS.md) |
| KeepGate：保gate/down、只压up | 816,728,096（−8.6102%） | 初 .539926 → 末5.196071 | 同q/v连续适应未恢复；损伤在更新前已出现。[证据](summary/final_weight_reduction_20261024/ffn_role_retention_v1/README.md) |
| KeepUp：保up/down、只压gate | 816,728,096 | 初0 → 末8.905754 | 同数据／87更新，仍不保质；不能在低质量地板挑通用赢家。[同组证据](summary/final_weight_reduction_20261024/ffn_role_retention_v1/OUTCOME.json) |
| up局部功能拟合：Static／gate-response Reuse | 816,728,096／816,706,592 | .255299／.071795 | 各1536个block更新，held局部误差均约降至.25，却没保住完整回答。[证据](summary/final_weight_reduction_20261024/gate_response_reuse_v1/OUTCOME.json) |
| 保gate/up、只压down：WeightD／ResponseD | 两者816,728,096 | .355919／0 | 48个合法S1教师请求、0适应更新，完整保存重载后各100题；两者均失败。[证据](summary/final_weight_reduction_20261024/functional_readout_carrier_v1/README.md) |

最后三组使用的同管线**未适应父模型**宏为**16.116268**，不是强训练基线。父分只按完整digest和逐ID输入绑定的既有证明复制。真实弱父仍明显高于这些替代图；不能把它们与原正式基线的差值当纯压缩因果。

较早半宽方案在5道teacher已见TRAIN探针上，完整／视觉单支／语言单支／联合均分47.089／33.333／22.051／21.754，全部未过原逐题门，**不是独立泛化测试**。[证据](summary/edge_deployment_20261019/width_component_split_20261021/v1/FINDINGS.md)

共享稀疏载体因捕获超过8GiB预算而失败，固定稀疏连接初始化出现非有限值，Smol原生起点在首题processor接口失败；这些都没有可用质量结论。故障与未执行提议详见补充说明。**目前没有一组基座前向权重已被证明可物理删除、同时保留目标质量。** 这些结果只关闭对应固定点，不否定所有低秩、共享或压缩。

## 参数量、最终权重与训练开销

### 最终checkpoint的原保存态

下面来自已保存完整checkpoint的元数据提取，包含视觉、语言、连接器、专家、路由、冻结基和必需持久状态，按底层共享storage去重。不是只算adapter，也不是文件容器总大小。

| 骨干／方法 | 完整保存系数数量 | 最终权重占用 MiB | 相对原基座增加 MiB |
| --- | ---: | ---: | ---: |
| OneVision／SeqFT-LoRA | 894,216,224 | 1706.613 | 2.062 |
| OneVision／FOREVER | 894,216,224 | 1706.613 | 2.062 |
| OneVision／CL-MoE | 930,877,024 | 1846.463 | 141.912 |
| OneVision／PASs-MoE | 1,034,447,216 | 2241.552 | 537.001 |
| OneVision／PaRSP | 893,675,552 | 2273.345 | 568.794 |
| OneVision／OSFT | 990,913,568 | 2757.984 | 1053.434 |
| OneVision／SpaRTA | 904,077,344 | 1744.231 | 39.680 |
| Qwen／SeqFT-LoRA | 2,210,075,136 | 4217.462 | 4.156 |

原OneVision基座**893,675,552**系数：语言494,090,112、视觉397,747,744、连接器1,836,800、image_newline896；原Qwen基座2,208,985,600。保存dtype和表示不同，因此**字节增加不等于新增参数**，例如PaRSP系数不变但部分FP32保存。普通LoRA／OSFT／ShortFixed公平常规fold后仍是原稠密形状893,675,552，消除表示冗余不算结构压缩；这里没有声称已运行数值折叠。[全部dtype、排除项和来源](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)

### 训练参数是辅助项

| 表示 | 未折叠完整权重系数 | 可训练坐标 |
| --- | ---: | ---: |
| 原基座 | 893,675,552 | 本项目起点冻结，不作为训练方法 |
| q/v LoRA-r8 | 894,216,224 | 540,672 |
| 七位置LoRA_all-r8 | 898,074,656 | 4,399,104 |
| 原OSFT | 990,913,568 | 113,766,144 |
| ShortFixed | 990,913,568 | 89,485,056 |

ShortFixed实测梯度357,940,224字节、两Adam moment715,880,448字节；较原OSFT源码计数均少21.342982%，但全部USV仍在，不能称最终模型轻量。其19阶段末释放训练局部引用后allocated约2.958–2.972GB，首末增加14.61MB；RSS后期仍上升。这些不是训练／推理峰或增长因果证明。[已有一次内存增长分析](summary/edge_deployment_20261019/shortfixed_memory_growth_20261022/v1/FINDINGS.md)

阶段图未来需分别画完整权重P_total(t)和可训练参数P_train(t)。固定结构画水平线，不编阶段增长曲线；模型权重存储、训练状态、运行峰值内存分别报告，没测的仍不填。

## 其它已完成的局部发现

这些结果保留科研价值及适用边界，不转写成已完成主方法。历史诊断使用不同题集和输入协议；其中已见TEST的分析仅作解释，不用来筛新配置。

| 已有发现 | 数据与限制 | 证据 |
| --- | --- | --- |
| 适应与遗忘不能只看末态 | 同医学任务学完20.36，后续同域34.17、跨域18.63；增加更新医学29.26→51.96、遥感76→44，不能统一说“多训就好” | [研究报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md) |
| 完整初始化继承在连续诊断有局部收益 | 温启动19阶段／155题AP31.025→35.025、医学15.841→25.339；另外四域未变，仍低于同协议Seq43.363 | [结果](summary/onevision_diagnostics_20261016/warm_start_20261018/first19_outcome/RESEARCH_FINDINGS.md) |
| 输入相似不保证适配器答得好 | 同155题固定相似度22.39、学习相似度31.03、TRAIN回答质量选择36.70，均低同协议Seq43.36 | [路由诊断](summary/onevision_diagnostics_20261016/functional_router/PILOT_FINDINGS_20261018.md) |
| 缓存调用点收益未稳定确证 | 同155题顺序43.363、缓存几何45.700、缓存质量42.860，域宏差区间含0；部分题NLL下降而F1下降 | [四格结果](summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/first_four_view_outcome/RESEARCH_FINDINGS.md) |
| 新TRAIN质量反馈没取得收益 | 新匹配／可学习距离均42.860；20锚只覆盖11／19任务，逐题最优相对最好固定状态空间仅.144pp | [五格结果](summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1/all_five_outcome/RESEARCH_FINDINGS.md) |
| 相同前向功能不保证相同更新几何 | 保BA不变、A缩放4倍，三任务真实梯度下AdamW权重更新约4倍；作者RITE更新相同，没有对应最终F1提升结论 | [几何干预](summary/onevision_diagnostics_20261016/optimizer_geometry_20261019/v1/RESEARCH_FINDINGS.md) |
| 知识记住不等于答题会用 | 16条专业事实可记住，但相关医学问答没有稳定改善 | [原报告](summary/onevision_diagnostics_20261016/RESEARCH_REPORT.md) |
| 少量更新时部分保护项可能来不及起效 | PASs代码诊断中9个单步阶段一保护系数一直0；以当前参数为中心的EWC首步保护梯度0。未证明最终分数因果 | [保护预算诊断](summary/onevision_diagnostics_20261016/regularizer_budget_20261018/RESEARCH_FINDINGS.md) |
| PAS保护机会不均 | 正式93步条件重建40步结构零、53步仅有正系数资格上界；不是实际有效保护见证，旧上游源码SHA有缺口，不是成绩因果 | [覆盖证据](summary/final_weight_reduction_20261023/pas_rs_coverage_v1/FINDINGS.md) |
| 选择变化不等于选择器学了 | SMoLoRA代码检查两选择器无梯度、专家正常更新；固定选择器仍能随输入切换，没有质量因果结论 | [梯度检查](summary/onevision_diagnostics_20261016/smolora_gate_20261018/RESEARCH_FINDINGS.md) |
| 名义更新范围不等于真实权重变化 | PaRSP部分o/down在名义mask并集外变化，占所测ΔW能量2.9344%；SpaRTA93步未到200步恢复周期 | [原证据](summary/edge_deployment_20261019/sealed_update_constraints_20261021/FINDINGS.md) |
| 自然重排Kron不值得当前点继续投入 | 首层三个FFN矩阵、近同系数预算，K69自然布局权重残差0／3胜普通r200；仅离线优先级阴性，非F1结论 | [原证据](summary/final_weight_reduction_20261023/kron_layout_priority_v1/FINDINGS.md) |

文献／方法分类整理了29篇2024–2026正式会议工作，以及五数据集、19任务、来源与复用依据。[分类、数据表和历史基座核查](SUPPLEMENTARY_NOTES.md#文献分类与数据组成)保留完整索引。LoRA-FA、SVFT／SVFit、RandLoRA、HOFT、共享／低秩／动态压缩等已有技术可作工具；它们本身不是我们的实测贡献，也不因此禁止有意义探索。

## 最新完整对照与缺测终态

真实单删定位借鉴功能消融，只在37条合法FIT上计算925次NLL，再固定8层并联合物理删除；最终父／删图各100DEV。父宏与已有wide-q/v点一致；各域NLL均升高，五域F1如下，不把减重本身当成功。

| 领域 | 完整父F1 | 删除F1 | 删除−父pp |
| --- | ---: | ---: | ---: |
| 自然 | 61.500 | 27.500 | −34.000 |
| 医学 | 44.500 | 25.000 | −19.500 |
| 遥感 | 45.000 | 25.000 | −20.000 |
| 金融 | 30.000 | 0.000 | −30.000 |
| 交通 | 37.960 | 20.601 | −17.359 |

完整权重893675552→789080096系数；五个固定输入、各恰32token的GPU peak allocated下降193.007–200.162MiB。显存与F1必须联合判断，本构造未获得保质证据。单层相对排名不是绝对可删性；单删损伤已经非零，不能把全部损失归于层间交互。所有原始产物和完整包已转存NEW private；细节及一次过程隔离修复见[完整结果](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md)。

**同输入机制补证：** 固定原37FIT与8层，新增37次联合NLL。五域宏实际联合损伤1.949195，原有符号单删累计0.306139（原截零选择cost为0.686164），余项1.643055；34/37题余项为正，五域联合损伤均高于原截零累计。由此否决用独立层累计认证本集合保质，不能单独归因输入漂移，也不据此重选层或恢复训练。[完整结果](summary/ffn_joint_effect_20261025/v1/FINDINGS.md)

**小视觉接口完整结果：** 冻结现有8643万参数学生视觉及已学wide-q/v语言，仅用S1的293组拟合768→896仿射桥（另32组观察）。完整权重893675552→581213056，DEV宏父43.791987→候选38.966256；出口`close_fixed_frozen_student_affine_S1_mapping`。实际五窗口GPU峰、逐域正负、加载/拟合与核验成本见[完整结果](summary/vision_interface_transfer_20261025/v1/FINDINGS.md)。这不是已完成CL框架。

**词面证据功能诊断：** 原20道自然DEV，父/小视觉图均固定，分别给完整同图OCR与等长错源OCR；原无OCR只复用已核验锚。

| 图 | 无OCR F1 | 同图OCR F1 | 错源OCR F1 |
| --- | ---: | ---: | ---: |
| 完整父 | 61.500 | 80.119 | 56.333 |
| 小视觉图 | 29.167 | 42.803 | 24.167 |

小图确获部分词面支持，但同条件父子差距反而扩大4.983pp，按预定门关闭本固定通路；不下载OCR/改prompt或桥救分，不称OCR无效或语义损伤已识别。父F1升而NLL反升，全部方向保留。0训练，80新增生成/评分；预计算OCR无真实部署成本证明。[完整结果与盲态路径补正](summary/lexical_evidence_probe_20261025/v1/FINDINGS.md)。

**紧凑表示先行学习对照：** 原始语言base先装同一固定小视觉／S1桥，再执行原1213 FIT／19阶段／87更新；不是继续旧S19或加步恢复。

| 域 | 紧凑初态 | 事后替换 | 紧凑先行末态 | 完整父 |
| --- | ---: | ---: | ---: | ---: |
| 自然 | 12.429 | 29.167 | 37.857 | 61.500 |
| 医学 | 6.286 | 38.500 | 32.833 | 44.500 |
| 遥感 | 18.000 | 60.000 | 60.000 | 45.000 |
| 金融 | 0.000 | 30.000 | 20.000 | 30.000 |
| 交通 | 8.751 | 37.165 | 36.646 | 37.960 |

从初态宏9.09学到37.47，自然较事后替换有所恢复，但其余域没有一致收益，四域仍低父；仅提前学习不足以保质。运行图893675552→581213056系数（−34.96%），实际完整CP 1787598858→1434984515bytes（−19.73%），五个匹配GPU allocated峰低12.01–27.41%；CPU进程RSS高水上升，不称CPU内存改善。固定点关闭，0新增DEV NLL／TEST；200新F1唯一评分，两位子agent复核。最近NEW基线CL-MoE阶段12、SMoLoRA仍推进，正式仍8/16。[完整证据、文件别名限制及费用](summary/compact_first_learning_20261025/v1/INTERPRETATION.md)。

**紧凑载体的同预算更新分配：** 在同OLD紧凑图上，原七位置r8与wide-q/v均4399104坐标、1213FIT／87更新；新臂训练0DEV，末态严格重载及四域末独立fold后唯一评分。

| 域 | 完整wide | 完整all折叠 | 紧凑wide | 紧凑all |
| --- | ---: | ---: | ---: | ---: |
| 自然 | 61.500000 | 64.000000 | 37.857143 | 37.500000 |
| 医学 | 44.500000 | 42.690476 | 32.833333 | 30.000000 |
| 遥感 | 45.000000 | 50.000000 | 60.000000 | 65.000000 |
| 金融 | 30.000000 | 15.000000 | 20.000000 | 26.000000 |
| 交通 | 37.959937 | 38.478358 | 36.645506 | 37.810209 |

未通过父各域质量底线／资源联合标准，关闭固定all-r8配方。完整图仍893675552→581213056系数（−34.96%）；实际文件与五个匹配GPU窗口见完整报告。四域末不合成新checkpoint宏分；完整视觉历史NEW3090、紧凑臂OLD2080Ti，部署统一不消除学习设备混杂，不能将差分之差归为纯表示作用。0TEST／新NLL，不按中间分数选择后续phase。[全部结果、逐题标量与费用](summary/compact_allocation_20261025/v1/INTERPRETATION.md)。

**固定域末功能读数：** S19零生成重建与封存最终图精确一致后，固定S1自然／S11医学／S15遥感／S16金融各20题，80新生成／评分、0更新。

| 域／固定阶段 | 域末F1 | 原末态F1 | 父末态目标 | 末态−域末pp | 逐题高／低／平 |
| --- | ---: | ---: | ---: | ---: | --- |
| 自然／S1 | 29.166667 | 37.857143 | 61.500000 | +8.690476 | 3／0／17 |
| 医学／S11 | 33.500000 | 32.833333 | 44.500000 | -0.666667 | 2／2／16 |
| 遥感／S15 | 40.000000 | 60.000000 | 45.000000 | +20.000000 | 4／0／16 |
| 金融／S16 | 25.000000 | 20.000000 | 30.000000 | -5.000000 | 1／2／17 |

四域域末都未达到父末态目标，没有“先达目标再跌破”的域；自然和遥感有明显后续净增益，医学和金融的下降也保留。不能把全部缺口归于域后遗忘，直接固定域末快照会失去部分有益变化；不据此断言容量不足或回放无效。医学／遥感含多个task，域末不等于严格刚习得；父末态不是匹配阶段控制，不拼不同checkpoint的宏分。原固定点仍关闭。[全部逐题、重建证明与费用](summary/domain_end_function_20261025/v1/INTERPRETATION.md)。

原OSFT阶段刷新对照：prefix与Fixed完整退出，Refresh因Errno122账户磁盘配额超限失败；原controller／worker已无，三个故障收据.tmp为空，旧queue仍running已失真。没有VERIFIED，不消费部分质量，也不判刷新有效或无效。[终态证明](summary/ffn_static_delete_20261025/v1/STAGE_REFRESH_TERMINAL_OBSERVED.json)。另一个门梯度定位点在联合完整性断言失败，没有删除质量；它与本次已完整测得的功能阴性分开。[实施历史](summary/ffn_static_delete_20261025/v1/README.md)

## 完整原生紧凑载体的学习与运行峰

原生整体预训练保留并不足以自动保质／降低GPU峰。固定原1213FIT／19阶段／87更新，普通q/v819200坐标、GPU AdamW，初100与末100统一评分：

| 域 | 历史完整父末态 | 原生初态 | 原生QV末态 | 末态−初态pp |
| --- | ---: | ---: | ---: | ---: |
| 自然 | 61.500000 | 92.500000 | 95.000000 | +2.500000 |
| 医学 | 44.500000 | 12.500000 | 21.885965 | +9.385965 |
| 遥感 | 45.000000 | 40.000000 | 45.000000 | +5.000000 |
| 金融 | 30.000000 | 10.000000 | 10.000000 | +0.000000 |
| 交通 | 37.959937 | 16.151667 | 18.071713 | +1.920046 |
| 五域宏 | 43.791987 | 34.230333 | 37.991536 | +3.761202 |

完整权重507482304（−43.214033%），实际包1015142664bytes（−43.211943%），无额外完整模型；五个同卡32-token窗峰为2478.055／2931.584／2931.584／2931.648／2477.592MiB，父为2171.565／2046.315／1862.113／2046.310／2171.557MiB，全部更高。原生13／17个512tile、832／1088视觉token；更少decoder token没有带来更低本次峰，尚未分离实现／激活原因，不靠换后端或缩图救跑。

自然高分主要已在作者初态；其它三域从较弱初态学习后仍低父，仅初末读数不能判纯习得／遗忘因果。父43.79是历史已探索DEV参照，不与正式主表混排。原配置恢复失败后只保留标准加载对象，公开配置严格相同；旧87更新不重做，原评价绝对截止未延，最终210生成／200评分完整核验。原起点到终态2176.544260秒含失败、审阅等待；成功phase和1796.685585秒为子集，档案13.001080秒另计。固定点关闭，无新框架、无正式Ours。正式基线最近仅元数据实查CL-MoE／SMoLoRA进程与评估文件仍推进，完成仍8／16。[完整结果、负结果边界与成本](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md)。

## 新会话接续

1. 先读本README、[补充说明的交接段](SUPPLEMENTARY_NOTES.md#新会话运行交接)和[AGENTS顶部](AGENTS.md)，以最新人类目标为准。
2. stage-refresh、直接8-FFN、小视觉接口及本固定词面证据通路均已终止，不继续等旧PID。后续必须检验新的功能承接问题，不能把相对低损伤排名说成参数无用，也不按已见DEV缩小删层幅度救分。
3. 人类目标仍为各域F1对最强基线略升、完整权重与GPU/内存显著有利。已有技术可借用；必须由真实功能与适应对照支持，不能仅靠工程准入或权重排名。
4. 正式基线保持原运行；只有新sealed proof才更新正式表。禁止重复旧评分、配对、参数／谱／梯度审计或泛读填时间。

项目权威仓库在NEW `/data1/liyapeng/program/LLMCL` 的Git main；本机为非Git镜像。模型、数据正文、图像、特征、checkpoint、密钥和作者源码均不入Git。详细运行身份、SHA、历史费用、故障和关闭分支见补充说明与原证据，不能从本文的历史快照推断永久存活。


</details>


### 匹配GPU朴素r8：容量对照的锚已完成

2026-10-25原两phase完整退出后唯一评分，100已探索DEV宏38.7753697307，五域60.6666667／48.3571429／30／20／34.8530391。wide45.0859628544较它高6.3105931237pp，但4,399,104对540,672是约8.136倍训练容量，不能归给调制；原匹配wide胜动态/固定的反证保留。此为新GPU实现r8参照，不覆盖历史39.870439，不声称完整TEST收益。原值和逐题正负均保存，0旧质量重评分。

实际1213FIT／19阶段／87正LR更新，105新生成／100唯一F1／0TEST，96训练张量、梯度2162688bytes、GPU AdamW4325760bytes。完整fold模型893675552系数／1787351104权重bytes，普通buffers9032；最终完整重载通过。模型951.875543319秒，controller时间重叠不加；唯一核验12.578896171秒，归档46.309331444秒及实现/准入/运输另计。155个新产物逐SHA核对已私有存NEW，OLD scratch保留，无QA正文或checkpoint入Git。结果consumer仅汇总既有标量，0模型／0评分。

[结果与全部域](summary/plain_r8_gpu_control_20261025/v1/FINDINGS.md) · [完整VERIFIED](summary/plain_r8_gpu_control_20261025/v1/VERIFIED.json) · [结果审阅](summary/plain_r8_gpu_control_20261025/v1/RESULT_REVIEW.json) · [归档收据](summary/plain_r8_gpu_control_20261025/v1/ARCHIVE_RECEIPT.json)。新方法借鉴I-LoRA，未围绕OSFT另开机制拆解；固定r8/无新增门参数的实际收益仍待三臂实验。与朴素对比训练GPU身份不同，回放和EMA整包曝光差不能归给门。


### 答案读出门：三臂固定实验首次启动

2026-10-25预注册提交aa100a7fb40779833ee9f46428e107983ac074e3先于激活。READY e654eb6f0c0599ef76022711866ff5e5b7061d129c1cf7f57565eb2ff36ff2fe；原8科学源、正式协议、核心交叉／root／最终准入审阅均绑定后，才在OLD `/tmp/llmcl-answer-readout-cl-1084-ft3em1z0` 启动 `llmcl-answer-readout-cl-20261025.service`。unix1792924605.145713快照unit active/running，gate worker3992422/start762302299/UID1084/PGID同PID、GPU2身份真实匹配；其余phase尚未STARTED。只是时点快照，无完整质量结果、无永久live声明。

顺序gate/original/shuffle在同GPU2训练，各5400秒；随后同GPU1逐臂评价各600秒；总18000模型秒、外层18300，从各首次STARTED累计，不重置。每臂1213 current FIT／886 past-only replay／87更新，三臂8955次训练decoder请求、2658次额外target-head、3639次组合loss backward；最终315生成／300唯一新评分／0TEST，全部六phase完整成功后才消费质量。冻结slow540672 FP32系数/2162688bytes，working540672训练坐标，新门0参数；最终只导出working常规GPU fold，完整893675552 FP16系数设计待实际证实。不能把小EMA成本称完整训练内存上限。

这是借I-LoRA检验固定参数预算下的选择性约束，不再把OSFT机制当主方法前提。只胜原法而未胜同数shuffle，不能支持样本对应选择；shuffle实际Hamming为0时也不识别selection。仅末态DEV不能区分习得/保持，模型失败记实施缺测；没有正式胜最强或科研目标完成。原Naive旧标量复制，无重评分；相对Naive的整包差还含回放/EMA及训练GPU身份差异。

源码交叉审阅在READY前修了两处真实接口风险：prepare_update原调用带processor，补形参并校身份；teacher输入embedding hook可在no_grad内留下leaf requires_grad标记，先detach再检查，保目标数值/门/dropout不变。原源码SHA与修正后SHA及作者均在CORE_CROSS_REVIEW记录，0真实模型/旧质量影响；不包装成首审无错。元数据准入0.274581533秒，0模型请求；实现/审阅/运输另计。首模型故障/超时关闭，不默默重跑、延长、改rank/LR/gamma/memory/门阈值或精度。

[正式协议](summary/answer_reliability_consolidation_20261025/v1/PROTOCOL.md) · [核心交叉审阅](summary/answer_reliability_consolidation_20261025/v1/CORE_CROSS_REVIEW.json) · [最终源码准入](summary/answer_reliability_consolidation_20261025/v1/ADMISSION_REVIEW.json) · [实际激活](summary/answer_reliability_consolidation_20261025/v1/ACTIVATION.json) · [时点观察](summary/answer_reliability_consolidation_20261025/v1/LATEST_METADATA_OBSERVATION.json)。

同轮NEW基线实际进程/队列快照在BASELINES_BEFORE_ADMISSION：原CL-MoE/SMoLoRA PID8260/8265仍匹配且CPU ticks推进，主1221743 active/running，GPU1/2/3当时均有进程，GPU0不由我方分配；其它5项待空卡/原reload，FOREVER原队列保持。没有新增sealed正式证明，主表8/16、FOREVER和8组存储原值不改。


### 全体最强Qwen基线的资源空白已补齐

原formal Qwen SeqFT stage19直接重载、原混合dtype/后端/processor与generation对象，五条相同raw QA、物理GPU1、各32新token，峰4311.996–4954.604MiB；匹配LLaVA朴素r8逐窗低55.147769–58.803849%。模型68.625645367秒，唯一metadata verify0.012348259秒，5生成/160token、0F1/NLL/训练。完整2,210,075,136系数/4,422,329,344权重bytes、普通buffers7504，原CPU cache元数据不是RSS。该结果只证明原实现这五窗，主要跨骨干差异，不证明新门质量/结构创新或最优fold。

按独立展示审阅，只将原Qwen Seq/PAS正式行GPU列从未测改4954.604†/2959.415†，列名注明固定五窗、脚注完整口径。全部18正式方法行的其它16列逐字符串相等，包含原F1/AP/FWT/BWT、8组保存态存储和FOREVER；不是将FIT模型峰移植进formal行，也未填Ours。主表整段字节随新真实指标合法改变，旧“整段不改”保护不再阻止补测；更新前后hash与逐列核对见ROOT_RESULT_BINDING。

原CP已从NEW canonical私有运输至OLD/tmp，原件/OLD副本保留；仅模型原加载器全hash一次。运输40.240818748秒、准入0.228828288秒、实现/源码审阅/发布/结果复核另计；客户端准入字段KeyError发生于远程/模型之前，同提交push ref竞争核实远端已成功承接后才启动，细节均在FINDINGS。首次真实模型完整成功，无版本救跑或新精度/后端；ARC三臂在另一GPU继续，未更改。

[全部窗口和解释](summary/strong_qwen_resource_20261025/v1/FINDINGS.md) · [完整核验](summary/strong_qwen_resource_20261025/v1/VERIFIED.json) · [结果审阅](summary/strong_qwen_resource_20261025/v1/RESULT_REVIEW.json)。


### 固定r8三臂的模型前资源拒绝与限定续接

gate已完整19阶段／87更新、1408.252424500模型秒、EXIT0；下一original的3999463在Context组合GPU资源断言被拒，未到STARTED/Torch/CUDA/数据/模型，0QA。原拒绝未记录具体子条件，当前空卡不能追认原因为NVML残留或他人占用。两agent按源/trace/身份判断这是模型前准入，非方法阴性；原科学协议的首模型故障关闭规则不应被扩大成因排队直接放弃。

已公开事前补正753d8288ebd2df6bf77f44041926d78e91717421：原4失败文件按原字节SHA/路径映射保留于独立attempt目录，旧unit仍failed；8科学源/READY/参数/数据/随机顺序不改，gate不重训。外部stdlib监督仅续接五个未模型启动阶段，原空卡谓词连续两次相隔2秒满足后才启动，等待/维修仍计入原绝对截止2026-10-25T15:41:42.561657+00:00，不延长。最终续接receipt将原失败和实际六phase／原一次300分核验连起来，不能伪装无故障。

unix1792926762.8731668新unit `llmcl-answer-readout-cl-prestart-continuation-20261025.service` active/running，original worker4001502/start762499983在GPU2身份匹配，已11梯度见证；这是时点证据而非完整F1。新Qwen资源probe已退出，GPU1释放，不影响其余eval顺序。NEW基线同轮实际仍运行两条，其它原队列待空卡。主表只补既成功的原formal资源值，不读本三臂partial质量。

[具体续接与唯一消费者](summary/answer_reliability_consolidation_20261025/v1/CONTINUATION_HANDOFF.md) · [原故障实查](summary/answer_reliability_consolidation_20261025/v1/PRESTART_FAILURE_OBSERVED.json) · [独立源码审阅](summary/answer_reliability_consolidation_20261025/v1/PRESTART_CONTINUATION_REVIEW.json) · [准许范围与原预算](summary/answer_reliability_consolidation_20261025/v1/PRESTART_CONTINUATION_PERMIT.json)。


### 固定五窗资源图与答案门先例核查

从三份既有sealed proof消费15个GPU标量，完成朴素LLaVA-r8、原LLaVA-PAS、原Qwen-SeqLoRA的300DPI PNG／矢量PDF。逐一核验原proof、附属memory／COMPLETE摘要、相同物理GPU／boot、五个raw输入及32新token。同LLaVA prepared输入相同；跨骨干保留各自原生展开，不误要求tensor相同。图明确fold／原dtype、每域单QA、非全TEST峰；不将朴素图冒充ARC实测或Ours成果。沿用项目grouped_bar样式；CPU仅制图和已记录标量，新增模型／QA／评分均0。root与独立子agent均实际查看PNG，15值和图注复核通过。主表所有原行与参数／存储／GPU单元格保持原字节，新增读者图入口。

定向阅读发现LSSD（EMNLP2022）已有样本相对教师表现开关，DA-KD（ICML2025）已有teacher/student gold CE比值指标，ACL2021已有随机50%同量控制；不声称一般gate、gold-loss选择或随机同数思想原创。LSSD以token概率算术均值评分，不能说等同本文平均NLL严格比较。当前具体适配只对实际max-hidden目标的末层读出作整包MSE准入代理；不能从代理NLL改善推出正确梯度或自由生成收益。已有方法允许借鉴，先例不独立否决本实验，但贡献须由完整效果建立。

独立论文方向评判：可以研究固定小容量的有效学习；最终内存优势须拆清骨干、专家表示与dtype，不归新增门。若门不能胜原法及实际改变对应的同接受数打乱，不保留为新增规则；更好的已知方法可如实选择。仅100反复探索DEV，尚未证明正式胜最强。同轮基线实查NEW主监督和CL-MoE／SMoLoRA真实身份仍在，GPU1为他人作业，5项原队列等待，不以无新sealed结果伪造完成度。

三臂训练仍由原固定监督推进，本次不读partial质量、不修改科学源／READY／预算、不新增模型分支。后置消费者只完成源码审核和准备，必须两份完整VERIFIED绑定后才运行，0新评分。实际结果出来后按原协议正负全记。

[资源图与精确数据](summary/edge_resource_comparison_20261025/v1/FINDINGS.md) · [独立图审](summary/edge_resource_comparison_20261025/v1/FIGURE_REVIEW.json) · [定向先例](summary/answer_reliability_consolidation_20261025/v1/RELATED_WORK_SCOPE.md) · [论文方向评判](summary/answer_reliability_consolidation_20261025/v1/PAPER_STORY_REVIEW.md) · [后置脚本源码审阅](summary/answer_reliability_consolidation_20261025/v1/POST_CONSUMER_SOURCE_REVIEW.json) · [本轮基线实查](summary/answer_reliability_consolidation_20261025/v1/BASELINES_DURING_RESOURCE_FIGURE.json)。


### 参数量优势与dtype影响的分离；低更新替代思路不盲开

对已sealed PAS／Qwen完整布局作精确整数账本：PAS比LLaVA单图额外140,771,328个神经系数，原为FP32；将它们仅在纸面统一按2bytes计，原增加量约537MiB中268.5MiB来自精度溢价，另一268.5MiB来自额外系数（336个FP32 mask另1,344bytes）。这是固定表示、每系数至少2bytes且不重编码的条件账本，不是转换成功、GPU峰下界或精度无损。Qwen原FP32 A/B只1,089,536系数，条件精度溢价2.078125MiB；fold为原FP16基座的纸面载荷仍4,417,971,200bytes，相对LLaVA差异主要来自骨干。不能把精度溢价从实测GPU峰相减。root用sealed COMPLETE／VERIFIED及原代码SHA独立核算，0模型／QA／评分，无CP载荷读取。

源核查还显示PAS原a路由随专家A.dtype计算，且masked_fill使用−1e9；HostedLinear禁autocast。原FP16副本不是经证实的无改forward合法参照。当前不追加halfcast／数值路由改写，现约25%GPU优势限定原实现／dtype；若后续研究要作独立于精度的更强主张，必须另有实际证据，不能由纸面乘法作保证。这项分析不会挽救质量阴性或替代ARC自身测量。

等待当前完整结果时另核一次“少更新能否靠已知梯度初始化提高固定r8习得”。只查LoRA-GA一个原法，历史函数补偿／固定SVD系数不等于真实task梯度对齐，因此不编造已被直接证伪；但现87更新／wide增益也没有识别随机子空间为根因。仅S1的梯度先验有早域偏置，逐stage fold再重初始化则改变累计可表达更新集合，不能借同训练坐标数冒充同适应容量。该工具当前不立项，不启动新协议、源码或GPU点；由已有三臂完整回答先决定是否有必要，不用泛读或梯度计数代替证据。旧RITE最终负结果本地未找到对应完整proof，明确未确认，不恢复旧分支。

[精确载荷账本](summary/edge_resource_comparison_20261025/v1/PRECISION_ACCOUNTING.json) · [独立精度口径评判](summary/edge_resource_comparison_20261025/v1/PRECISION_FAIRNESS_REVIEW.md) · [低更新替代思路核查](summary/answer_reliability_consolidation_20261025/v1/LOW_UPDATE_ALTERNATIVE_REVIEW.md)。最新基线实查仍为CL-MoE完成16阶段／第17阶段评价与SMoLoRA原初评，实际身份存在；没有新增完整sealed组合，不增主表完成数。


### 主表补显五域宏，保留AP19与全部原结果

独立核查8份正式sealed全精度标量后，在最外层README主表AP前新增“五域宏F1”，原AP列标注AP19。五域宏是5个完整域内QA均分等权，AP19是19个原生任务等权，不互换，也不把21454题直接池化。8个值从原proof的domain_macro_final逐值复制，FOREVER读取合法shared_V2_proof；未完成10行继续空白。全部18方法行原17列逐字符串保持，含FOREVER、域F1、AP/FWT/BWT、8组存储、现有GPU和未测项。没有新评分、预测读取或完整组合完成，Ours仍空。

Qwen原sealed宏48.9723511766575与既有FORMAL_ANCHORS的48.972351176657504相差1ULP，原精度与旧锚分别记录，不改任何阈值；3位显示同为48.972。此为读者展示澄清，不是新科研质量结果。具体源路径、摘要、域全精度和展示值见FORMAL_MACRO_VALUES。

[8组原宏来源](summary/edge_resource_comparison_20261025/v1/FORMAL_MACRO_VALUES.json) · [原17列保全核对](summary/edge_resource_comparison_20261025/v1/MACRO_DISPLAY_BINDING.json)。


### 固定r8三臂完整结果：共同收益明确，新增门只有微小点差

六个模型phase已完整退出，原300回答只评分一次。五域宏naive 38.775369730699、original 43.703262379384、gate 43.719890663435、shuffle 43.671936498450。门相对原法+0.016628284051pp、相对打乱+0.047954164985pp，4高2低94平／3高1低96平，所有差异在交通，80个其它域逐QA F1相同。425／886接受、shuffle对应Hamming392、59个step改变，事前非退化／同数／严格点方向满足，不改写成阴性，也不夸成显著、稳健、纯机理或正式优势。共同配方相对naive约+4.93pp，遥感+35、医学−10；只有终态不能区分习得与后续保持。完整stage／域NLL、原MSE／接受、逐题正负均在OUTCOME，描述性统计不冒充有效梯度。

每臂working540672 FP32训练坐标，冻结EMA另540672，新增门0；梯度2162688bytes、GPU Adam4325760bytes。最终完整图各893675552个FP16系数／1787351104bytes、普通buffers9032，实际5窗baseline／allocated峰／reserved与朴素r8逐值相同。相对PAS／Qwen的已测原表示资源优势可保留，来源不是门；相对朴素和原法没有新推理减重。每臂1213current／886replay／2985decoder，三臂8955decoder／2658额外head／261更新／315生成／300新F1／0TEST。六phase总4564.060149965秒，模型完成累计5148.511217594秒，全部修复／核验完成累计5798.325146437秒；重叠成本不相加。

原preSTARTED资源拒绝仍归档，两个旧服务最终failed/1均保留。其后六模型已成功，但原verify403索引了producer未保存的phase而在VERIFY_STARTED前KeyError，0评分；这是我们源码审查遗漏。独立review后于d6b9ed8事前公开一次外部AST修复，仅将该Assert替换成明确phase缺失、保全部refs/cap检查，逆变换恢复其它全部原AST；8科学py与6项冻结文档、READY、模型和生成未改，未写假phase。原错误两文件原字节另归档，新元数据核验服务成功，原评分公式与300次首次评分不变，修复收据和扩展CV明确披露两个失败服务及成本。原绝对截止未重置。Python3.11／3.12的整树AST dump哈希可不同，修复以原源字节及同进程逆变换为准，没有凭跨版本hash放宽科学检查。

本地后置consumer另把READY14个绑定项（8py＋6文档）误当8个总项，读stats之前失败，0输出／评分；改为精确核8个py集合后唯一完成标量消费。收集器首次遇客户端重建ACTIVATION的CRLF与RUN原字节不同，两个JSON值严格相同；旧客户端字节单独保留，原位置保存RUN精确字节，receipt列双SHA，不改服务端记录或科学数据。此后只恢复文件取回，没有重模型或重评分。

NEW私有归档732个新输出逐SHA通过，含所有完整CP、19阶段快照、生成和两故障archive；134.355476790秒，OLD原目录保留，权重／QA不入Git。独立结果review接受证据与限定；两位评判均认为微小门差不足直接确立为论文主创新，不设新的逐域floor或显著性硬门。保留已知共同配方与最高点候选，下一具体问题是已存状态中的习得／后续变化，尚无新点准入。

[完整结果](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md) · [独立结果审核](summary/answer_reliability_consolidation_20261025/v1/RESULT_REVIEW.json) · [方法选择判断](summary/answer_reliability_consolidation_20261025/v1/DECISION_REVIEW.md) · [核验缺字段修复](summary/answer_reliability_consolidation_20261025/v1/POST_MODEL_SCHEMA_VERIFIED.json) · [私有归档](summary/answer_reliability_consolidation_20261025/v1/ARCHIVE_RECEIPT.json)。

## 1%与10%的新习得／监督先验比较（2026-10-25）

人类进一步要求“可以看看10%会不会也这样，有可能是数据太少”。四个10%正式run仍在OLD执行初始完整评价，尚无当前任务刚习得质量，不能用TRAIN计数代替模型结论。后续比较同方法／同骨干的task训练前、刚习得与增量；金融同时看单／多选／数值F1和集中错答。单轮更多数据也增加更新预算，不称纯数据量因果干预。

已唯一完成TRAIN-only审计：1%医学99＋金融89、10%医学984＋金融887，0TEST、模型、生成、更新或评分。金融单选A/B/C/D由22/18/7/6变为189/163/99/76，标签偏斜减弱但没有消失。计数不证明监督偏斜导致错误；保留全部异常及完整分母，没有新数据排除／改答案。实际审计0.412201秒，外层0.421534秒与其重叠不加。独立审阅接受该描述边界。

[结果解释](summary/train_answer_prior_20261025/INTERPRETATION.md) · [封存汇总](summary/train_answer_prior_20261025/AUDIT.json) · [独立审阅](summary/train_answer_prior_20261025/RESULT_REVIEW.md) · [跨比例功能对齐](summary/train_answer_prior_20261025/ACQUISITION_COMPARISON_PLAN.md)。

10%完整结果核验器已先发布后启动，默认实机discover为0候选／0评分；OLD CPU4063535仅等待完整native run真实exit0后进行一次原评分核验，不运行模型或修改Markdown。原单轮公式实际828次optimizer调用=822次正LR＋6次合法warmup零LR首步，修正的是新核验器继承的1%过严断言，未改训练。FOREVER完整动态调度两源仅WIP，尚未GPU验证、未运输／激活，两组合继续defer。旧1%CL与SMo实际PID身份仍匹配，新服务器GPU0不由本队列分配。

[10%核验源码审阅](summary/two_backbone_pct10_20261025/completion_observer_native/SOURCE_REVIEW.md) · [实际0候选入口与激活](summary/two_backbone_pct10_20261025/completion_observer_native/ACTIVATION.json) · [FOREVER WIP边界](exp/two_backbone_pct10_forever_dynamic_20261025/IMPLEMENTATION_NOTES.md)。
