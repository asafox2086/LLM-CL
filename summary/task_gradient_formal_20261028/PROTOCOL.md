# TaskGradReuse：独立fresh 1%正式确认v1

本协议在模型启动前冻结；实现经独立源码审阅，但尚无GPU预检或正式质量结果。完成元数据准备与原allocator接入后才请求阶段保存让卡。

## 固定方法与数据

LLaVA-OneVision-Qwen2-0.5B-OV，fresh原预训练W0，seed42、原1%完整1313 TRAIN与21454完整TEST、19任务、原一轮/微批1/有效批16、原QA等权answer loss、AdamW、学习率/裁剪/调度、原输入/视觉缓存/生成/原Answer F1保持。沿原manifest排序，总93主更新，所有stage warmup0。原manifest SHA36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777。不使用探索FIT/DEV划分或任何旧后缀权重开始正式运行。

唯一新方法TaskGradReuse：q/v-r8、alpha32、scale4、dropout0.1、FP32 A/B。每任务前将上一BA按GPU FP32合并到原FP16 W；用当期TRAIN原顺序前min(16,n)条的QA等权dense q/v梯度，执行FP32默认torch.linalg.svd(G,full_matrices=False)，A=(1/sqrt3)*Vh[:8]，B=0。两因子均参与原主优化器。总额外222次校准反传，无dense optimizer更新，不增主更新或看未来任务。

方法定义是**实际默认SVD返回值**，不是满足原1e-4门的理想正交构造。保留有限性、形状、B0、状态/RNG恢复与真实更新硬验，逐层报告范数/正交误差，不将旧失败改成通过。B0意味着每task首步A梯度为0；单更新任务不学A，是本配方的一部分。阶段fold有FP16舍入；累计更新不受全程单rank8限制。

校准只在GPU反传，暂时解冻22020096个q/v dense坐标、G载荷84MiB；恢复参数dtype、mask、模式、RNG和完整状态。主optimizer为540672个A/B坐标；不永久保留G/SVD、历史参考adapter、专家库或路由。最终stage19主更新后fold为单dense图，再按原协议评估和保存，不把未fold最后图的分数冒充部署图。

## 完整评价与资源

沿原正式S0/各stage当前与下一任务/S19全任务的74格评价，不缩TEST、不改原任务生成长度或评分。报告五域最终F1、五域宏、原AP19/FWT/BWT；科学解释聚焦新习得，额外从已有R[t,t]汇报19任务刚习得与QA加权五域，不另跑模型或展开遗忘分支。是否成功以完整质量—实际部署取舍判断，不能由某一金融小集替代。

对比原同骨干SeqFT-LoRA、PAS/OSFT及其它完整正式基线。最终完整graph/配置/processor/来源需保存，并独立进程重载；同原五输入/每窗32新token实测GPU allocated/reserved，记录完整系数、真实dtype权重字节、相对基座新增参数/状态、常驻bank及可用的延迟/速度。训练峰、校准时间和反传成本单列，不把其数值当部署成本。正式前不预填新模型资源。

## 执行与保留

只接现有OLD分配器的单卡研究优先链，不另建分配器，NEW GPU0规则保持。科学源/数据/主更新/评价范围固定；新正式运行以这些有限工作量结束，不沿用旧探索5400秒期限，也不重置/续旧失败run。先走原GPU preflight及完整checkpoint重载入口，再fresh正式运行；预检权重不用于正式起点。启动后确认真实身份/GPU再分段sleep，原生阶段checkpoint和日志持续保留。

新的runner每完整stage保存完整checkpoint、支持显式阶段暂停恢复；最终S19不可删除。已完成前期checkpoint按人类允许的保留策略处理，主表的原始答案、分数、配置、阶段证明和最终模型记录不得删除或覆盖。研究模型结束释放卡后恢复原暂停基线，分析期间让基线继续运行。

监督顺序固定为原GPU preflight（1800秒）、独立原checkpoint_probe重载（600秒）、fresh正式training。正式阶段由原19阶段/93更新/74格的有限工作量终止，不设置沿用旧探索点的计时限，不自动重试、接旧权重或追设置。完整来源见同目录BINDING；controller只监督本研究精确身份，不选择GPU、不操作基线。正式结束释放卡、接续基线之后，再按上述固定范围做唯一完整核验与独立最终资源测量；资源完成前不称联合目标实现。
