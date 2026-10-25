# FOREVER动态调度：唯一必要GPU检查

当前仅源码审阅，0模型／CPU模型／QA／服务器操作。S2原预检为107 QA／7更新，只能覆盖短阶段末回放，不能证明新增的>24分支。GPU真正空闲后仅做下列一个disposable preflight→原完整CP→独立reload流程，不叠加toy门，不占用正在运行的基线。两骨干使用同一门型；未实际执行的骨干不宣称已通过。

**固定合法输入。** 绑定已冻结10% manifest及原TRAIN文件SHA。取原T13 presence按原`SHA256(42:13:ID)`顺序前385 QA，构成25次current更新（24×16＋1）；原T1–T12各取同一原顺序首1 QA，构成12条可供原selector选择的past-only history。19任务编号／原指令／图像／问答不改，仅形成显式标记的独立诊断view：T13为current、前12task各1条、后续task不提供QA。绑定所选ID／view SHA，不能把view计数／状态冒称正式13105轨迹；不要求模型已完整学习这12条history，不据此作学习质量判断。原selector仍实际运行，不按答案挑memory或手工设置其结果。

**一条真实顺序。** fresh原基座及原FOREVER方法，用原current optimizer／cosine／GradScaler／micro1／clip和完整25更新，原`after_update`每次调用。第24步校准24-step drift后，须在未耗尽current时自然触发并完成至少一次dynamic replay，随后确有第25次current前向／反向／正LR更新，再执行`end_stage`的final replay。不改tau、model_day、threshold或beta强迫触发；若未自然触发，只是本门未覆盖目标分支，不能写通过或科学阴性，也不能滚动换输入。动态及final都直接调用原`execute_replay`：两epoch、HF linear／5 warmup；保留实际memory M、2M曝光、零warmup步和正LR步、真实参数移动及原AMP unwrap两次TRAIN NLL一致证明。回放前后drift step／tau／mu不因回放推进，reference更新；当前optimizer／scheduler／scaler在中途回放后继续，不能悄然重建或跳过最后current。收据绑定current顺序、25次更新、dynamic→current25→final事件次序及最终`core=None`／scheduler summary。

**完整图重载只做一次。** 复用原`checkpoints.save`保存method＋processor＋RNG完整私有图，以已使用的T13 TRAIN QA作probe，随后原独立`checkpoint_probe.py`／`checkpoints.load`实际重载，核原NLL容差和有限backward（0新optimizer更新），包含动态回放后的可序列化AMP清理结果。不只导出A/B，不调用TEST evaluate，不生成或评分。此为原preflight checkpoint路径，不伪造前12阶段events去调用`rolling_io.commit_stage`；原rolling合同仍是stage-atomic提交后才能恢复，未保存stage内current optimizer，因此不得声称验证了中途断点续训。正式从fresh另起，绝不接续本门CP／RNG；原probe删除disposable CP的行为如实保留。

**固定费用／资源。** 每骨干单卡从首次加载至reload结束上限2400秒，包含输入准备、385个current decoder前反向、两次回放的实际2M＋2M decoder训练曝光、每次unwrap额外2个TRAIN NLL、保存probe、reload NLL／backward、完整CP序列化／SHA及资源记录；0TEST／teacher／F1。实际按日志计量，不把HF零LR步／重复history／检查前向省略。报告原基座、可学参数、task-entry theta_star／drift副本、current与HF optimizer同时驻留、CPU raw-gradient／前后参数副本及trace磁盘字节；不称训练坐标就是推理权重。

现adapter每次回放前reset CUDA peak，且current事件在`after_update`之后写：**stage末单个peak不能保证覆盖全部current。** 若要声明完整门峰，应在每次reset前保存旧peak并与回放／继续current的scope取max；否则只能分scope报告或明确整体峰缺失，不拿采样或末峰冒充。`rolling_io`的fsync／commit在RAM tmpfs仅代表该文件系统事务边界，不是断电耐久；NEW collector latest／final也不意味着全部阶段CP归档。

本门只准入固定基线技术路径，不是自研方法效果或正式FOREVER成绩。原1%源、已运行native组合及冻结10%初版三源不改；新动态目录的代码／诊断view／预算先绑定后执行。任何真实失败或覆盖不足均留原证据，不自动延时、续CP或换输入补门。

**实施前的集成更正（仍为0模型）。** 上文“以已使用的T13 TRAIN QA作probe”更正为保留原run的T19首条TRAIN probe及独立重载尾段；T13仅用于派生25步诊断。原run先建立stage1 core，再执行19个不调用`after_update`的单题诊断；新adapter须只丢弃这个未推进的调度器，保留已实际改变的权重，并记录它不是学完S1的前缀。单题诊断、T13 385题、回放及额外NLL、T19保存probe和独立reload全部纳入同一2400秒预算。此更正修复源码衔接前提，不更改正式TRAIN、任务编号、原run或模型参数设置；源码审阅不代表GPU门已通过。
