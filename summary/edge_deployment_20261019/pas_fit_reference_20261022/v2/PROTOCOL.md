# Fresh FIT-only PASs v2：重启封存后一次独立重做

本独立v2保留原科学条件和完整证据门槛。v1外部重启且逐步optimizer witness未落盘，不能用完整checkpoint伪补；只允许一次fresh重做。预算在本次结果前透明修订为v2最多4h、连同v1保守预留2h累计模型预算最多6h。详见[BUDGET_AND_REPAIR](BUDGET_AND_REPAIR.md)。旧v1不当原2h协议成功，所有费用/未知inflight保留，v2再失败则停止，不自动追加。

## 问题、已有证据与判别用途

ShortFixed原1213FIT/19阶段87步后的100已探索DEV宏42.30132909166463，严格高于原FIT-only OSFT40.78380056394307和q/v39.870438970139254，实际gradient/moment字节通过原判据；两对区间跨0，主要宏增益来自自然/医学各一题。这只保留可行性。当前已完成最强OneVision正式PASs见过全部1313（包括DEV），不能用于此筛选。独立必要性评审接受只新增一个fresh FIT-only PASs参照，解决是否值得投入完整确认，而非隔离短侧内部机制或证明1%独有原因。

## 唯一新条件及原科学配方

从同一原基座新鲜执行，继承原冻结SPLIT的1213FIT/100DEV、原19阶段/87有限正LR、seed42/epoch1/batch16/micro1/训练顺序、原TRAIN主体AST、图像/QA/提示/全原参考AnswerF1。T7/8/9/12没有DEV；同组FIT/DEV不混，未知近重复关系不保证。禁止官方TEST/正式1313 teacher/回放，不改路由/位置/rank/LR/提示/步数，不从DEV挑最优。

使用原pass_adapter.PASMethod及作者router/learner，不用旧部署推理剥离版。method.name保持PASs_MoE，原optimizer实际默认LR1e-4/HostAdamW/eps1e-8/betas(.9,.999)/weight_decay0，逐阶段新optimizer/scheduler/scaler及原floor warmup/cosine/clip。原七位置、r128/alpha256/dropout .05/两个专家、gate_rank16/tau1.5/router=a/tie=A_T、lambda1e-4/B_reg .1/top_p .4及其余作者默认值保持。FP16基座与外层autocast，FP32适配器内部禁autocast、训练内部CPU offload；公开这个作者端口数值例外，不宣称与ShortFixed的适配器计算严格同精度或将差异全归功算法。

保留同一专家/learner跨阶段历史_I_agg/_A_ref/_B_ref。before_train仅重置当阶段统计与tau；每微样本recording(True)→forward→False→backward→post_backward，保留原GradScaler除乘及进度start/len(order)。十个单批阶段该进度为0，不“修复”作者warmup。after_update保持阶段局部tau余弦，单步阶段一步即.7；end_stage聚合历史后才保存。设置private state_root和原save_checkpoints=False，仅关闭作者冗余持久化分支，RAM历史照常聚合，完整checkpoint保存method/learner图；作者I_snapshots仍计费。标量回调计数/状态元数据观察不改数学或RNG。

## 结果前固定比较、正负出口

只完整封存后读取质量。唯一primary比较ShortFixed−fresh PAS五域宏；将既封存ShortFixed42.30132909166463和原两个FIT锚固定，不重新运行它们。ShortFixed严格高于fresh PAS且仍高于原两锚、原ShortFixed实际状态字节门槛通过，才进入是否完整确认的独立必要性评审，否则关闭当前候选。即便通过也不是自动1313/21454授权；本参照无新候选方法、不按排名挑旧条件。

仅新增ShortFixed−PAS一对：新参照全部100ID/全原参考原score AST、逐域/逐题正负、整已知图像/场景组10000次/seed20261022探索区间，QA域内等权、再五域等权。ShortFixed只按原封存预测SHA复制既有分数，不重跑其旧两配对/bootstrap或旧F1；区间跨0只约束稳定性，不临时改点估计门槛。已探索DEV/单seed/小组样本不当新留出、显著/稳定/等效；末态不能分离习得/遗忘。全部负结果、逐域下降和费用保留；不填Ours或正式主表/FWT/BWT。PAS的实际梯度/优化器/历史状态及完整开销单列；即便候选点质量更高，也不能把仅对OSFT的状态字节下降写成质量和成本同时支配PAS，完整确认必要性另审。

## 投入、冻结与完成边界

仅NEW物理GPU1一流，最多占NEW三卡，保全他人及两机正式基线，不用GPU0/OLD空卡。v2 worker STARTED→COMPLETE总4小时含加载、原19FIT图文门槛38NLL+38最多4token生成、1213FIT/87更新、19完整checkpoint及持久SHA、100DEV、回调观察/保存费用。CPU必要小HF准入与独立核验另计，所有历史探索成本保留。不是部署工程或算法加速测试。

记录每步loss/实际LR/原始梯度及clip、19首步完整梯度、19validity、实际optimizer参数ID/gradient/moment字节、19阶段recording/post_backward/after_update计数、原learner已收usage/模块数/参考状态bytes、全训练冻结基座digest、DEV前后训练参数与learner历史digest、同PID原ResourceMonitor及全部保存。仅新checkpoint持久SHA核验后删除前一private权重；完整图不冒充部署重载证明。

源/协议经独立审阅，唯一小HF CPU准入将既有脚手架经过新每步持久化路径并四次严格重载，在进程导入Torch前禁用全部CUDA可见设备，并检查前后CUDA未初始化；验证原方法与仅计数子类初态forward/RNG、两阶段四更新loss/梯度/权重/learner历史精确、缺目录梯度保存重载及完整图checkpoint重载后再READY冻结。CPU仅准入非科学收益。READY后不热改；STARTED后故障留证停止，独立版本修复，不拼partial/降门槛/盲重试。只有STARTED/CUDA前资源变化75可未尝试wait重入。单流dispatch唯一原score/全部参考/权重和源SHA/19阶段87步/106梯度文件/回调及唯一新配对核验后正常退出。端侧只是背景，科研目标尚未实现。
