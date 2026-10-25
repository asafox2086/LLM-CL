# 最小成对扩展协议草案：未冻结、未准入、未运行

本草案只验证METHOD_PROPOSAL中的QA级末层读出代理门。固定完整OneVision、同q/v-r8/alpha32/dropout0.1、540,672个working训练坐标；不扩容，不改原1213 FIT/19阶段/87次更新、答案目标、stage LR、GPU AdamW或GPU FP32 fold配方。全部训练在同物理OLD GPU0串行，终态评价在同物理GPU1；不因卡闲增加分支。独立Naive-r8由已有控制提供，禁止重复训练。

三个新分支为直接移植I-LoRA、target-gate、reference-mask shuffle。三者共享fresh初始化、current QA顺序、reservoir抽样账本、CE归约、slow初始化与更新，以及原整包隐藏态MSE。固定gamma=1；EMA方向为`slow←0.25*slow+0.75*working`，在每次实际optimizer更新后仅一次，不按micro1 QA更新。该系数对应官方`min(1−1/(step+1),0.25)`在首次真实step及之后的值。每阶段开始slow复制该阶段working起点，作为源中阶段同步意图的共同可执行定义；源的default/ema同名匹配与函数名错误不原样继承，不据此评价原法。末态只保存/评价原default即working，禁止按质量改选slow。

共同memory约定采用论文的历史记忆语义：每个原task训练完成后，才将其FIT按固定seed纳入容量500 reservoir；当前task只从已完成task的缓存抽样，不回抽本task，0未来FIT。回放数不超过该次current QA数，缓存不足则取实际可用数；首task无历史回放，仅CE。使用独立固定数据RNG封存插入/替换/抽样账本；三臂逐update回放ID与顺序一致。该共同阶段移植明确不同于官方代码的“当前QA先入库再抽样”以及文本特定C-STANCE跳过规则，不是新门贡献，也不能叫官方源码逐字节复现。不得把1213 current QA计为总模型曝光：回放QA、working/slow forward、backward、额外head、500缓存的实际QA/存储和时间分别计数。

所有臂保同训练模式与dropout；slow no_grad，复用已有working/slow forward结果。原MSE覆盖embedding及各层隐藏态，batch1/micro1时先对该QA原完整有效packed位置和通道归约，再按QA等权累积，三臂一致。门仅用末层post-norm目标及原答案预测位置，逐答案token平均NLL，严格`q<p`，相等关闭；无新增norm/head参数或decoder。直接移植分支使用全1 mask；三臂都可计算同一诊断head量供计费和观测，避免把少计算混入门效果。

先盲态完成target-gate训练，封存每optimizer update的回放QA ID和gate位向量，不消费任何生成质量。shuffle随后逐当前update读取该封存向量，用独立固定seed42排列，重演在自己的参数/EMA轨迹上。每步接受数量与reference逐值一致；不要求两臂自己的NLL门率一致。micro1即时backward，无需保留双图或额外探针forward；仅需该update的标量mask。shuffle是reference轨迹的反事实控制，不是可独立在线部署的方法，也不是使用未来阶段信息的候选。直接移植不读取mask。三臂一致保留原全部回放分母，不能按接受数放大MSE。

三臂训练全部退出后，才统一消费预定终态100个已探索DEV的自由生成F1，五域宏为主、逐域/逐题完整报告，不用DEV调配方。不新增初态100或阶段质量选择；仅最终差异不能区分即时习得与后续保持。每终态实际working-only fold、自包含保存/严格重载，再测同卡同五题32-token资源窗口；全量训练buffer/EMA/回放成本与推理成本分列。现阶段不授权正式TRAIN/TEST；具体phase秒数与输入/源绑定由实施协议一次冻结，本草案不构成启动许可。

解释出口：gate高于直接移植且高于shuffle，才支持本固定配方的样本对应准入净效用；只胜直接移植不胜shuffle，不能区分选择质量与弱正则效应。Naive胜出则不包装新门有效。门全关或全开须报告退化，不作为机制阳性。所有分支和域的结果保留，不滚动救LR/rank/gamma/memory/阈值。单seed、每域20条探索DEV是点估计，不能称显著、等效或胜正式最强；最终完整图相对普通fold LoRA/I-LoRA同规模，强基线总体F1和实际资源优势仍需独立正式证据。
