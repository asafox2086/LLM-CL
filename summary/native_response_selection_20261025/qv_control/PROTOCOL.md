# 匹配GPU宽QV-only：响应调制净收益对照

这是新独立的必要控制，不修改旧成对协议、PAIR_VERIFIED、旧五域判决或问题账本。人类已明确以总体Answer F1略高于最强基线、同时最终完整权重和实际GPU内存明显有利为目标；不再要求各域分别领先。当前动态/静态候选仅有已探索100DEV结果，保留候选不等于论文目标达成。本文件是固定实施协议，不是激活或运行成功记录。

## 问题与唯一干预

原动态44.757602698、静态44.681200950均较历史wide43.791987409高，但历史wide的HostAdam/CPUfold等执行条件不匹配，不能据此将增益归给新增响应模块。本控制只问：同完整OneVision、同GPU学习配方和wide-QV预算下，去掉完整响应调制包，最终总体F1和资源如何变化？它识别包含24标量、动态/静态响应分支及其学习作用的整包净贡献，不单独识别语义路由、奇偶分组或某条梯度机制。

仅一个fresh QV-only流程：保全原视觉、projector、语言、词表及所有原始参数/普通buffers；不安装beta、response_sign、calibration_scale或额外signed-Down。相同BudgetQV rank_plan的48个q/v线性位置、96个FP32 A/B、4399104个训练坐标，scale4、dropout0.1和原初始化不变。最终仅将q/v合并回原FP16线性，完整893675552系数/1787351104权重字节及9032普通buffer字节须实际布局确认；不把训练A/B、optimizer、旧预测或外部base隐存在部署图。此控制不提出新精度、算子或工程优化贡献。

## 起点、输入与学习匹配

原base采用与成对运行相同的标准GPU FP16/SDPA加载；完整state、所有buffer、结构/共享别名、功能config和generation公开值逐项绑定原train_dynamic_COMPLETE.original_base。state和buffer digest亦与原32FIT诊断COMPLETE绑定。先导出完整base-only初态，再通过自包含meta构图strict reload，确认全部状态不变。不重新生成初100；旧初态观测只可引用已核验标量，不将数学beta0关系冒充本控制新生成证明。

实际重载后reset42，再调用原BudgetQV构造器；48个A/B初始化digest和rank_plan必须与两原成对臂的ADAPTER_INITIAL精确相同。初始化器原CPU随机参数产生后置GPU属于继承实现；禁止CPU模型前向/反向/优化器/矩阵合并及activation offload。固定数值条件为FP16基座、外层CUDA FP16autocast、原SeqLinear FP32存储参数；不能误称适配器全部FP32计算。GPU数学禁TF32，折叠时GPU FP32且禁autocast后转FP16，与原pair同源算术。

相同1213 FIT、19阶段、87正LR更新，micro1累积batch16、原确定样本顺序、原stage-reset schedule/warmup、GradScaler和clip1。GPU AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、foreach/fused false、capturable true，每stage重建。复用原训练AST，不改变loss、dropout、epoch、数据或学习率。训练前第二次reset42；无DEV训练/教师/replay/新增校准或额外恢复步。原answer512+EOS、原生成cap、面积401408、AnyRes和context检索全部保持。GPU-only无视觉cache的transport与原pair一致。

逐QA保存1213个实际输入记录，task/phase、原raw/image/context、transport等字段须与原dynamic FIT_INPUTS逐值一致，阶段QA/更新/顺序SHA和训练AST绑定原PAIR_READY。记录87次实际clip时的96个有限GPU FP32梯度及17596416梯度字节；保存19个A/B状态/更新有效性，optimizer moments和step共35193216 GPU字节。训练期原冻结参数/所有buffer不变；A/B实际发生更新。未出现的质量不得由这些完整性检查推断。

## 完整末态与唯一评价

训练GPU0，与原dynamic使用同一物理卡；训练无DEV生成。GPU合并48个q/v后完整导出末态，不保适配器；独立eval进程在物理GPU1自包含strict reload，与原pair和PAS资源卡相同。先按原顺序执行每域hash最小ID共五个固定资源窗，各强制32新token，记录baseline/peak allocated/reserved、实际prepared/raw/transport/shape/图像绑定；再生成同100个原DEV完整回答。新共105生成，其中100用于唯一原token Answer F1评分，资源160 token不计F1。没有新NLL、TEST、初态生成或bootstrap。

评估完整state/buffer前后不变。所有训练和评估文件、实际身份/首次STARTED、源/环境/资产链、加载/折叠/完整图与五窗证明、退出均核验后，唯一消费100个新回答的原评分。原动态/静态质量只复制PAIR_VERIFIED既有标量，绝不重评分旧预测；旧prepared/输入证明可用于匹配。本控制与原pair不是同时运行的新随机种子重复，单seed且DEV已探索；报告这些界限，不称显著、稳定或非劣。

## 结果出口与论文边界

固定质量主量仍为五域等权宏Answer F1；同时报告各域/逐题得失、动态-QV及静态-QV差、实际完整成本和额外Down计算差。不能根据结果改任务平均/QA平均，正式AP19另报。若dynamic宏不高于匹配QV-only，不得称新增动态调制取得正净收益；若高，也仅支持本固定预算下的点估计净收益，不证明24beta语义机制、输入相关性普遍必要或已胜强PAS。static比较同样完整保留，不能隐藏更好的简单控制。

用户允许域间取舍，不以医学/金融或任一域低于旧wide淘汰；也不因总分比历史QV高便声称已达到“胜最强基线”。下一正式质量仍须与最强可比参照在合法同数据/评价协议下比较。此前PAS同卡五窗较候选高24.86–26.80%仅是原混合dtype实现的实际窗口资源证据；不转成同dtype最优部署、全分布峰或当前QV-only优势。折叠OSFT完整系数可回原base，本控制不凭其保存态重复表示宣称结构胜利。

无论正负全部保留，不改beta尺度、分组、LR、rank、QA、步数，不因本控制结果自动追加另一变体，也不以原pair没有逐域全胜继续淘汰其候选资格。此对照改变的是方法净效用的证据，不自动授权正式训练、TEST或另一个框架。

## 预算、执行与故障

只OLD真实空卡：train_qv首次STARTED起3600秒，eval_qv首次STARTED起600秒，顺序执行，共4200模型秒；包括加载/全部摘要/初末保存和重载、1213反向87更新及105生成。对应卡固定train物理0、eval物理1，原pair/PAS GPU身份须绑定；其他工作不受信号影响。controller隔离PGID，保存实际EXIT/ALL_EXIT，不restart或enable。源码、协议、全部小prior证明、原runtime/environment和数据在本控制READY.json绑定；未冻结或无明确启动记录不视为实际运行。

任何时钟不因观察/文档重置。故障/超时保原证据与实际费用，质量缺测不称科学阴性，不静默重启/续CP/加时或自动版本；必要纯实施偏离须透明独立裁决，不能按质量追配方。无新增CPU模型门/算子探针；原算术已使用，仅保必要实际前向/梯度/重载证据。原pair诊断、2426反向174更新及全部生成/审阅成本原样承接，不因本控制而消失或重计为本次新计算。
