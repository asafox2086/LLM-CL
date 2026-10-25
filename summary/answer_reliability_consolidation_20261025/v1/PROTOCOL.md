# 末层答案读出代理的回放巩固：固定三臂协议

本文件是科学原型的实施协议，不是READY、准入或启动证明。原METHOD_PROPOSAL与PROTOCOL_DRAFT原字节保留；本协议落实随后确定的GPU2训练、GPU1评价及实际计数。无模型结果时不能声称新方法有效。

## 问题与唯一干预

借用[I-LoRA](https://arxiv.org/abs/2402.18865)共享基座、working/slow适配器、历史回放及隐藏态巩固，检验在1%少步VLM开放问答中，按末层人工目标的答案读出选择约束样本，是否比无条件约束和同数随机约束更有用。Seq正BWT、域后正迁移和OSFT较强域末表现不能证明遗忘是总根因，也没有证明I-LoRA有害。

三臂均保完整OneVision和原q/v-r8，48模块、96个working FP32 A/B、540,672个训练坐标，alpha32/scale4/dropout0.1。slow是同一共享基座上的96个冻结FP32 A/B buffers，另540,672个系数/2,162,688 bytes；门新增训练参数为0。全部基座冻结，无额外稠密教师。

回放QA的working/slow隐藏态产生原坐标max目标`T_l=max(H_work,l,H_slow,l)`，保原embedding与全部24层输出的整包MSE。三臂统一以完整有效packed位置/通道及25项等权归约；micro1无padding。只对已过原final norm的`T_last`，在原答案预测位置经过同一冻结LM head，得到逐答案token平均NLL `q`；`p`复用working原答案CE。严格门`g=stop_gradient(1[q<p])`，相等关闭。无新norm、中间层head、温度、阈值或门率搜索。

有回放时每current QA的目标为`(current_CE+replay_CE)/2 + mask*MSE`，沿原minibatch大小累积；gamma=1，不按接受数重归一化。首task无回放，仅current CE。original的mask恒1；gate用本QA自然门；shuffle用下述reference mask。三臂都计算同样的额外target head及诊断，新增门不增加完整decoder，也不通过少算诊断制造训练速度优势。slow只需decoder隐藏态，不执行废弃答案head。

末层人工目标的较低规范答案NLL只是整包MSE的代理，不认证中间态、梯度方向、可实现轨迹或自由生成。teacher保同训练/dropout模式，no_grad不等于eval，复用同次forward；不添加额外确定性探针。

## 共同学习、记忆与顺序控制

原1213 FIT、19阶段、每QA目标最多512答案token后追加原EOS、原输入/检索/图像政策、seed42、hash训练顺序、stage scheduler、GPU AdamW、clip和GradScaler均继承已核plain-r8条件。原训练AST不改：87次真实optimizer更新，逐current QA的组合loss共1213次backward调用；96份有限FP32梯度共2,162,688 bytes，GPU AdamW完整state共4,325,760 bytes。每阶段optimizer仍reset；不增加更新、rank或学习率分支。

采用paper式past-only记忆：每个原task训练结束后，将其实际训练顺序的FIT纳入固定seed42 reservoir500。插入替换RNG、回放抽样RNG与模型/dropout RNG分离；S2起每current QA有放回抽1个已完成task QA，S1没有。每臂实际1213 current QA+886 replay QA；插入/替换/抽样和原task边界全记账。缓存是冻结FIT资产的QA引用，不缓存teacher logits/hidden；账本编码bytes不是RAM/RSS上限。共同移植不同于官方代码的当前QA先插再抽，不能称逐字节复现官方运行，不能把此差异归新门。

每阶段开始slow按明确A/B对应复制working；每次成功optimizer step后仅一次`slow←0.25*slow+0.75*working`，不按micro1更新。修正官方源的EMA同名匹配/阶段同步函数名问题是所有臂共同实现条件，不把源错误当原法阴性。评价唯一固定working/default，禁止按结果选择slow。

严格顺序为`train_gate → train_original → train_shuffle → eval_gate → eval_original → eval_shuffle`。gate训练完成封存每step current/replay IDs、自然gate向量及源SHA，不消费生成质量。shuffle仅在自己的当前step读取对应封存向量，以`42:reference-mask:stage:step`确定的独立排列重演，同步验证QA IDs和reference SHA。每步接受数精确等于gate reference，自己的自然门仍另记；不声称自己门率匹配。它是reference轨迹反事实控制，不是独立在线算法，无未来阶段模型请求；micro1即时backward，不保双图或多做探针。三臂current的1213条原FIT输入记录及raw/transport摘要逐值绑定独立Naive，初始化48组A/B摘要也精确绑定。训练记录未逐题hash视觉feature tensor，不能称训练prepared tensors逐值核验；DEV/资源readout另有完整prepared摘要。

独立Naive-r8已另行运行，不重复训练。它训练GPU0，新三臂均同物理OLD GPU2串行，主归因比较三臂；对Naive的比较公开训练GPU身份不同，不能声称历史执行逐值相同。

## 唯一评价与成本

三train各首次STARTED起5400秒；三eval各600秒；总phase上限18,000模型秒，controller外层18,300秒，包含加载、回放/统计/额外head、保存、重载、实际生成与清理，不重置或延长。模型数值计算在GPU，CPU仅源/数据元数据、检查点I/O与后置评分；没有CPU模型门、无TEST、无未来FIT训练。

每臂先保存并实际完整重载共同base初态，0初态/阶段生成；训练结束GPU FP32、禁TF32/autocast的原q/v fold转FP16，导出working-only自包含完整包。最终图必须严格重载，设计893,675,552个FP16系数/1,787,351,104权重bytes、普通buffers9,032 bytes，不含A/B、EMA、回放、门或额外head。最终相对同样fold的plain LoRA/I-LoRA没有新减重，不借训练参数少宣称推理优势。

所有训练完整后，三臂各在同物理GPU1 fresh重载，先同五题固定32-token资源窗口，再各100个已探索DEV自由生成；每臂105次、合计315次生成，0初态/阶段质量选择。六phase完整退出、身份/源/输入/预算/包/EMA移除和全部文件链通过后，唯一后置评分原300条，五域宏为主、逐域/逐题全报告；不重评Naive旧预测，其核验标量仅复制。

每臂另报working current1213+replay886次decoder、slow886次decoder、额外target-head886次；组合loss backward调用1213次不等于仅1213个有梯度QA。三臂总3639 current、2658 replay、8955次decoder请求、261次optimizer更新、2658次额外target-head。训练峰包含EMA、回放和隐藏态，不混称推理峰；五个推理窗口不是全分布上限。完整包、权重、普通buffer、运行峰和时间分别报告。

## 事前解释出口

gate相对original的宏点差检验准入包净效用；reference与shuffle实际mask对应至少发生一次改变、且gate还高于同数shuffle，才支持样本对应选择的额外作用。后置记录mask Hamming及混合/退化步骤；若shuffle从未改动对应关系，即使全局门率非0/1也不能识别选择作用。只胜original不胜shuffle，不能排除弱正则效应；Naive胜出则不宣称新门带来总体净益。Naive对比还包含ER/EMA整包差异，不能全归门；不新增ER-only第五臂来追结论。

接受数全0/全1须如实报告退化，不当机制阳性。逐域允许有取舍，不要求各域最强。只有终态100不能区分即时习得与后续保持，不能称解决遗忘或强化视觉语义。单seed/已探索小DEV不支持显著、等效、正式胜最强或总目标完成。所有正负与成本保存，模型故障记实施缺测而非质量阴性；本固定点不按DEV滚动搜索LR/rank/gamma/memory/阈值，不隐瞒故障或擅自重启。是否进入正式确认由完整证据另行决定，本协议不授权启动。
