# 原生FFN响应选择：固定成对学习协议

状态：本文件定义待源码审阅与READY绑定的固定科学协议；本文件本身不是激活记录。只定义本功能比较，不恢复旧native E/Q/R、旧24beta单臂、spatial读出故障或其它关闭分支。原生指完整OneVision既有FFN响应，不是改用SmolVLM。

## 问题与已获得的前提

同完整OneVision与连续wide-q/v学习条件下，原生FFN响应的输入相关重加权，是否优于相同标量容量的静态重加权，并保留合法wide探索锚的五域功能？最终目标是避免强方法永久新增专家，不要求破坏或删除预训练W0。

已封存32个S1 FIT组诊断，VERIFIED bfcba0a0ad4a15fe37e91b0a5f3967bd47d83243d62de750aef8edb2a94bb2f6，COMPLETE 1c90c048b9c775df8571b1f2923ce3eaf5c15a978d93fa9ad028ed8611dfd30e。24层门和加权响应均非退化，c_l为该COMPLETE.layers对应RMS，固定转GPU FP32；不重新校准、不改分组／epsilon／幅度。这不是有用梯度或Answer F1的证据。诊断31.973244秒及32NLL／768抽样Down是已发生共享费用，另列承接，不能消失或按新模型费用重计。

## 唯一两臂，保持相同显式计算图

共同原OneVision完整视觉、projector、Qwen语言、原tokenizer／面积401408／AnyRes／上下文检索／原answer512+EOS／生成cap。无额外teacher、回放、检索库或未来域校准。G/U/D权重冻结；q/v变化会改变输入，不能称响应全程固定。

每层phi=SiLU(Gx)*Ux，原h=Dphi保持原计算路径，额外h_diff=D(sign*phi)，sign按4864个通道索引奇偶为+1/−1。动态d=(mean(G_even²)−mean(G_odd²))/(mean(G_even²)+mean(G_odd²)+1e−8)；静态d=c_l>0。两臂输出均为(h.float()+beta.float()*d.float()*h_diff.float()).to(h.dtype)，原h、Gx、phi及h_diff按共同原FP16路径，门统计／标量乘加显式GPU FP32。固定具体括号顺序，不隐式改变输出dtype。正常链式求导穿过d与h_diff，不单独detach动态门。

两臂每层beta为一个FP32零起点标量，共24；beta=0不跳过分支，以保首步beta梯度。初态只对有限值保证原h路径，0乘非有限不作保真依据。beta无约束，不能整体称凸混合或概率路由。

最终两臂都保相同显式signed-Down分支，**不把static折入稠密D**。static仅在GPU FP32、禁autocast／TF32时计算gamma=beta*c，最终每层保存一个gamma，d固定1；不保第二份c权重、不改D字节／bias。dynamic最终每层保存beta并计算原d。常数编译前后实际乘法系数应逐值绑定，不能重复乘c。此转换只合成一个FP32系数，避免D重参数化至FP16造成额外舍入对照。两臂共同q/v用相同GPU FP32合并至FP16。

## 共同学习预算与初末图

复用合法BudgetQV原rank_plan／初始化，q/v4399104个FP32坐标，加24beta后实际训练坐标4399128（预计96个A/B张量加24标量；以真实枚举确认）。原1213FIT／19阶段／87正LR更新、seed42、确定样本顺序、micro1累积batch16、原stage-reset cosine/warmup与clip1；GPU AdamW betas(.9,.999)、eps1e−8、wd0。base FP16／外层CUDA FP16autocast／原SeqLinear FP32参数语义公开；不加beta专用LR、缩放或额外恢复步。所有模型、梯度、优化器和合并GPU，CPU只数据与I/O。

每臂fresh相同原base，安装响应模块后先导出完整初态并自包含strict reload，再初100原DEV；不消费初分决定训练。随后原87更新，保存19阶段adapter/beta状态、真实梯度／更新有效性和输入证明。折叠q/v及static常数编译后导出完整末态，独立fresh重载；先五个固定hash最小ID、每域一个恰32新token资源窗，再末100原DEV。最终资源两臂同一实际卡串行；训练若并行，公开实际卡／dtype／backend，保持对应源、初始化与预算一致。

本两臂共2426真实FIT反向／174更新、初末400新回答评分及10资源生成，共410生成；额外h_diff Down为训练／初末／资源实际前向的一部分，不能称零代价。两臂和最终资源完成前不读取部分质量；全部退出后唯一原Answer F1评分，无DEV NLL、TEST或bootstrap。

## 是否需要第三个QV-only

识别本固定动态与静态处理差异不需要第三臂，本协议不自动追加QV-only。旧合法wide43.791987是已探索历史锚，不是当前所有数值／硬件条件匹配的因果控制。若论文要声称任一重加权模块本身优于普通wide-QV，必须另有匹配的QV-only对照；当前不得从历史差分替代这一结论，也不能把组间差分称全部性能提升的原因。

## 完整成本与资源记录

预期最终两臂均为893675552个FP16基座系数加24个FP32标量，即893675576系数／1787351200权重字节；这是设计数，须真实布局和完整包确认，普通buffers及配置单列。static训练／初态仍需要24个数据导出的c常量，按真实训练／初态持久内容计，不把它们藏成免费普通权重；最终gamma编译后方可移除。sign若缓存为可重建张量，其实际buffer／峰值计费，不另存dense Dsign或复制D。

报告各完整包字节、全部必需权重／buffers、常驻基线和五窗peak allocated/reserved、额外Down以及所有加载／训练／保存／校准成本。资源优势最终应相对公平完整强参照：PAS须去掉可合法折叠的表示冗余但保全部必需动态专家；OSFT常规fold可回同稠密893675552，不得拿其USV保存态夸大本图优势。本图相对wide或折叠OSFT略增24标量，不能声称系数更小；本次两臂资源也不能自行证明胜强PAS。若需要fresh参照资源，须另列预算／请求并在冻结前明确，不由本协议自动追加模型。

固定每臂train3600秒（含初态加载／重载／100生成／87更新／保存合并）、每臂final-eval600秒（含freshload／五窗／100生成）；新两臂合计上限8400模型秒，共享历史诊断另列。四阶段合计8400模型秒；训练只OLD两张真正空卡并行，两个eval在同一实际卡串行，外层unit上限5100墙钟秒（覆盖3600+600+600及准入前后开销），不enable、Restart=no。每个实际STARTED只开始一次，任何时钟不因文档/观察重置。故障／超时保原证据与费用，质量缺测不作科学阴性，不静默新目录重跑。

## 结果出口

逐域及逐题报告两臂初末与原wide历史目标：自然61.5、医学44.5、遥感45、金融30、交通37.959937043。宏均不掩盖单域下降。主功能比较是最终dynamic与static点差；单seed、20题／域、已探索DEV不称显著、等效或非劣。

任臂五域均达到旧wide目标，只保留作后续公平强参照／框架评审的候选，不称已胜正式PAS。只有dynamic达到则保dynamic；只有static达到则保static、不为偏好动态追参。均达到则末态宏高者优先，精确同分static，但不能宣称dynamic必要；实际资源权衡全部保留。均未达到则关闭此奇偶／相对能量／24beta／wide预算配方；dynamic较高也只算固定功能局部用途。不得因未过而滚动改变分组、LR、rank、beta尺度或更新数。

实际beta与梯度是否变化、修正幅度和开销应记录，用于判断实际干预发生情况而非寻找新门槛。正的门RMS不保证87步功能足够；若修正弱，不回头按结果提高幅度。原生响应选择、分组和缩放已有先例，当前研究贡献仍须来自可避免永久结构的完整质量／成本证据，不能靠借用组件起名或本协议完成宣告。


## 实施与消费约束

新独立scratch和PAIR_READY绑定八源（response/paired_common/paired_readouts/paired_adaptation/paired_deployment/paired_worker/paired_controller/paired_verify）、本协议、必要性与源码审阅。旧诊断源码及READY不改，校准只复制完整COMPLETE/VERIFIED原字节；历史wide来源四小proof/prior_parent_predictions只为绑定复制，不重生成或评分。asset_view为既有合法资产精确硬链接镜像，仅添加原BudgetQV core源，所有运行源码/环境版本、manifest/SPLIT/图像均绑定。

训练器保留原训练AST，唯一模型处理是新response参数化及共同wideQV；GPU AdamW阶段reset、GradScaler和原逐阶段LR保持。原BudgetQV A/B初值由原CPU随机初始化器产生后位于GPU，属于继承初始化；没有CPU前向/反向/优化器/激活offload/矩阵合并或CPU模型门。相对旧历史wide的HostAdam/CPU fold条件有差异，历史点不是单因果控制。

完整导出保底层共享storage、全部注册/非持久buffers、混合参数dtype、共享词表、原generation_config公开参数和框架标准加载哨兵语义。自包含meta重建无需外部原始权重；不得用仅成功保存替代严格重载。原视觉参数和普通基座权重不变证明覆盖训练期，response beta和q/v则须真实更新；beta若未改变或过弱只能如实解释，不能看结果调LR/尺度救跑。

只有四阶段各COMPLETE、实际exit0且PID/startticks旧身份不再存活、unit终态成功、样本/权重/所有梯度与optimizer/训练有效性/完整文件链均验证，才能唯一生成VERIFY_STARTED并消费400次原token Answer F1；不读取partial质量，不重评分、bootstrap或换聚合追方向。模型工作失败/超时不自动重启/版本/续CP/增加数据步数；真实故障保原字节与费用，缺测不是质量阴性。

本轮没有强PAS的fresh资源或质量请求，不能把本pair的门当成最终论文联合门。阶段原始费用、完整CP和每域初末分数全部保留，正式8/16主表、FOREVER和已测存储列无新正式proof不改。只有后续真实质量与完整部署成本共同证据才可能推进正式确认。
