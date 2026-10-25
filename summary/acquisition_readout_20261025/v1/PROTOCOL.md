# 医学与遥感：已有快照的刚习得回答诊断

本协议只落实一个解释问题，不是READY、模型准入或启动证明。不得写入旧科学源或原RUN，不新增基线训练。其科学价值是决定简洁小模型学习机制需要改善习得还是后续变化；本诊断自身不是新框架或结构压缩成果。

## 唯一动机与固定范围

本次已完整核验的共同I-LoRA移植original，相对匹配GPU朴素r8 Naive，遥感终态Answer F1由30到65（+35pp），医学由48.357143到38.357143（−10pp）。答案门未改变这两域的终态F1；不能把共同收益/损失归给门。仅终态不能判断这些差异在当前task刚学完已存在，还是随后跨域更新中形成。

只比较original与Naive两方法，medical/geography两个域，各使用原SPLIT已探索DEV的全部20个原ID，不挑题。对每个QA，按原manifest/SPLIT映射到它所属的原task及该task训练结束时的快照；输出`acquired_stage`必须等于该task，不能统一替换成整个域的最后阶段。两方法×两域×20题，恰80次新自由生成、一次80条原Answer F1评分。没有初态、其它域、额外终态、NLL、TEST或新FIT模型请求，没有训练、恢复或未来FIT优化。

这是从已观察终态结果提出的回顾性诊断，不是预先未见的测试集确认；已探索DEV不用于宣布正式最强、显著性或总目标完成。原five-domain宏主目标不改为此两域指标。

## 只读来源与完整状态重构

沿用两条已封来源：Naive的`summary/plain_r8_gpu_control_20261025/v1/VERIFIED.json`（SHA `3b172989a57817cb99dbc4f70649efa0193ca5ee94ba6f5d80c1b274f5e3bc66`），original的`summary/answer_reliability_consolidation_20261025/v1/VERIFIED.json`（SHA `84ed17f99ed21a01ffc6f7a5ba479ed3670c4ba34b54a0e2037a324a92876aa8`）及其CONTINUATION/POST_MODEL_SCHEMA完整证明。READY冻结旧source目录、各方法原RUN、train_COMPLETE/VERIFIED、原initial完整包、所需stage快照/证明、数据/环境/运行源码与新诊断源码。此处不把旧RUN路径称为永久实时状态。

两方法save_stage均保存了全部96个working FP32 A/B、540,672个系数，原基座和普通buffers逐stage不变。ARC EMA/回放影响学习轨迹但不参与默认working推理；本诊断不需要继续学习的EMA/optimizer/RNG，不恢复训练。

每次重构从该方法已核的**未适应initial完整包**独立strict reload，构建原48个q/v-r8位置（alpha32/scale4/dropout0.1），严格核adapter全部96个Bridge参数名、shape、FP32 dtype、总数、源文件SHA及stage adapter_digest，再加载该stage快照。不得把旧字典当通用PEFT包unchecked加载，不使用最终fold包为base，不在前一stage已fold模型上累计叠加下一stage。所有原视觉、连接器、词表、位置/旋转缓存与功能配置保持；不重建/修改原数值配方。

### 先完成两条S19一致性检查

任何新回答生成之前，必须分别用两方法S19快照＋原initial包，按原GPU FP32、禁TF32和autocast后转FP16的fold构造完整working-only图。重构的完整语义receipt必须与各自已封train_COMPLETE.checkpoint精确相同：完整权重及全buffer digest、layout/普通buffer清单、共享别名、功能/generation配置及attention类等；允许物理device不同。原checkpoint路径、容器文件SHA/bytes是既有产物的来源证明，不要求新命名容器具有相同序列化字节。不能只比较系数数量、局部A/B或少数输出。

两条检查都通过后才可进入任何新生成，S19检查自身0生成。任一不匹配即本诊断不能接受，不以F1相近替代一致性、不补训练。通过只证明重构契约与原已封末态口径一致，不代表其它stage重构已自动通过。

## 实际生成、输入及状态证明

固定方法顺序original、Naive；各方法按所需原stage升序，每stage只读所属的两个域QA，保持原manifest/DEV顺序。每stage独立fresh initial→完整96 A/B→原GPU fold；保存源stage SHA/adapter_digest、完整重构receipt和评价前后state/buffer不变证明。评价关闭dropout、固定working，不加入EMA/门/额外head。原输入模板、检索、图像及面积401408、原答案参考、原task生成cap、EOS/停止政策和greedy解码均继承。

每个新预测显式记录`method,id,task,domain,acquired_stage,phase=ACQUIRED_DEV`，以及实际raw/image/context、transport/prepared digest、形状、参考答案、generation_cap、output_tokens和prediction。IDs恰覆盖两方法同40条QA，每域20条；不能额外生成S19或末态回答。新worker只产生新回答与完整性记录，不读取旧终态预测或F1来选择输出。

所有模型加载、embedding/视觉/decoder/head计算与fold在GPU；CPU只用于初始化/数据与元数据、序列化和后置原评分，不运行CPU神经网络门、恢复训练或精度/算子探针。原RUN/原完整证据只读，新文件只写新RUN。重构图设计仍893,675,552系数／1,787,351,104权重bytes及9,032普通buffer bytes，不把0训练称为0GPU/时间成本。

## 唯一核验与解释量

单worker完整成功退出、源/数据/身份/预算、两S19一致性、逐stage快照链、80输入/输出和状态不变全部通过后，原scorer第一次且仅一次评分80个新回答。先写VERIFY_STARTED，再产生VERIFIED；无重评分或bootstrap。旧final_F1逐ID直接复制上述两个父VERIFIED已核标量，不解析旧prediction重新评分，不生成新终态回答。

`quality[original/naive]`保存每域20QA的acquired_F1、final_F1、final_minus_acquired及逐ID；可另列两域宏描述量。这里acquired是按各QA所属task多个时点聚合的描述量，**不是单一域末模型**。`comparisons`保存original−Naive的acquired差、final差及变化差，逐ID正负平全保留，核恒等式：`final差 = acquired差 + 后续变化差`。

刚习得已有优势/损失，说明差异先在学习阶段形成，后续设计应优先考虑当前有效习得；刚习得相近而终态分离，优先考虑后续迁移/保持。混合按域和逐ID描述，不追求预设方向，不设置每域胜出或非劣门。它只定位两条轨迹差异出现的时期，不单独识别ER、EMA、回放曝光或某一次更新的因果；Naive原训练GPU0、original原训练GPU2也应公开。

统一fold使早期与既有终态在相同部署口径比较，不能称重现当时unfolded训练图的逐位回答；不把舍入差异自动当遗忘。选择QA由当前完整结果决定，不能再据新读出挑域、加题、改stage、追rank/LR/gamma或阈值。诊断正负均记录，不自动授权新方法或正式TRAIN/TEST。

## 资源与首次故障边界

只使用OLD上一张真正空闲卡，单worker承载全部模型工作，物理卡及UUID由READY/STARTED绑定。首次实际模型STARTED起总1800秒，包含所有加载、重构、必要摘要/哈希、两S19检查、80生成、保存及清理；外层监督2100秒，不能因换阶段、文档或观察重置。所有stage共用这一总预算，不是每stage1800秒。具体资源准入/排队是外部调度，不把未满足空卡条件称为科研阴性，不向他人/基线进程发信号。

首次实际模型故障/中断/超时保留原证据与费用并停止本固定诊断，不重训、补恢复、扩大时长/QA/方法或自动新版本。缺测是缺测，不是学习/保持方向的结果。既有基线仍按原调度运行，本诊断不重复Naive训练、不改旧源/READY/结果，不新增CPU模型门。
