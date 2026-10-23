# 固定原生载体的起点回答基线

状态：必要性审阅草案；没有模型/QA运行或启动授权。它不是失败E/Q/R的v3、兼容修复、参数必要性结论或新框架。

## 尚未回答的问题与实际用途

已选SmolVLM-500M载体有固定官方资产，但没有原五域上的任何真实AnswerF1。r200旧先导说明仅有规模下降不能证明起点功能承接；上一实现的CPU失败也没有给出载体质量。以后若只有适应末态分数，就会混淆作者预训练能力与我们新增机制的收益。因此先固定一次无本地FIT适应的起点回答基线，作为未来任何同载体研究的共同原点，避免默认先添永久专家再解释必要性。

这不是按分数挑型号的资格门：不设通过阈值，不因高分宣称Down替代专家，不因低分否决87步适应，不自动训练/改提示/更换型号。每域及逐题得分只记录该固定集合的已观测起点表现与质量缺口；单题高分不认证稳定能力，不能据题目得分挑路由/模块、定位可删权重、保证跨输入能力或确定参数无用。后续方案若提出增益，须明确相对同一未适应原点的变化，不能把作者预训练规模与能力当成我们贡献。

所有结果均保存并成为后续固定参照；不采用正留负弃规则。高分时，新增分支必须解释相对已有能力的增量用途；低分时，先承认质量缺口及跨骨干/接口原因尚未分离，不能宣称小数据必然学不会。无论结果如何，本次不触发另一模型或SFT。100DEV已经探索；任何据它进行的后续设计均属探索，不称新留出或干净确认集。

## 单一固定条件

- 原生载体HuggingFaceTB/SmolVLM-500M-Instruct，revision a7da5b986cb59b408707209984f360a5f4ad7e47。只复用既核12项资产和原ASSET_RECEIPT/ASSET_TRANSPORT，不下载别的模型，不重复来源审计。官方预训练数据与原DEV重叠未知；“无本地FIT适应”不等于预训练未见评测。
- 零本地训练、零optimizer/backward、零adapter/新专家/learner，无NLL/短生成/校准或额外QA前向。只原冻结100DEV各一次完整生成，原SPLIT每域20题/89已知组，按原任务/ID顺序；原FIT不用于统计/优化。原TRAIN容器中取DEV行必须按冻结DEV ID过滤。
- 使用原问题/图像/上下文/全部参考/任务instruction和唯一原AnswerF1。输入沿已审原生runtime的面积401408、长context BM25及固定官方processor/chat template；不搜分辨率/提示/答案映射。不同骨干tokenizer/processor不称严格同数值输入或因果对照。
- 固定BF16基座/外层autocast，vision=eager、text=sdpa，与v2已越过的tiny构造路径一致。没有E包装、梯度托管或作者回调。实际原生from_pretrained必须拒绝missing_keys、unexpected_keys、mismatched_keys及error_msgs，不忽略权重键或借换后端继续。该forward-only检查不为失败梯度门提供通过证明，不新增CPU梯度门，也不放宽旧门。
- 原task max_new_tokens、greedy、num_beams1、use_cacheTrue，使用固定generation_config的assistant停止语义；实际解析并记录，歧义即停，不按输出调EOS。输入/原生成cap不能静默截断，超8192上下文即停。
- 模型全部requires_grad=False；开始/结束完整state digest一致。报告真实完整权重参数/权重buffers共享去重与dtype字节；派生索引buffers另列。作者小模型自身规模不是我们的减重成果。

## 成本、完整性与唯一评分

拟定STARTED→COMPLETE最多3600秒，包含加载、语义解析、100生成、完整权重/digest、保存与观察；这是一项有界基线投入，不是用户时限或1%定义。独立核验另计。必要性review可在源码冻结前否决或修订此预算，不能在结果后延长。

只NEW物理GPU1一个forward worker，最多占NEW三卡；CPU等待需实际资源可用，尊重他人，不用GPU0/OLD。不恢复旧任何service/worker。使用独立controller单调钟期限，绑定自身worker的PID/startticks/UID/隔离PGID，超时只终止并回收这个自身进程组，不触碰正式或他人作业。任一首次STARTED后失败/中断/超预算即关闭这项固定检查，不重做/续预测/延时/v2；只有STARTED/CUDA前资源变化75是未尝试wait。

新最小源码必须先独立审阅。复用已审NativeRuntime的input/generation和无副作用layout/digest工具；不调用旧make_method、训练、CPU门或E/Q/R控制入口，不改旧六源。小图构造通过的范围只作已知接口证据，真实BF16 forward须由本次完整运行本身证明。

保存全100逐题预测/全部原参考、准备输入/原图SHA、实际输出token/停止/费用、原生policy、配置/资产/源/环境绑定、同PID资源、开始/结束布局与完整state digest、STARTED/COMPLETE/实际退出。实际退出后才执行唯一verifier，检查原100ID精确一次、输入链、冻结参数不变、0训练及100生成、预算和文件SHA，再调用原score AST一次并写VERIFIED。只完整封存后读质量；不读partial预测，不重score/bootstrap。仅描述五域宏/逐域/逐题原分及完整权重，不算重复旧配对，不输出FWT/BWT/74格/Ours。

## 正负出口与贡献边界

这次产物是一个此前缺失、以后可复用的未适应起点；不是保质减重方法、机制有效性、专家冗余或新颖性证明。原OneVision合法FIT锚可引用既有标量，但跨骨干分差不归因我们的算法。用户未给定掉分容限，不宣布非劣/等效。

结果报告必须分别写作者已有能力、观察到的质量缺口、未解原因和对后续功能需求的约束，不把“高分保留/低分换型号”作为筛选行动。本次没有后续训练预算或自动新科学实验授权。旧native E/Q/R v1/v2、视觉、PAS、r200、ShortFixed及全部封存条件继续保持关闭，正式基线保全。
