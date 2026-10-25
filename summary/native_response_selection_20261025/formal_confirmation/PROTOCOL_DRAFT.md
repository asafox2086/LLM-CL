# 固定容量宽QV：单次正式终态确认草案

状态：科学方案草稿，尚非冻结协议、模型准入或启动记录。只写本文件；不修改原成对实验、QV控制、历史关闭结论或原正式表。任何真实运行前仍须将实际数据/来源/实现/身份/预算固定，不由本草稿自动执行。

## 已发生的选择与研究问题

人类最新要求总体原Answer F1略高于最强基线，同时最终完整模型和GPU内存明显更小，允许各域取舍。本次候选选择只用已完成、已探索的100DEV五域等权宏：匹配QV-only45.085962854378714、dynamic44.75760269784193、static44.68120094961681。QV控制VERIFIED SHA为1b0a8f8df3e4f1a4f3338a439398cd9c3197930002302dd8e8a72015fb9fa99c，原成对VERIFIED SHA为482d531edc4ab26455986b2f722204b9ac350368f3ef71bdd70da8de85beca76。

预定dynamic与QV择宏高、精确平局选简单QV的规则现在选出**bare wide-QV**。static原点更低但全部结果保留，不因其未入选抹除。QV比dynamic高0.3283601565pp、比static高0.4047619048pp；这不支持本固定调制包有正总体净收益，不能按此调整beta/LR/rank/步数或再选一个模块。45.086是FIT-only已探索DEV成绩，与正式PAS46.619976395不可直接作正式胜负比较。

尚值得回答的问题是：完整保留OneVision预训练功能、只用一个跨阶段共享固定容量宽QV学习状态，能否在原1%正式条件下达到稍高于强基线的最终总体回答质量，同时避免永久专家银行的完整权重和实际GPU内存成本？这是有直接设计作用的终态检验：若可行，可以保留固定容量架构；若不可行，不能再拿探索锚和显存优势声称目标已完成。普通LoRA/QV本身不是新发明，成功最多先形成配方/容量使用的经验发现，论文仍需诚实界定贡献，不能把被控制否定的响应模块重新写成创新。

## 唯一固定正式训练

从原完整OneVision base fresh开始，不续FIT控制或任何已训练CP；原冻结全部1313 TRAIN、19原生阶段、seed42、单epoch、micro1累积batch16，按各阶段实际ceil(QA/16)共93次正LR更新。固定48个q/v位置及原BudgetQV rank_plan，4399104个FP32 A/B坐标、scale4、dropout0.1、同初始化，无beta/额外Down/sign/c校准。原训练loss、answer512+EOS、确定样本顺序、stage-reset schedule/warmup、GradScaler/clip1保留；GPU AdamW lr1e-4、betas(.9,.999)、eps1e-8、wd0、foreach/fused false、capturable true与刚完成匹配控制一致。不追加rank/LR/epochs搜索，也不为正式TEST追加恢复步骤。

原完整视觉/projector/语言/词表、FP16 base、外层CUDA FP16autocast、FP32存储SeqLinear语义及SDPA条件明确记录；GPU-only无视觉cache transport与本控制相同。原BudgetQV初始化器的CPU随机参数产生/CPU checkpoint I/O不算CPU模型前向；模型前后向、梯度、optimizer、fold均GPU，禁CPU模型门/activation offload。合并48个q/v使用同GPU FP32禁autocast/TF32后转FP16。

全部1313包含先前探索划出的100DEV，这是原正式TRAIN的合法恢复，不是新的训练数据；这些已见过的行不再充当独立验证。19阶段训练只用TRAIN，0阶段DEV/TEST质量消费。保存逐阶段adapter、实际93更新/1313反向、真实正LR与有限梯度、训练输入和冻结基座/buffer证明。完整末态物理图预计893675552系数/1787351104权重字节，普通buffers及包字节实测另报。导出后独立自包含strict reload，最终完整图不保A/B、optimizer、教师或外部base权重。

## 唯一正式质量评价与聚合

只生成最终S19模型在原完整21454 TEST上的回答，共19个原任务终态cell。测试全集数量固定：natural5000、medical2094、geography10004、finance2233、traffic2123。无S0/域末/历史peak评估，不补74格，不报告未测FWT/BWT。原task instruction、上下文检索、面积401408/AnyRes、tokenizer、EOS/pad、任务max_new_tokens、batch1、do_sample=False、num_beams1、use_cache=True及原Answer F1 scorer不改。有限性检查如保留，只能验证原greedy score的有限性，不可修写logits或引入新decoding policy。

主指标为**五域宏**：先在每域完整QA上求原Answer F1平均，再对五域等权平均。伴随AP19为19任务等权终态F1，不能用AP19或全QA加权均值替换不利的主结果。每域/每任务/逐ID得失均保留，但不以各域都高为成功条件。

本草稿采用已核验PAS五域宏46.619976395和OSFT45.937438210作为当前正式参照；发布结论时须检查是否已有更强完整正式基线，不能永久把PAS称最强。仅主宏严格高于最强可比参照，才支持“总体略胜”的点估计陈述；精确平局不写提高。单seed不据此声称显著、稳定、等效或非劣。正式TEST只作这一次终态检验，不能用其结果改图/超参、退回dynamic、挑任务/答案长度或追加备选。旧LoRA_all正式44.894422831负结果原样保留，不能据其TEST差分调本次配方；本次只承接新匹配QV控制预先选定的不同位置分配配方。

## 资源、完整成本与公平参照

正式末态完整重载后的实际图，在原PAS测量同一OLD物理GPU1/同UUID、同五个raw/prepared/transport输入、相同顺序、每窗32新token、无视觉cache且窗口含准备与生成，记录baseline/peak allocated/reserved。正式只增加这五次资源生成，不计算F1；它们属于已经探索TRAIN中的固定资源题，不从TEST挑峰。可放在正式质量生成之前的独立fresh进程，避免将已扫全TEST后的allocator状态当fresh窗口。

PAS资源锚为strong_pas_resource_20261025/v1/VERIFIED.json SHA c987ec2ac68980944a1ebdc515fab08a792667f923d142f6f8489d3725753d08。现已测FIT QV资源不能替代正式末态实测，更不能把此前dynamic的24.86–26.80%直接复制为正式结果。完整权重系数/实际dtype字节、必要mask/普通buffers、包文件、常驻基线和窗口峰分列。

原PAS为混合dtype实现、保必要动态专家/路由及原CPU保护历史，测量并非同dtype最优部署；只能称相对此原完整实现的实际优势。CPU metadata字节不是RSS，不把它们作为被省掉的学习权重。OSFT常规fold可回原base完整系数，bareQV不能宣称比折叠OSFT结构更小。正式candidate和原PAS的训练硬件/优化器/分支实现差异公开，不把最终差异纯归给架构。五窗不代表全分布最坏峰，不据此报告未测延迟/功耗或算子创新。

所有训练、93更新、初/末完整保存重载、5资源请求和21454最终生成、评分与实际墙钟/分进程时间均计费；先前全部探索、调制诊断、两臂、QV控制及失败费用另列，不能因模块被淘汰而消失。训练并行或测试分片的重叠时间不能直接相加成实际墙钟。

## 一卡与四卡TEST：只改变工作分配

原正式runtime_utils.evaluate源码是逐QA独立generate、batch1、model.eval且greedy解码；同一题不读取上一题生成的KV，原输入检索无可学习跨题状态。因此按固定ID分片可以沿用原单题规则，不需要模型并行或新部署适配。单卡按原manifest/task/QA顺序最简单；历史LoRA_all完整RUN14412.548秒，其中training_before→after1126.508秒，余下含21454测试及保存/I/O等，不能当纯测试时长，也不能直接外推到OLD/折叠QV。当前QV105请求eval101.374960秒同样不代表完整TEST的分布或可靠ETA。一个合理预期是单卡需数小时量级；实际模型预算须在冻结前根据这些已有费用、任务cap和完整数量保守确定，不能启动后按结果或进度加时。

若选择四张真正空OLD卡，则在首次TEST生成前固定四个shard。建议对(task_id,id)的固定前缀SHA256取模4，每题只属于一个shard；每shard内部仍按原任务和原QA相对顺序，四个worker各自加载同一正式完整CP、eval模式、相同dtype/backend/单题policy，加载后统一seed42。记录各shard ID清单SHA、各任务数量、实际GPU UUID/环境/完整CP摘要以及实际token IDs；不得根据输出/质量/难题长短重新分片或迁移已发生请求。四卡只用于离线测试吞吐，不能把四份模型的总驻留说成端侧部署成本；GPU1单图五窗仍是独立资源主证据。

Greedy和关闭dropout使分片不依赖抽样RNG序列；但seed42不能保证不同GPU/CUDA kernel的逐token bitwise一致，尤其接近argmax边界时。不得宣称四卡与假想单卡一定逐值相同，也不因差异改用更有利卡。固定实际拓扑即本次测量条件；若需要声称跨卡token一致，必须有直接证据，当前方案不以此为额外研究目标、不另加全TEST重复。全部卡须同型号/软件/固定数值条件并公开与历史正式参照的差异。

只有四shard全部完整成功且ID并集恰等21454原TEST、无重复/遗漏、每题任务/参考/输入/生成cap一致，才合并19cell并唯一评分。分片域/任务均值按各自QA数量合并，绝不平均四个shard均值；canonical顺序合并后再运行原scorer，所有输出完成前不消费partial质量。若原evaluate边生成边打分实现被复用，必须将评分部分后移为纯后置原函数消费，不能照搬历史resume路径偷偷续跑/多次评分。

## 正负结果与论文边界

若正式五域宏略高于最强完整参照，且完整权重/实际同卡峰确有明显优势，可以报告一个固定容量、无永久专家银行的经验性质量—资源结果，并据此保留这一架构选择。它不证明专家普遍无用，不识别PAS增益的唯一机制，不把普通QV变成新算法；创新和可推广性仍需论文诚实讨论。

若总体未胜最强，即使GPU明显小也未达成人类当前联合目标；完整报告质量/资源Pareto点，不能用探索45.086、某域高或AP19更好替换主判据。若总体质量赢但正式窗口资源优势不足，亦不能称联合目标完成；如实呈现条件和权衡，不滚动救跑。执行故障只记缺测和真实费用，不当质量阴性。

本草稿不授权新模型/CPU门/TEST解析或运行。候选已由完整QV控制明确选定，不需要再新增一堆FIT强基线做阈值筛选；已有合法FIT OSFT40.783801仅是探索背景，正式run才回答公平强参照的关键缺口。冻结前只落实必要源/数据/聚合/身份/预算/退出与完整图合同，不把工程审阅和准入当科研成果。
