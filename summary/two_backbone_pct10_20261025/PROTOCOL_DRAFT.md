# 双骨干10%训练／完整测试：协议草案

2026-10-25人类已明确授权新增第二主表：训练10%，使用完整测试集，其余沿用1%正式设置，利用空闲服务器卡。此文仅为只读本地审查后的准备稿，不是READY、运行指令、已启动或已完成证明。原1%代码、清单、服务及表格均不修改；不实施optimizer-carry试点。

## 当前准备情况与授权边界

- 原V2 `execution_policy.json`仅有`full`和`pct01`；`run.py --condition`也仅接受这两项。FOREVER v3仅接受`pct01`。现有两套policy均无`pct10`清单、分配卡或ownership。
- 本地检索未发现可追溯的`pct10` manifest、frozen_ids、split_lock或完整正式结果。原生五域正式资产主要在服务器；本地并未声称已核服务器数据、空卡或任何实时作业。服务器路径、quota、scratch容量及空卡由root另行核实。
- 新指示授权扩展10%条件，覆盖旧“当前只派发1%／不自动full”的比例范围限制；不授权启动100% full，不取消1%既有队列，不恢复关闭的科研分支。原先停止旧聊天的交接文字不覆盖本接手会话的明确新授权。
- 本草案与[空白主表](MAIN_TABLE.md)不填训练量估计、结果、参数或资源估计，不复制1%结果。精确TRAIN总量、各stage更新数和数据SHA待10%清单冻结后填入。

## 方法、骨干与共同设置

两骨干：`Qwen2-VL-2B-Instruct`、`LLaVA-OneVision-Qwen2-0.5B-OV`。每骨干8种比较方法，共16组合：SeqFT-LoRA、FOREVER多模态适配、CL-MoE、SMoLoRA、PASs-MoE、PaRSP、OSFT、SpaRTA。ER-LoRA、FV-guided Training、ALER等已移出方法不借本次扩展恢复；Ours只保留空行，尚无正式新方法。

继承seed42、单轮当前任务训练、有效batch16／microbatch1、QA均值归约、原梯度累积和clip1、原初始化、作者方法超参数、gradient checkpointing及各方法原有额外机制。冻结相同视觉／连接器／语言基座范围，不更换适配目标或扩大rank；每任务仍重建optimizer／scheduler／GradScaler，不借10%运行测试carry或新优化器。

公共图像政策为RGB、等比例BICUBIC、最大401408像素，之后使用各骨干原生processor；原图不覆盖，原模板／检索／EOS／greedy及各task生成cap不变。正式基线保留原128MiB CPU LRU视觉缓存，不借用探索实验的GPU-only transport替换。两骨干原生展开不同，不能要求prepared张量跨骨干相等。

环境沿原正式框架固定来源（Torch2.6.0+cu118、Transformers4.46.3、Tokenizers0.20.3、Cloudpickle3.1.2及实际依赖SHA）；实际启动时重新记录环境与源码，不把旧报告文字当本次环境已验证。

## 数据与冻结清单

五来源仍为TextVQA／SLAKE／RSVQA-LR／FinMME／CODA-LM，AQUA不恢复。沿用原来源TRAIN／TEST隔离，不以TEST答案或效果选择10%训练样本。原唯一无答案`SLAKE:1622`的已授权排除继续生效：有效full TRAIN131018，原始full131019；不改原文件。10%应从这一合法来源生成独立派生清单，完整TEST保持原21454条及全部19个task文件的原ID、参考答案、文件SHA与顺序。

抽样沿原seed42、原题型分层／最大余数分配／哈希选ID规则，仅将比例改为0.10；需要明确并冻结实际精确TRAIN数量，不把1313简单乘10。核验`pct01 ⊆ pct10 ⊆ 合法full TRAIN`及TRAIN／TEST不交叉；若原抽样实现不能自然保持嵌套，先明确确定性规则与实际ID关系，不能各方法单独抽样或用TEST选择。19个task必须都有合法TRAIN，不静默删QA、改答案、合并小task或平衡曝光。

冻结新的manifest、frozen_ids、split_lock，记录来源full／pct01 SHA、19个TRAIN文件及共同TEST SHA、分域／分task计数。两骨干和两主机使用同一冻结版本；每个方法fresh原基座开始，禁止从1%或full已学checkpoint继续。任何preflight生成的权重不得用于正式训练。

固定19阶段顺序不变：T1 TextVQA；T2–11 SLAKE（Modality／Position／Organ／Size／Abnormality／Quantity／Plane／Shape／Color／KG）；T12–15 RSVQA-LR（rural_urban／presence／count／comp）；T16 FinMME；T17–19 CODA-LM（general／regional／suggestion）。这些是原研究冻结课程，不宣称来源论文规定了该顺序。

## 评价：完整测试与74个必要格

原唯一Answer F1沿用`token_f1`、0–100，不做选项映射。stage0测全部19个task；stage1–18只测当前及下一task；stage19测全部19个task，共74格、沿原TEST共80316条逐题生成／评分，其中最终21454条。未测格为null，不补零或伪造完整矩阵；完整TEST每次按原任务分组读取，不改成10% TEST。

第二主表保留与1%相同的最终五域QA均值、五域等权宏F1、19任务等权AP、FWT及BWT展示口径，清楚标记不同权重。最新人类科研重点为**新习得**，BWT只为继承原正式覆盖的描述量，不用作优化／选方法目标，不另研究遗忘。

另列新习得：取各task训练结束时的对角格`R[t,t]`；新习得AP19为19个对角格等权均值；新习得五域宏为各域对角格按该task TEST QA数加权后，五域等权。可从既有74格直接得到，无需新增测试请求。必须标明这些读数来自不同task时点，不把它们写成stage19最终分数或单一域末模型。可由同样74格报告对应task训练前→后的变化，不以低最终分数单独断言“没学会”。

只在完整19阶段、74格、逐ID／引用／原评分器及原最终checkpoint来源核验成功后填表。10%与1%分表，不通过TEST追rank／学习率／方法变体，不依据未完结果选方法或更换主要指标。

## 方法资源与来源保留

下表是现有实现的数值／资源路径，**不是10%实测内存或吞吐估计**。继承方法差异，不能全部替换成研究用GPUAdam或统一FP16。

| 方法 | 原学习率／主要更新 | 原资源路径及不可静默改动项 |
| --- | --- | --- |
| SeqFT-LoRA | 1e-4；语言q/v r8、alpha32、dropout0.1 | FP16基座、FP32 LoRA；原torch AdamW |
| FOREVER多模态适配 | 当前任务沿原1e-4；作者校准／回放／正则另计 | 独立v3及冻结analysis helpers；额外历史TRAIN曝光、模型时间、快照、回放成本原样记录，不能把一轮当前任务称为相同总计算 |
| CL-MoE | 原2e-4；作者专家／双动量／路由 | FP16基座、原FP32适应及作者状态；保留activation CPU offload、任务锚／模型外状态 |
| SMoLoRA | 原1e-4及作者原参数组／衰减 | 原native BF16基座／autocast、FP32适应；需要Ampere或更新且原生支持BF16的GPU；2080Ti不得强行FP16替代；保留activation offload |
| PASs-MoE | 原1e-4；作者2专家r128等设置 | FP16基座、原FP32专家／路由例外、HostAdamW及activation offload、掩码／参考历史成本保留 |
| PaRSP | 原3e-5及作者正则／共享适应 | HostAdamW、activation offload、保护统计保留 |
| OSFT | 原1e-5、原25%低子空间／阶段refactor | HostAdamW、activation offload、冻结／可学SVD状态全计，不换成LoRA或研究关闭变体 |
| SpaRTA | 原1e-4及作者双秩／渐进router | 保留原恢复触发、正则与实际跨stage update计数；10%步数更多可能触发原机制，不人为关掉或按1%步数封顶 |

所有组合分别记录完整参数量（系数总数及dtype构成）、去重权重／必需持久状态bytes、相对原基座增量、最终checkpoint容器bytes。不能把trainable参数、文件大小、逻辑权重或缓存合并为同一个“参数量”。训练资源分别记录CPU RSS／PSS／USS、CUDA allocated／reserved和NVML、阶段前后与采样峰；推理五窗／首词延迟／速度没有独立实测就留空，不把训练峰写成推理峰。原1%资源数值仅可作为注明条件的历史链接，不能填入10%行。

## checkpoint与完整恢复

继承现有正式保存政策：SeqFT-LoRA保留全部19阶段完整checkpoint；其余V2六方法正式训练保留最终S19。保存完整method graph、冻结基座／视觉／连接器、专家／router、必需状态、processor、RNG和来源配置，不能仅保存requires_grad张量。optimizer／scheduler／scaler在每stage重建，所以该正式边界checkpoint不声称包含持续optimizer状态。

FOREVER保留原v3 rolling transaction：每stage写完整图与必要method state，commit指针及相关阶段评价／事件持久化后才移除更旧权重；最终完整S19必须保留。不可按普通“final only”使它失去合法边界恢复。SeqFT及FOREVER只允许匹配本10% source／manifest的耐久边界恢复；其它无中间完整CP的方法不得拼接失败阶段或1%预测。

私有QA、图像、checkpoint与密钥不提交Git。实际存储需求由root按主机quota及scratch核算，不将1%文件大小乘10作为参数估计；完整19checkpoint／rolling临时文件／保护历史都要计入。必要完整重载证据沿原实现复用已验证路径，不新增泛化部署工程。

## 入口、调度与启动模板

当前旧入口**不能直接运行`--condition pct10`**。独立快照lane需在新目录保留原科学实现，仅扩条件解析／manifest锁／policy／报告范围，明确config.train_fraction=0.10。原V2 run目前把所有非full写成0.01；config.primary_score还遗留rougeL，而权威正式主指标为token_f1，新metadata必须显式正确，不更改原评分器。FOREVER同样有condition及fraction硬编码。旧V2／FOREVER源和1%活跃队列不得热改。

以下是完成独立快照与10%冻结之后的命令结构，不是现阶段可执行命令，路径和GPU均待root核实：

```bash
CUDA_VISIBLE_DEVICES=<ownerless_gpu> LLMCL_HOST=<verified_host> \
  <bound_python> -u <independent_pct10_v2>/run.py \
  --method <SeqFT_LoRA|CL_MoE|SMoLoRA|PASs_MoE|PaRSP|OSFT|SpaRTA> \
  --condition pct10 --run-dir <independent_pct10_v2>/runs/<backbone>/pct10/<method>/<run_id>

CUDA_VISIBLE_DEVICES=<ownerless_gpu> LLMCL_HOST=<verified_host> \
  <bound_python> -u <independent_pct10_forever>/run.py \
  --method FOREVER --condition pct10 \
  --run-dir <independent_pct10_forever>/runs/<backbone>/pct10/FOREVER/<run_id>
```

正式进程从基座fresh开始；必要preflight／独立checkpoint_probe使用原入口同结构、独立run，预算与额外请求计费，不能将其训练过的图接作正式起点。单卡一GPU worker、隔离PGID／精确UID-startticks身份；来源／manifest／environment SHA与实际GPU UUID全记录。

沿现有资源guard：ownerless、util<10%、足够空显存（原OLD最小8500MiB、NEW最小16000MiB），不杀他人或现有基线。NEW继续至少保留一张不由本作业分配的卡，原GPU0 reserved约束保留；具体空闲快照不能写成永久live。OLD可并行用空卡跑其已支持组合；SMoLoRA等资源不符组合留合适硬件排队，不能改算法／精度／测试集迁就卡型。NEW现有1% CL／SMo等身份保全，10%调度与1%作业按condition区分，禁止同组合重复派发。

原协调器ownership和本地capacity wrapper目前仅针对pct01，不能只给旧服务增加条件后声称10%已启动。启动前冻结新lane来源／10%数据／主机policy与队列；不自动full、不重做sealed1%、不占旧research分支。耗时按实际硬件分别报告，不因卡型差异宣称算法加速。

## 权威本地参考与当前结论

- [原正式规则017](../../rules/017_five_domain_native_sparse_two_hosts.md)、[V2执行policy](../../exp/two_backbone_native_cl_20261017_v2/execution_policy.json)、[V2 run](../../exp/two_backbone_native_cl_20261017_v2/run.py)、[稀疏评价](../../exp/two_backbone_native_cl_20261017_v2/sparse_results.py)、[完整checkpoint](../../exp/two_backbone_native_cl_20261017_v2/checkpoints.py)。
- [方法清单及移出范围](../two_backbone_pct01_20261017/BASELINES.md)、[FOREVER v3 policy](../../exp/two_backbone_forever_20261019_v3/execution_policy.json)、[FOREVER rolling I/O](../../exp/two_backbone_forever_20261019_v3/rolling_io.py)。旧文中的“待准入／当前队列”均为历史，不能覆盖2026-10-25新授权与实际新proof。

当前只完成协议草案和空白主表：10%数据冻结、源快照／condition扩展、实际资源分配与运行均待对应证明；没有本次新QA、模型、评分、参数或效果结果。
