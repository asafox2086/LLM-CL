# 双骨干10%训练／完整测试：正式协议

2026-10-25人类明确确认新增第二主表：**训练10%＋完整测试集**，其余沿用1%正式配置，使用现有空闲服务器卡。当前数据、独立runner与OLD workspace已完成冻结／运输，**待首次启动**；没有10%训练、评分或效果结果。原1%源码、清单、队列与结果不覆盖。metadata完成前的[历史草案](PROTOCOL_DRAFT.md)保留，不再用其“尚未准备”描述当前状态。

## 冻结数据与实际预算

- 条件`pct10`，seed42，单轮当前任务TRAIN **13105 QA**，完整TEST **21454 QA**，原生19个task；有效batch16／microbatch1，对应主任务 **828次optimizer更新**。这些是已冻结清单的实际计数，不是将1%数量乘10。FOREVER等作者额外回放／辅助工作单列，不包含在828中。
- 五域TRAIN实际计数：自然3461、医学984、遥感5723、金融887、交通2050。所有方法与两骨干共用相同ID、task顺序、分层配额及stage内`SHA256(42:stage:ID)`训练顺序。
- 从合法full131018条按原dataset／题型分层、ceil(10%×来源总量)、最大余数及`SHA256(42:ID)`层内前缀选择。已实际核验`pct01 ⊆ pct10 ⊆ 合法full`且逐task嵌套；TRAIN／TEST不交叉。原唯一授权排除`SLAKE:1622`保持，无新增排除。
- 每个新TRAIN选中行保留原full JSONL字节；完整TEST继续引用原文件，SHA、ID顺序与QA对象不改。没有按TEST答案或效果选择样本，没有每方法重新抽样，没有合并小task／改答案／改题型。

manifest原相对路径：`data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json`。本地保留同字节的[FROZEN_MANIFEST.json](FROZEN_MANIFEST.json)，SHA：`02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7`；frozen_ids SHA：`52a1002dca56477b20331ebe864424345ecb4fa9c14eab950a7a8de0adbcc036`；[SPLIT_LOCK.json](SPLIT_LOCK.json) SHA：`c4c9b97858ad12d5a83d603f4c3fd90e9353ed9fbabb20b9bf5b362c1d8b7667`。

源full manifest SHA为`cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952`，源pct01为`36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777`。完整抽样、逐stage计数／更新数及非空答案检查见[SAMPLING_AUDIT.json](SAMPLING_AUDIT.json)，SHA：`218ce70e3f7e4600e25ae227bbac5118a6794a5a59a41b7f2a267dc87e93b9f4`。

## 方法、任务与配置继承

两骨干为Qwen2-VL-2B-Instruct与LLaVA-OneVision-Qwen2-0.5B-OV。每骨干8项：SeqFT-LoRA、FOREVER多模态适配、CL-MoE、SMoLoRA、PASs-MoE、PaRSP、OSFT、SpaRTA，共16组合。Ours保持空行，不自动启用任何探索配方。ER-LoRA、FV-guided Training、ALER等已移出方法不恢复，100% full不在此次授权中。

固定task顺序仍为TextVQA；SLAKE的Modality／Position／Organ／Size／Abnormality／Quantity／Plane／Shape／Color／KG；RSVQA-LR的rural_urban／presence／count／comp；FinMME；CODA-LM的general／regional／suggestion。这是原研究冻结课程，不宣称来源论文规定了该顺序。

继承原正式方法的初始化、学习率、rank、训练目标、QA归约、梯度累积／clip1、gradient checkpointing与作者额外机制。每stage仍重建optimizer／scheduler／GradScaler，单轮、seed42、无验证集；不借10%实验实施optimizer-carry或新算法。各stage实际LR、正更新、梯度、参数变化与额外曝光均记录，不将828解释为所有方法相同总计算。

原公共图像输入保持RGB、等比例BICUBIC、最大401408像素，之后进入各骨干原生processor；原模板、检索、greedy／EOS及各task生成cap不变，图像不覆盖。正式基线保留原128MiB CPU LRU视觉缓存；不替换成探索用GPU-only transport。不同骨干prepared展开可不同，原始QA与图像源一致。

保留原资源／数值差异：SeqFT默认q/v-r8及原AdamW；CL-MoE原2e-4与activation offload；PASs-MoE原1e-4、FP32专家／router例外及HostAdamW；PaRSP原3e-5、OSFT原1e-5，均保留HostAdamW及offload；SpaRTA保留原双秩／router／恢复触发。SMoLoRA仍使用原生BF16，须适合的NEW硬件，不在2080Ti上强行改为FP16。FOREVER作者校准、回放、正则和快照成本原样记录。方法来源与具体设置见[源码审阅](SOURCE_REVIEW.md)及各独立runner，不能用研究用GPUAdam配方替换正式基线。

## 完整测试、主表与新习得

唯一Answer F1沿用原`token_f1`，0–100，不做选项映射。stage0测全部19 task；stage1–18只测当前与下一task；stage19测全部19 task：共74格、80316条逐题生成／评分，其中最终完整21454条。未测格为null，不补零，不伪造完整矩阵或历史峰值ForgetRate。

[第二主表](MAIN_TABLE.md)与1%同口径报告S19最终五域QA均值、五域等权宏F1、19任务等权AP19及FWT／BWT。最新人类科研重点是**新习得**；BWT仅保留原正式覆盖的描述，不作优化／选择目标，不另研究遗忘。论文总体最终F1与完整权重／GPU内存评估目标保持。

新习得单列：各task刚训练结束格`R[t,t]`的19任务等权均值为新习得AP19；各域按task TEST QA数加权对角格，再五域等权，得到新习得五域宏。它们来自不同task时点的模型，不能写成S19最终成绩或单一域末模型。原74格也可给出所属task训练前→后的功能变化，无需额外测试请求。

只有完整19阶段／74格、逐ID／引用／原评分器、源／数据与完整S19 checkpoint全部核验后填表。10%与1%分表，空白不是0；不得移入1%数值、按TEST追超参、改指标或只选择有利方法／域。

## 独立runner、checkpoint与部署证据

独立目录为`exp/two_backbone_pct10_native_20261025`和`exp/two_backbone_pct10_forever_20261025`。原native／FOREVER源码未覆盖。run只扩`pct10`入口与准确train_fraction／primary_score metadata；FOREVER另将两处1313训练量guard改为manifest精确总数，19task不改。训练／方法数学、原评分器和pipeline.launch保持。

首次构建已在0模型状态下完成；发现FOREVER规模guard后，在首次启动前作纯断言修订，初始receipt保存在[PRESTART_GUARD_AMENDMENT](PRESTART_GUARD_AMENDMENT/INITIAL_RUNNER_BUILD_RECEIPT.json)。最终[RUNNER_BUILD_RECEIPT.json](RUNNER_BUILD_RECEIPT.json) SHA：`9d3a2c76b0229a25be6ecbf2e17b5ab732d6dd0ac451797cec8c518786e1a573`，包含修订、逐文件原／新SHA与byte-identical名单，不能称只有run.py变化或已运行通过。

正式每组合fresh预训练基座开始，不从1%／full已学CP继续，不使用preflight训练过的图。原preflight、独立checkpoint_probe和三阶段调用路径不改；正式training新建run，不传resume。SeqFT保留全部19阶段完整checkpoint，其余V2方法保留最终S19；FOREVER保留原rolling stage-atomic commit，持久指针与相关文件就绪后才移除较早CP。阶段恢复必须匹配本10% manifest及源码，不能拼接失败预测。

完整参数／dtype构成、去重权重及必需持久状态bytes、相对基座增量与CP容器bytes分别实测，不把trainable参数当完整参数量。CPU RSS／PSS／USS、CUDA allocated／reserved、NVML和训练前后／采样峰分列；固定五窗推理峰、首词延迟、速度没有本10%独立测量就留空，不抄1%或用训练峰估计。

## 已运输workspace与首次启动状态

OLD独立overlay为`/dev/shm/llmcl-pct10-1084-zu7e3hb_`，原根`/data2/liyapeng_grp/program/LLMCL`的模型／来源资产按只读链接复用；旧磁盘quota满，因此本次输出使用RAM tmpfs，**不是耐久归档**。实际[OLD_TRANSPORT_RECEIPT.json](OLD_TRANSPORT_RECEIPT.json)记录84文件逐SHA、原source不改、manifest一致、0模型／0评分；SHA：`9c30b2912af3f29b6be836a0ef7e8f6ab1b1b3e574ca073389e8926c141892f4`。[WORKSPACE_LOCATION.json](WORKSPACE_LOCATION.json) SHA：`8cca19b99ae3c2c235329ee366f101627146acb325d6243c5d3a743de3abed33`。

计划由单一独立dispatch复用原pipeline.launch管理OLD四卡14个非BF16组合；两SMoLoRA待NEW原生BF16空卡。NEW0继续保留，NEW mode有与现有1%调度交接的显式guard，当前不启动NEW10%以抢现有卡。所有启动前ownerless、足够显存／低util、实际GPU UUID及UID／PID／startticks／PGID仍须核对；不向他人或1%worker发信号。

参数文件、QA、图像和密钥不进Git。RAM容量、并发临时文件、SeqFT19 CP及作者状态须按真实占用和NEW归档进度协调；瞬时free-space检查不能冒充总量预留或抗重启保证。只在校验归档后按原保留策略处理副本，不删旧证据腾空间。

**本文定稿时仅已freeze待首次启动。** 未来只有实际ACTIVATION／PID／phase与完整proof才能改状态；准备成功、source review或数据冻结均不写成训练／效果成功。原1%队列继续原样，不重跑sealed组合，关闭科研分支不复活。
