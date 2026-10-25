# 10% native正式完成核验：适配说明

本入口仅补齐完整结果的核验／JSON消费路径，不是科研方法或质量收益。源码：`analysis/pct10_verified_baseline_report_20261025.py`，最终SHA `2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376`。原核验器 `analysis/v2_verified_baseline_report.py` SHA `d7aac920f738c6ee8fde9a636560f2d2d6ee7ea1ba9de9e030e6548b4b4cad0a` 及旧1%训练／核验／表格源码均未修改。

## 完成准入与最新修订

必须先有当前主机对应的 `queue_old.json` 或 `queue_new.json`。候选同时满足：原run状态 `completed`／S19、非preflight、队列job为 `completed_unverified` 或 `completed_verified`、job的 `last_exit` 为同一run的 `training`／returncode0。仅run_status completed不够，因为worker可能写完状态而尚未退出。

第一版新脚本SHA为 `632e17c7285c4384e9ec4cbc4af0414d817e577fc5b3b595ea8f5cfec970c9ef`；root审阅后的首项修订把candidate的可选queue检查改为必须存在，并让discover在缺queue时返回空候选、明确的exit0未证明原因及 `scoring_started=false`，修订后SHA为 `71cb729c06c4069c30c3e690f67c0e8cc8026b930aa92641b0b62e517d788bae`。此时不扫描预测或评分。未来无queue归档模式必须另有真实exitreceipt准入，当前不自动放松。随后按原runner的warmup数学修正了下述正LR计数检查，形成当前最终SHA；没有修改训练实现或超参数。

## 对旧核验器的配置适配

- ROOT改为必须的 `LLMCL_ROOT`，直接操作指定实际overlay／记录根；不含SSH、RPC、子进程或模型调用。
- 独立native目录 `exp/two_backbone_pct10_native_20261025`，条件pct10；7项native方法×2骨干共14组合。FOREVER由其完整动态适配另行审阅，不能套旧零动态触发proof；Ours不在本入口范围内。
- 固定manifest SHA `02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7`，13105 TRAIN／21454完整TEST，19阶段／828主任务optimizer调用（822正LR、6合法零LR）／74格／80316逐题评分。
- 执行源和run内runner_source逐SHA相同，且源码集合精确绑定原84文件运输receipt的native全部Python源。运输receipt SHA固定为 `9c30b2912af3f29b6be836a0ef7e8f6ab1b1b3e574ca073389e8926c141892f4`，不信任run自行声明的一套替代源。
- 原评分器SHA固定为 `16c59998413d12b820d6f57269f9e1f45ac78e08595d20bf1f4fbc967fd94c4b`，仍为原始Answer F1，不选项映射、不新增提取或语义评分。
- 新增acquired AP19为对角格 `R[t,t]` 等权均值；各域先按task完整TEST QA数加权对角格，再五域等权得到acquired macro。明确这些来自不同task时点，非S19或单一域末模型；从已核74格计算，无额外生成或评分。

## 继承的核验断言

保留完整run路径边界、seed42／1epoch／公共输入框架／401408像素声明、逐数据文件SHA、逐预测ID及原references顺序、原逐题token_f1相等、每格count及均值。19阶段durable／validity／真实参数变化、实际正LR更新数、有限loss与非零有限梯度均核查；最终AP19、FWT、BWT、域内QA加权F1及域宏与保存值一致。S19最新checkpoint必须stage／method／bytes正确并流式复核完整SHA，绝不反序列化模型。核验前后candidate文件stat签名一致，自身源码不得热改。

旧核验器的 `sha` 与 `eligible` 函数在AST层面与新入口完全相同。不存在按TEST挑选最好run：相同方法／骨干若出现不同完整run，必须显式审查来源，不能自动取高分。

## 原warmup数学与10%预期计数

原 `training_validity.warmup_updates(N)` 为 `min(floor(.03*N), N-1)`，原run在构建HF cosine scheduler后记录每次实际 `learning_rate_used` 再step scheduler；有warmup时首个optimizer调用所有组LR为0。原 `validate_stage` 本来允许 `0 < positive_lr_updates <= updates`。不能继承1%短阶段“每次调用均正LR”的过强假设来否决合法10%结果。

新consumer按冻结各task TRAIN量计算N，要求事件step连续为1..N、每条 `warmup_updates` 等于原公式、每组实际LR有限且非负。当warmup>0时只允许且要求step1所有组LR为0，其余调用均须至少一个组LR为正。实际正LR计数必须同时等于 `N-int(warmup>0)` 和原validity记录。总828是optimizer调用，不能写成828次正LR更新。

以下为冻结manifest的纯元数据预期，不是已经观察到训练完成：

| Stage | optimizer调用N | warmup | 预期正LR | 预期零LR |
| --- | ---: | ---: | ---: | ---: |
| 1 | 217 | 6 | 216 | 1 |
| 2 | 7 | 0 | 7 | 0 |
| 3 | 11 | 0 | 11 | 0 |
| 4 | 16 | 0 | 16 | 0 |
| 5 | 5 | 0 | 5 | 0 |
| 6 | 9 | 0 | 9 | 0 |
| 7 | 3 | 0 | 3 | 0 |
| 8 | 4 | 0 | 4 | 0 |
| 9 | 1 | 0 | 1 | 0 |
| 10 | 2 | 0 | 2 | 0 |
| 11 | 8 | 0 | 8 | 0 |
| 12 | 4 | 0 | 4 | 0 |
| 13 | 107 | 3 | 106 | 1 |
| 14 | 107 | 3 | 106 | 1 |
| 15 | 141 | 4 | 140 | 1 |
| 16 | 56 | 1 | 55 | 1 |
| 17 | 31 | 0 | 31 | 0 |
| 18 | 68 | 2 | 67 | 1 |
| 19 | 31 | 0 | 31 | 0 |
| 合计 | 828 | — | 822 | 6 |

最终proof会保存逐stage预期／实际positive计数及warmup信息，index明确区分828调用、822正LR与6零LR。FOREVER另行适配的S2真实7步预检warmup为0，不受此处长阶段规则影响；FOREVER仍不在本native consumer scope。

## 唯一评分缓存及输出边界

默认 `--mode discover` 只读，且不能带 `--write`。`verify`／`once`／`watch` 必须显式 `--write` 才能开始原F1核验并写JSON；任何模式都不启动模型。主进程持文件锁，每host／run在评分前以排他创建方式写 `VERIFY_STARTED`，成功保存proof及绑定其SHA的COMPLETE receipt。已完成缓存复用只核对candidate和receipt，不再调用F1；失败保留started／failure，后续不得自动再次评分。

输出仅在本目录下的 `completed`、`VERIFIED_INDEX.json`、observer／异常记录。index只消费完整proof，空结果不当零；不写根README、1%文件、现有MAIN_TABLE或调度队列。完整参数与推理资源不是本脚本估计的结果，未独立实测时仍为空。

## 本次实际执行范围

只做源码写入、AST parse／compile、静态import清单及继承函数AST比对。**0真实run核验、0原题评分、0模型、0服务器连接、0表格发布。** 未执行脚本模块或main，也未借当前运行中的partial质量选择配置。root可另行进行无候选的实机discover检查，不能把该检查写成科研收益或完整结果。
