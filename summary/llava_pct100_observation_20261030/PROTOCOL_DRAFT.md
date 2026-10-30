# LLaVA 100%训练数据观测协议草稿

这是训练数据比例100%，不是FullFT（全参数训练）的新任务。只扩LLaVA；不扩Qwen100%，不设计新方法。第一批已明确：先原PaRSP，随后原OSFT作为异机制对照。当前只有metadata入口准备，没有模型准入、队列或启动。

## 既有数据与来源

沿用原已授权canonical full variant：`data/FiveDomainCL/five_domain_native_cl_20261016_sparse_labeled_20261017/full/manifest.json`，SHA `cf737555dab3646c9916dc63a66f1d52285eebbeed4aa414e807ae5b08a0d952`。有效TRAIN **131018**，完整TEST **21454**，19个原native task。

原未排除full为131019，manifest SHA `3dac40b318c4bc7c792eeebc4e5667b172cfdf42ae835b92f4fe56998ef44504`。唯一差异是T04空gold TRAIN `SLAKE:1622`；不是本次新排除。原2026-10-17收据记录用户授权“只有这一个有问题？那就把这个直接踢掉就可以了”，首次提交 `95f2be98966ee99bb8f4d043114a9939f665d79f`，时间2026-10-17T16:23:05+08:00。收据SHA `bfc58fd0571b3d07b00e900394406fc954a63cce6985dc2340a0315ea341a1f4`，原派生source `prepare_full_labeled_revision.py` SHA `6a3e098920e810162ec1b26168c8612e9617e3a280a8839f67b6e073a7561e2a`。当前只读核验保留行字节相同，其余37个QA文件完全复用，绝不重跑派生脚本/重split。

1%1313、10%13105的原TRAIN逐task嵌套于该full variant，两者都不含被排除ID。19个TEST逐文件实际count/SHA在原full/既有variant/1%/10%一致；每条件总21454。没有读取TEST答案进行选择、生成或评分。原更早六域ancestor manifest当前未保留，现存19task full manifest保存其原SHA；这个祖源文件没有在本轮伪装成已实复核。当前19task全量QA文件均真实在盘且实际SHA闭合。

## 冻结配置与隔离边界

LLaVA骨干沿当前baseline的原完整模型、原method图和参数，不合并、fold、量化、剪枝或转换基线。epoch1、seed42、effective batch16、micro1、原固定QA顺序hash；保留原各方法LR/初始化/保护/校准预算、warmup公式、AMP、梯度裁剪、原图像processor/输入规则、生成与metric源码。不会重新实现manifest的生成预算字段来替换当前原生生成逻辑。

100%一轮的理论原native optimizer迭代数为8197；是否正LR更新依原warmup/scaler实际日志如实记录，不因新数据量改公式。原1%93与10%828迭代不动。每阶段完整checkpoint、最终checkpoint和全部主表QA/metric/proof保留；沿原暂停边界为研究让卡。临时训练状态/训练峰与最终部署权重/推理峰分别报告。

NEW既有 `exp/two_backbone_native_checkpoint_20261027/run.py` 原CLI支持 `full/pct01`。OLD原先缺该入口，且原policy缺old2080的full host条件；现已仅将原25科学文件逐字运输到OLD不存在的同名隔离目录，唯一policy语义增量是 `hosts.old2080.conditions.full=[0,1,2,3]`，reserved仍为原空列表。NEW active与OLD旧1%/10%全部源/policy逐SHA保持。没有新pct100科学runner或allocator。使用既有数据condition `full`，论文标100%，与FullFT方法名分开。原8方法及FullFT原图仍为后续候选，但本批只PaRSP→OSFT，不把候选清单冒称全部启动；实际槽由root另定。不执行prepare_native/旧full派生脚本，不重采样，不复用另一数据比例已训练checkpoint作为新起点。

主结果与既有表同样按19小类/五域汇报，附原已要求的学习指标和实际总/新增部署权重系数、dtype字节、实测推理GPU内存。缺测不填。当前只核数据/配置可行性，不预设100%结果会支持任何原因或方法。

## 逐任务固定计数

| Task | 域 | 小类 | TRAIN | TEST | batch16单轮迭代 |
|---|---|---|---:|---:|---:|
| T01 | natural | — | 34602 | 5000 | 2163 |
| T02 | medical | Modality | 1070 | 228 | 67 |
| T03 | medical | Position | 1695 | 375 | 106 |
| T04 | medical | Organ | 2542 | 507 | 159 |
| T05 | medical | Size | 650 | 124 | 41 |
| T06 | medical | Abnormality | 1410 | 311 | 89 |
| T07 | medical | Quantity | 437 | 89 | 28 |
| T08 | medical | Plane | 505 | 116 | 32 |
| T09 | medical | Shape | 83 | 12 | 6 |
| T10 | medical | Color | 268 | 65 | 17 |
| T11 | medical | KG | 1174 | 267 | 74 |
| T12 | geography | rural_urban | 572 | 100 | 36 |
| T13 | geography | presence | 17008 | 2955 | 1063 |
| T14 | geography | count | 17089 | 2947 | 1069 |
| T15 | geography | comp | 22554 | 4002 | 1410 |
| T16 | finance | — | 8864 | 2233 | 554 |
| T17 | traffic | general | 4884 | 500 | 306 |
| T18 | traffic | regional | 10727 | 1123 | 671 |
| T19 | traffic | suggestion | 4884 | 500 | 306 |

实际绝对TRAIN路径、每文件SHA和所有跨条件核对见 DATA_EVIDENCE.json；TRAIN图片仅stat核验见 TRAIN_IMAGE_AVAILABILITY.json：42256个唯一文件、131018次引用，全部存在且非零；没有冒称重hash/解码了18.88GB图片内容。


## 现有入口与评分标签边界

OLD入口：`/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_checkpoint_20261027/run.py`，原字节SHA `ea347e655c8f3a20d07c86568b351ab1615868bcf9b88d209756c45d1cef1280`。原full.load_protocol函数AST已独立以stdlib环境执行，所有manifest/splitlock/frozen IDs/38文件SHA/计数/ID互斥断言通过；只做原完整性加载，没有生成/答案评分/模型导入。host full门禁对OLD0..3可满足，这不表示任何卡当前空闲。

隔离policy SHA `edb12b9fd320fc88476f1ecf0bca5d84ddb05f9f88c364db0b6cd8cd07e7d71d`；政策唯一语义diff见OLD_FULL_ENTRY_PREPARATION.json。原顶层allowed_gpus旧字段仍保留来源值，原直接run实际检查host conditions与reserved；不启动pipeline/旧broad allocator。

原full runner的config.primary_score='rougeL'是保留的旧metadata标签；25科学文件中该字段只在config赋值处出现一次，不参与optimizer、LR、earlystop或checkpoint择优。原一轮/固定更新循环无best-checkpoint选择。实际runtime_utils照原样计算并保留token_f1/exact_match/rougeL多指标。本100%正式报告只读取原token_f1字段，沿analysis/answer_f1.py原SHA `16c59998413d12b820d6f57269f9e1f45ac78e08595d20bf1f4fbc967fd94c4b`及原metric_lock SHA `a146f037f835afb66251ce35498a4a5df8e7ea97468725b49ea34043f7d1c948`；绝不把Rouge或R_primary当F1，不更改生成或重新评分。

首PaRSP的RUN提案为上述入口同目录下 `runs/onevision/full/PaRSP/training_full_20261030_first`，当前未创建。位于OLD persistent，原producer每完成stage保存完整checkpoint，最终stage19保留；用户授权手工退旧非最终payload时保当前恢复点/研究锚/最终及全部小记录，不借此修改训练。任何真实GPU启动仍需root的当前槽/身份准入，此协议没有启动它。

100%沿epoch1，故训练数据和optimizer迭代数从93/828同时增至8197；这是既定条件扩展，不是纯数据量的单因素因果实验。若floor数量仍相同而新习得恢复，可反驳floor数量本身足以导致质量低的强说法；不恢复也不自动证明保护造成损害。两原方法只对本数据现象提供不同机制证据，不宣称多数CL普适。
