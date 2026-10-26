# 10% FOREVER 单 run 完整核验：源码审阅

接受为后续完整终态的独立 CPU 消费入口，**不是已经验证了任何 FOREVER 10% run**。文件为 `analysis/pct10_forever_complete_verify_20261026.py`，最终 SHA256：`00e9f9f97e86b53a029e2441db7df1d1b08a3dfa0cf672ea26cba25282721ca0`。本次仅 AST parse／compile 与静态源码／现有小元数据核对，0 main、模型、CPU 模型门、服务器连接、预测消费或评分。

baseline_review 编写新入口；plain_r8_control 对原 native 复用边界和新源逐字段独立审阅，root 亦交叉核对。源码期发现并修正 request 没有 stage（改为显式阶段参数）、纯 selector namespace 缺 REVISION 两项接口错误，均在任何执行前。另按 root 建议照抄原 beta 运算顺序；这只是保守一致修订，不是已观测的浮点错误或模型故障。

复用边界明确：固定 SHA `2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376` 的10% native verifier，在私有模块 AST 中只替换两处旧 transport 路径为实际 FOREVER READY、一处 frozen_sources 为 READY.dynamic_sources；局部 FRAME／METHODS 指向动态 FOREVER。原19阶段、13105 TRAIN、828 current调用／822正LR／6合法零LR、74格／80316原 Answer F1、21454 final、五域宏／AP及 acquired 计算体不改。继承输出的 scope 标志明确改为 FOREVER，不混入 native14 缓存／索引，不修改旧 verifier 或科学源。

新增核对：真实 queue training exit0／对应 run、各骨干独立2400秒 gate与重载链、29科学源及两个外部 helper、固定作者 revision／selector／callback与loss源；按冻结 TRAIN 用原纯元数据 selector 重建历史 IDs。逐步 main_update 账本与动态阈值／索引、dynamic_XX及final的请求／关闭／实际HF trace、参数移动、额外QA／更新／零warmup／unwrap NLL／分scope资源分别核验。最后核19 rolling commits、committed_latest与latest一致、仅完整S19权重仍在；完整CP与原gradient载荷只流式SHA，不反序列化。请求两epoch与实际trace曝光分列，不把828冒称全部计算，不声称完整训练GPU峰或NEW归档已由本消费者证明。

该入口只处理显式 `--run`，真实 completed／exit0 尚未具备时拒绝。CLI：设置 `LLMCL_ROOT` 为可访问原run与冻结资产的根，运行 `…/pct10_forever_complete_verify_20261026.py --run <ROOT相对完整run> --host old2080 --output <绝对输出JSON> --write`。NEW来源可用 `--host new3090`，仍须对应原 queue／源／资产完整。

每 host/run 在独立 `completion_observer_forever/completed/` 用 O_EXCL 写唯一 VERIFY_STARTED；后续失败保留，不自动重试。完整缓存仅校验并复制，不重评分，换输出路径不能绕过 run 缓存。不自动发现／启动服务／改表，不重评既有1%或native已核验组合。将来只有真实完成并唯一核验后才可用其 proof 入表，最终参数及NEW完整模型留存仍按独立元数据／归档证明报告。
