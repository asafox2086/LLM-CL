# 10% FOREVER 单运行结果核验入口：独立源码审阅

结论：接受源码。最终 `analysis/pct10_forever_complete_verify_20261026.py` SHA256 为 `00e9f9f97e86b53a029e2441db7df1d1b08a3dfa0cf672ea26cba25282721ca0`；这不是实际结果核验通过，也没有启动新的 observer 服务或自动消费两个 FOREVER 结果。

已对照真实 producer 审阅原10% native 核验入口及新包装。私有加载固定 SHA 的原核验器，仅替换两处 transport 来源路径和一处 frozen-source 映射，局部配置动态 FOREVER 的 FRAME/METHODS；原逐题 raw Answer F1、19阶段／74格／80316次评分、21454条完整 TEST、828次主 optimizer 调用（822次正LR、6次合法零LR）及最终完整 checkpoint SHA 核验保留。FOREVER 使用独立单运行排他缓存，不进入 native14 observer。外层对应实际 producer 核对动态源、两外部 helper、作者固定源、真实 queue training exit0、GPU gate、24步校准和 dynamic/final 回放序列、实际回放 trace／额外成本、rolling 提交与最终完整 checkpoint；没有把共同主训练配置移植声称为逐字节复现作者 HF 阶段重启。

初审发现并已修正两个真实元数据接口问题：`begin_replay()` 的 request 没有 stage，现由调用者显式传入；抽取的纯 metadata `source()` 依赖 REVISION 全局，现注入固定作者 revision。未改模型或训练 producer。

最后实际检查了相对已接受 `0adc23c82d22d3d67b3bfd4a5896a73a825576dbee28285e11b017eecb6e195d` 的差异：唯一变化是把简写 beta 表达式恢复为作者原运算顺序 `ratio=mu/(mu0+1e-12)` 后 `min(3.0,max(0.5,1.0+(ratio-1.0)))`。将这两行逆替换回旧单行可精确重建上述旧 SHA，确认其余字节未变。此修订用于遵守原浮点表达式合同，并非已观测到实际浮点故障。

范围：源码、producer 字段及 stdlib AST／字节差异检查；没有运行 main、真实 consumer、评分、模型、训练或服务器操作，没有读取真实结果分数、QA 或权重载荷。真实完整运行仍须经该入口唯一核验后才能报告为 verified。
