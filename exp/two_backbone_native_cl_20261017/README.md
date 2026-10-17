# 双骨干共用训练框架：2026-10-17

正式当前目录。两个backbone使用完全相同的25份公共Python源码、冻结训练1313QA／完整测试21454QA／19原生阶段、seed42／1epoch／batch16、更新前LR记录与warmup修复、loss-scale1、作者方法核、评测与checkpoint流程。仅模型加载及必要原生输入／位置接口由onevision_bridge.py适配。对应机器由execution_policy.json的backbone字段选择，不另维护一份不同训练循环。

[公平协议](../../summary/two_backbone_native_cl_20261017/FAIRNESS_PROTOCOL.json)、[主表](../../summary/two_backbone_pct01_20261017/MAIN_TABLE.md)。双方使用各自原生冻结processor默认值，取消旧Qwen单独512视觉token限制，不改原图／题／答案／划分；文本渲染统一处理原<image>标记。原生tokenizer／processor差异属于骨干，记录token数，不能把它称作完全相同FLOPs。

每个拟训练阶段必须实际有正lr更新并核验参数摘要改变；未变时停止准入并检查是否为数值故障或作者机制预期，不能自行改作者算法强行产生变化。指纹会读取参数至CPU，其额外开销属于审计，不冒充优化器计算。

运行：同名llmcl-two-backbone-native.service，旧机环境LLMCL_HOST=old2080/LLMCL_BACKBONE=onevision，GPU0–3；新机new3090/qwen，GPU2及当前旧版全量完成后释放的GPU3。新GPU0保留、GPU1痛点研究。SMoLoRA两种骨干都需原生BF16；旧2080Ti无法准入，其OneVision任务明确等待新24GB卡机会，不强制改精度或从十方法中删除。

源码冻结后不热改；修复另起可追溯版本。旧five_domain_native_cl_20261016_sparse的1%结果失效隔离，当前全量保留独立运行；qwen_native_cl_20261017_lrfix与onevision_native_cl_20261017_pct01为前期准备／准入，不能当作本目录正式结果。预检、重载、初始评价和正式训练阶段分别报告，不能把PID存在叫成训练完成。
