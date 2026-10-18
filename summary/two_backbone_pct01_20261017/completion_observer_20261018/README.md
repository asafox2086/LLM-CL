# 正式V2：全部基线的只读完成核验与主表发布

2026-10-18，新服务器CPU服务`llmcl-v2-verified-results.service`已实际启动，PID382456/startticks15233185。源码为`analysis/v2_verified_baseline_report.py`，两台服务器分发字节一致；没有修改原训练、方法、数据、队列或评价源码。

## 覆盖范围与实际完成数

覆盖Qwen与OneVision两个骨干、各十个既定方法的**1%协议，共20个组合**。首次实际核验为OneVision SeqFT-LoRA和ER-LoRA两条已结束轨迹，即2/20；这不是其他18条已完成，也不是新增模型性能。初始结果与原`COMPLETED_20261018.json`逐项精确一致。

实时核验索引见[VERIFIED_INDEX.json](VERIFIED_INDEX.json)，逐运行证据在`completed/`。不按测试分数挑选运行；同一方法/骨干有多个不同已完成run时停止并要求溯源审计，只有同run、同配置/checkpoint/全部预测SHA的相同副本可去重。

旧`onevision_v2_completion_report.py`仍作为两方法历史快照保留，不热改它。本服务独立覆盖后续方法，并只更新原主表对应数值行，保留分骨干结构、Ours空行、其他未测行及人工说明。

## 接受结果的门槛

1. `run_status`明确completed且第19阶段完成；不凭pipeline的training/evaluation标签。
2. 固定V2框架、401408公共图像预算、seed42、一轮、冻结1313TRAIN/21454TEST；当前/保存的执行源码SHA与配置一致。
3. 全19阶段完成证明和有效性文件齐全；实际日志更新数与batch16预算一致，每次正LR、有限非零梯度、有限loss，参数前后SHA确有变化。
4. 全部74个必要测试格按原ID、原答案、唯一原始Answer F1逐题重算，和保存分数一致。主AP/FWT/BWT、五领域QA均值与原聚合精确核对；缺格不补0，不混用研究155题。
5. 最终checkpoint以流式SHA核验，**不加载/反序列化模型**。原始权重与QA/预测均留服务器，只传输和保存JSON证据。

不强求所有方法的S0逐题相同；方法自身初始化可能改变函数，FWT使用它自己的真实S0，不能替换为其他方法起点。这里核验运行有效性，不把参数变化或passed标记当学习效果因果证明。

## 低成本观察与失败处理

每60秒只发现已完成候选及文件版本；已有核验结果走缓存，不反复重算旧题或读取整个checkpoint。新完成条目才进行完整核验。已封存文件/源码版本变化则停止报告并保留`needs_attention.json`，不悄悄改分或挑更高的run。

SSH观察失败/超时记录为transport wait，不把它解释为训练失败，不启动、停止或重启GPU作业。服务`Restart=no`；真实核验错误需要审计，不盲重启。运行时状态为本目录`observer_status.json`，它不属于模型结果证据。

报告器不会自动Git提交/推送；后续由本聊天核对新结果、显式白名单提交。主Acc/AP定义保持，不因某种聚合有利而切换口径。

## 实际验证与激活

8项CPU门槛/重复运行保护检查通过；两条真实完整轨迹共148个评测格逐题核验，并检查全部训练阶段、checkpoint和原74份预测SHA。主表初始数值未变。[CPU_VALIDATION.json](CPU_VALIDATION.json)。

激活前后，当前GPU358091、CPU调度376997、新机两基线，以及功能调用/温启动/数据因子/旧cap确认报告进程PID与startticks全部一致；未停止任何已有服务。[ACTIVATION.json](ACTIVATION.json)。

本记录是完成结果发布链的补齐，不是新的方法提升，也不替代GPU1的温启动、数据量及领域控制实验。
