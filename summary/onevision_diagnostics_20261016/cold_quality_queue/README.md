# 当前唯一研究调度：冷启动继承＋功能质量调用

2026-10-18实际激活。原同nohup服务`llmcl-onevision-causal-priority.service`的新冻结CPU入口为`analysis/onevision_cold_quality_queue.py`，PID389122/startticks15440343；旧376997仅CPU计划退役，`KillMode=process`保全GPU与所有受保护进程。当前GPU358091/startticks14710479仍自然执行原质量硬调用的完整21454题确认，**没有启动新的GPU组合实验**。

用户最新目标是从冷启动弱习得出发设计两个模块，使领域最终Answer F1明显提高。这里将A完整功能继承与B TRAIN质量硬调用接通为四格消融，未改任何原活动/排队GPU源码、训练ID、优化预算、旧候选或评分。

## 预先固定的接续顺序

1. 原可用共同S1、相似度训练与冷池四视图依赖保留。
2. 当前质量硬调用完整确认自然完成并通过原SHA证明。
3. 原温启动19阶段：只改变新专家完整A/B初始化。
4. 新温池四视图pilot：原500NLL/155预测精确重载，通过后校准TRAIN质量头；全部辅助视图报告。
5. 全部18条既定数据量×初始化轨迹，均保留1/4步终点。
6. 原可学习冷池完整21454、原温池几何调用完整21454、新温池质量硬调用完整21454，补齐四格。
7. 原质量软调用完整21454及全部cue、48机制、固定相似度、其他方法、cap3与稀疏转移继续。

顺序依用户更新的方法开发目标固定，不读取TEST成绩决定调度或取消条件；未减少总计算。新格11的完整候选固定quality_hard，是对应B质量校准的独立因子，不按辅助四视图选最优。原冷池quality_soft仍保留。

## 真正的依赖与失败门槛

新温池CPU计划在实际warm19封存且全部初始化/几何门槛通过后，才用空CUDA环境生成；不是使用尚不存在的父状态预填计划。CPU计划绑定实际父证明/权重及全部源SHA。GPUpilot先精确重载父池，再校准原43TRAIN/172次NLL前向/16参数400CPU步；完整确认需新进程精确重载，不能拼接旧预测。

新增66项CPU策略、祖先82/25/28项策略及8项实际存活继承/命令检查通过。这些证明调度门槛，不是模型收益。只读`llmcl-onevision-cold-quality-report.service`已通过一次真实CPU复算后nohup启动；只在完整证明/CPU计划变化时重算四格，缺失保持待测。初始只有00/01两行155题诊断，0/4完整格。

权威实时入口是本目录`queue_state.json`与实际PID/startticks/日志。旧`causal_feedback_queue/queue_state.json`已redirect，禁止恢复376997及更早所有者。其他用户实际占用GPU1时健康等待，不因缺worker_pid重启；故障保留，不盲重试、不热改源。新GPU0保留，两机正式基线持续nohup。

[两问题与方法](../warm_quality_20261018/PROBLEM_AND_METHOD.md) · [四格结果](../warm_quality_20261018/FOUR_CELL_RESULTS.md) · [授权](AUTHORIZATION.json) · [实际激活与受保护进程](ACTIVATION.json) · [策略检查](CPU_POLICY_CHECK.json) · [继承检查](INHERITANCE_CPU_CHECK.json)。
