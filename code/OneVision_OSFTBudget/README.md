# SVD-r8训练参数预算对照（OSFT启发）

当前独立v2：[冻结协议](../../summary/onevision_diagnostics_20261016/osft_budget_20261018/v2/PROTOCOL.md)。两组540672可训练坐标、q/v/rank8/S冻结、沿用当前权重／19阶段，区别为是否执行作者梯度及post-Adam参数投影；无专家、无路由增长。这是参数预算对照，不是原论文完整OSFT，也不是已经有收益的新方法。

源码：analysis/onevision_osft_budget_core_v2.py、trial_v2.py、entry_v2.py、report_v2.py、queue_v2.py（均带onevision_osft_budget_前缀）。作者原始四核函数固定SHA，原baseline代码不改。

CPU原矩阵／导入／算子与全部队列检查通过；v2 CPU75694已接入原nohup服务，目前GPU0/1他人作业实际存活，未启动GPU训练。以[实时权威](../../summary/onevision_diagnostics_20261016/osft_budget_20261018/v2/queue/queue_state.json)为准。两新完整21454与所有旧作业保留。

v1外层autocast降精度在任何GPU／QA训练前被拦截；原源码／激活和故障保留，v2显式关闭重建内部autocast，其他科学设置相同。不得恢复v1旧CPU73625或旧coverage CPU61499。
