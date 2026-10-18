# 当前唯一研究队列：真实TRAIN输出反馈与轻量匹配

同nohup服务CPU34096，入口`analysis/onevision_answer_compatibility_queue.py`。以[queue_state.json](queue_state.json)、[ACTIVATION.json](ACTIVATION.json)、实际PID／日志和完整证明为准。旧CPU30721计划退役，KillMode=process；当前同域补偿GPU33289、新基线3256／7965、外来GPU33604等进程PID/startticks精确保全，没有GPU中断。

顺序为：当前GPU自然结束→全部v2补偿控制完成→原20TRAIN×3状态共60条真实Answer F1反馈／1792参数CPU头／两新诊断→原18数据因子剩余条件→两个新完整21454确认→所有原作业。新方法尚未GPU执行，两个新条件都必须保留，不以测试分数选择赢家。公共完整确认及所有旧负结果继续保留。

新9策略／2新旧命令、实际旧分支转发／新分支／gold字段不影响路由检查及祖先检查均通过；CPU不是GPU效果。当前v2冷组全507已精确原cold、F1=14.681，只说明控制实现通过；同域／跨域补偿仍需完成。v1导入故障留证，不恢复旧版或旧CPU。

研究只在实际空闲GPU0／1中选一张，保持一个研究流、新机我们的作业最多占三卡，不动其他用户。源码准入后冻结，故障保留并在新版本修复，不热改或盲重试。

[预定条件／成本／边界](../PROTOCOL.md) · [全部结果](../VERIFIED_RESULTS.md) · [CPU分支证据](../CPU_DISPATCH_CHECK.json) · [原授权与保全](AUTHORIZATION.json)。
