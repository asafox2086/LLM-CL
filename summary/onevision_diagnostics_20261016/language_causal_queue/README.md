# 语言混杂的固定权重干预：版本化接续

2026-10-18研究调度修订，依据用户已委托GPU1研究方案自主安排；并非用户逐个挑选了这些病例。

新增`analysis/onevision_language_causal_queue.py`，不修改旧活动/已冻结GPU源码。继承原CPU空闲/他人进程/源SHA/完整证明门槛，原共同S1推理优先；其后执行固定同域/跨域各probe32和full507的四个进程，再回到全部原48及完整方法、最终确认与稀疏转移评测。

四进程共8个完整测试格，全部执行，不按probe或TEST效果择优。无优化器、无新知识、无新LoRA；[完整协议](../language_cue/PROTOCOL.md)。这不替代缺失的共同S1，不把完整最终差距全部叫遗忘，不生成虚构完整FWT/BWT。

CPU23项通过：11继承策略检查，12新策略检查，包括他人占卡优先、共同S1优先、probe后仍须完整全部格、跳过完成证明、恢复原48顺序。实际GPU输入/重载/成绩仍待。

激活须保留旧CPU/所有GPU实际PID与startticks，核验systemd `KillMode=process`后仅退役旧CPU所有者；同一nohup服务切换至新入口，不启动第二个GPU调度者。旧CPU源码不热改，原GPU作业不打断。详细保全过程见`ACTIVATION.json`；其产生前仅为准备，不宣称激活。

唯一实时入口：`queue_state.json`。其他历史队列路径留redirect，不能将旧MainPID0当故障或重新启动旧服务。只读CPU结果观察器不占GPU。
