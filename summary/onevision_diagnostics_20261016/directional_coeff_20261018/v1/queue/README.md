**最新实际进程：** 原86799正常完整结束，继承已核验；新worker96818在GPU1执行principal_coeff_r16初始一致性评测，六真实图文前向门槛通过。初始500/155和训练／F1仍待，实际身份见STARTED_GATE。

# 当前唯一研究队列：直接轻量改造优先

同nohup llmcl-onevision-causal-priority.service，CPU95973/startticks1824688，analysis/onevision_directional_coeff_queue.py。以ACTIVATION.json及实际PID为准；旧86675仅CPU计划退役，继承原GPU1 worker86799/startticks1649052自然完成，所有受保护GPU/基线/CPU观察器保全，不恢复旧所有者。

顺序：当前自然finish→新principal/random两组19TRAIN→两组完整21454→原SVD剩余、18数据控制和所有原完整／旧工作。按最新用户直接试方法的优先级，旧条件只后移不删，不按TEST挑赢家。8项CPU实现及真实48权重／导入门槛，8新调度策略／2实际新旧命令和祖先门槛过；不是GPU收益。源／协议READY后冻结，修复独立版本。

[协议](../PROTOCOL.md) · [准入](READY.json) · [激活](ACTIVATION.json) · [继承回执](COEFF_INHERITED_WORKER.json)。实际空闲0/1单研究流、我们新机最多3卡，尊重他人，两机基线持续nohup。缺XS worker_pid且原86799存活时是健康等待，不重启／抢占。
