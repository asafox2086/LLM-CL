# 学习向量空间／LoRA优先复用：研究接续

新增可追溯CPU入口`analysis/onevision_learned_reuse_queue.py`，不修改任何旧worker源码。使用原唯一`llmcl-onevision-causal-priority.service`，交接前核验KillMode=process、旧CPU等待状态及所有基线/观察器/外来GPU1 PID/startticks；仅替换CPU所有者。最终实际激活证据看`ACTIVATION.json`，当前状态看`queue_state.json`。没有激活证据时不称已经排入后台。

用户最新明确授权“向量空间、cos相近自己识别；这个可以学出来……你试试”，并已委托GPU1方案安排。新两组完整19阶段训练插入共同S1补测之后、固定指令推理之前；固定指令全部4进程/8完整格与原48对照/其他完整方法全部保留。新两组完整21454最终确认放在原48之后、原四组完整方法之前；两组不按测试分数筛选。

只用新GPU1；存在任何实际外来GPU1进程时健康等待，不抢占新GPU0/2/3，不让多个CPU队列竞争同一GPU。两机正式基线继续nohup。当前CPU等待不等于算法训练已经开始。

[实验协议](../learned_reuse/PROTOCOL.md)、[核验结果](../learned_reuse/VERIFIED_RESULTS.md)、[策略检查](CPU_VALIDATION.json)。当前旧路径`language_causal_queue/queue_state.json`在交接后仅作为redirect保留，不恢复旧CPU277043。所有旧GPUworker/source/ID/设置保持原样。
