# 当前唯一GPU1部署研究入口

冻结入口：`analysis/edge_visual_priority_queue_20261019.py --watch`；服务仍为`llmcl-onevision-causal-priority.service`，`KillMode=process`，nohup运行。

2026-10-19实际激活CPU788779／startticks7850156；当前权威是本目录`queue_state.json`及`ACTIVATION.json`，以后仍需核验实际PID与启动身份。原CPU739098仅计划退役，所有实际GPU、基线与发布器身份保全。旧入口不恢复。

顺序：当前GPU自然结束（若存在）→四条件GPU部署探针→原12格TRAIN-only→连续LoRA-RITE及完整21454→随机系数完整确认和其余全部已准入工作。所有旧科学源码、数据与确认义务不变，不按TEST分数删负条件。

新worker仅接受物理GPU1，启动前独立核查实际空闲与我们最多占三卡；不使用GPU0，也不抢占他人。GPU1忙时健康wait，缺worker_pid不是故障。四个新路由检查、三个真实CPU张量去重／恢复检查和全部历史策略检查通过；它们不是GPU方法收益。

首次准入前发现两项调度集成问题：未复制历史SHA绑定的操作文档；历史允许GPU0的合成检查被过早替换为GPU1选择器。两项均发生在CPU准入、没有GPU启动，原失败记录保留为PRE_ADMISSION_FAILURE_01/02。现在历史检查使用其原函数，新运行时另执行GPU1限制；没有放宽科学门槛或改动活动旧源码。

GPU完成证明不存在时，不能填正式Ours或部署主表，也不能声称重复视图复用无损或获得速度提升。见[探针协议](../PROTOCOL.md)与[当前证据](../../../RESEARCH_FINDINGS.md)。
