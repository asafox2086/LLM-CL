# v2唯一队列

修复前版本因初始一致性门槛停止，不恢复原95973/96818。独立v2在TRAIN探针与CPU准入后接续；基线不停止，所有原作业保留。实时状态以ACTIVATION.json、queue_state.json及实际进程为准。
