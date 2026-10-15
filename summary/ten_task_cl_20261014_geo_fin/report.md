# 本轮并发基线状态

实验：ten_task_cl_20261014_geo_fin；同一划分 SHA-256：`dab25e3940758cfac3e1bb16da5305ce6393dfa4eb22e1c39d4f99a28fdc4208`。

GPU0保留；仅GPU1/2/3并发。每方法独立完成 T1→T10，阶段前／后评测全部十任务。100 训练／50 测试；seed42；无验证、无 checkpoint。

| 方法 | 状态 | GPU | 完成阶段 |
|---|---|---|---|
| SMoLoRA | running | 3 | 5 |
| PASs_MoE | completed | 2 | 10 |
| Hyper_LLaVA | completed | 1 | 10 |
| PaRSP | adapter_pending |  | 0 |
| OSFT | completed | 2 | 10 |
| SpaRTA | completed | 1 | 10 |
| FV_guided_Training | adapter_pending |  | 0 |
| ALER | completed | 3 | 10 |

adapter_pending 不表示已排队或已启动；只有保留作者机制、通过新任务 GPU 预检的方法才能正式训练。
