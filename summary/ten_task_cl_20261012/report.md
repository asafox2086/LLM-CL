# 八种非回放论文方法的对比进度

2026-10-14T12:19:10.270495+08:00

小模型、小数据、轻量化；五领域十任务；每任务 100/50/50；seed 42；不保存 checkpoint。Frozen/SeqLoRA 不计入八种论文方法。

| 方法 | 状态 | 完成阶段 | GPU |
|---|---|---:|---|
| SMoLoRA | running | 0 | 0 |
| PASs_MoE | waiting_gpu | 0 | — |
| Hyper_LLaVA | adapter_pending | 0 | — |
| PaRSP | adapter_pending | 0 | — |
| OSFT | adapter_pending | 0 | — |
| SpaRTA | adapter_pending | 0 | — |
| FV_guided_Training | adapter_pending | 0 | — |
| ALER | adapter_pending | 0 | — |

adapter_pending 表示适配未完成，不能算成已排队或已启动。
DGG 作者流程含历史样本回放，本轮不参加；论文二维表和源码保留。
