# 全量与 1% 持续学习实验进度

状态：preparing_assets；更新：2026-10-16T10:55:39.013092+08:00

GPU0 保留；GPU3 专用于全量组，GPU1/2 用于 1% 训练组。两组均完整评测各自同一套公开有标注测试集。

| 条件 | 方法 | 状态 | 完成训练阶段及全部测试 | 当前阶段 |
|---|---|---|---:|---:|
| full | SeqFT_LoRA | waiting_data | 0/6 | — |
| full | Hyper_LLaVA | waiting_data | 0/6 | — |
| full | PASs_MoE | waiting_data | 0/6 | — |
| full | SpaRTA | waiting_data | 0/6 | — |
| full | OSFT | waiting_data | 0/6 | — |
| full | ALER | waiting_data | 0/6 | — |
| full | PaRSP | waiting_data | 0/6 | — |
| full | FV_guided_Training | waiting_data | 0/6 | — |
| full | SMoLoRA | waiting_data | 0/6 | — |
| pct01 | SeqFT_LoRA | waiting_data | 0/6 | — |
| pct01 | Hyper_LLaVA | waiting_data | 0/6 | — |
| pct01 | PASs_MoE | waiting_data | 0/6 | — |
| pct01 | SpaRTA | waiting_data | 0/6 | — |
| pct01 | OSFT | waiting_data | 0/6 | — |
| pct01 | ALER | waiting_data | 0/6 | — |
| pct01 | PaRSP | waiting_data | 0/6 | — |
| pct01 | FV_guided_Training | waiting_data | 0/6 | — |
| pct01 | SMoLoRA | waiting_data | 0/6 | — |

数据准备及任务数量见 `data/FiveDomainCL/six_domain_cl_20261016_finmme_coda/preparation_status.json`。
每个方法和每组均强制记录 resources.jsonl、memory_summary.json、训练 loss、裁剪前后梯度 L2、学习率和 token 数；没有完整测试矩阵不填最终指标。
