# 只用训练探针选择旧LoRA：离线审计

每域到来时只允许原基座和此前领域的独立LoRA；先冻结TRAIN探针决策，再读取测试输出。本报告不是在线算法或Kmax容量实验。

| 当前领域 | 只从此前选择 | 原基座NLL | 所选旧适配器NLL | 本域独立LoRA NLL | 训练探针最好／次好差 |
|---|---|---:|---:|---:|---:|
| natural | base | 5.853 | 5.853 | 1.112 | — |
| medical | isolated_natural | 9.595 | 3.646 | 4.407 | 7.2195 |
| geography | isolated_natural | 3.396 | 0.933 | 0.688 | 0.0230 |
| finance | isolated_geography | 3.955 | 3.132 | 2.477 | 0.2414 |
| traffic | isolated_medical | 2.303 | 2.298 | 2.228 | 0.0038 |

医学域未用自己的LoRA时，TRAIN探针选择自然域旧LoRA，NLL明显低于原基座。这支持继续检验复用价值，但自然域训练347QA、医学99QA，且格式／语言适应可能贡献增益，不能因此断定专业知识已经足够。

交通域候选TRAIN损失差很小；仅两个探针的选择可靠性不足，必须用预定的更充分TRAIN探针和真实在线干预再核验，不能按这批测试分数更改选择。

限制：

- Offline audit of completed independent adapters, not actual online training or learned routing.
- Retrospective diagnostic: these test subsets were already inspected during research; not untouched confirmation.
- Domain identity known; only two in-training probes/domain, no independent validation set.
- Earlier adapters have unequal training budgets; no claim generic knowledge replaces professional knowledge.
- Old isolated control dropout streams differ; does not fix the failed matched native reproduction gate.
- Per-domain100 NLL and10 generated outputs; not full21454 test or final CL performance.
- No modules are added by this audit. Kmax1/2/3/5 automatic reuse/growth remains untested.
