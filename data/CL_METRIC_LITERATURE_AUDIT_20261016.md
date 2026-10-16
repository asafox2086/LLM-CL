# 多模态持续学习的遗忘指标：顶会原文核验

2026-10-16。研究问题：是否必须报告当前实现的历史峰值型`ForgetRate`，能否精简阶段评测。本文是文献核验与方案建议，**尚未更改运行中的评测策略或指标公式**。

## 直接相关的MLLM持续学习论文

| 正式论文 | 主实验所报指标 | 遗忘定义／原文位置 | 是否必须取历史峰值 |
|---|---|---|---|
| SMoLoRA，ICCV2025 | 各任务最终性能、AP、MAP、BWT、MIF | [作者正文§4.2，式10–12及Table1](https://arxiv.org/html/2411.13949)。BWT为旧任务最终性能减该任务刚学完时的性能，再对旧任务平均。 | BWT不需要；但该文额外报告MAP及阶段曲线，需要更多中间旧任务评测，不能声称本文精简方案完整复现其全部指标／图。 |
| CoIN，NeurIPS2024 Datasets and Benchmarks | 各任务性能、MAA、BWT | [正式论文§3.1.3性能指标，PDF第5页、Table2/3](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a45500d9eda640deed90d8a62742be5-Paper-Datasets_and_Benchmarks_Track.pdf)。使用BWT衡量遗忘，MAA衡量整个训练过程表现。其BWT分母为任务总数T且包括最后任务零差项，与SMoLoRA的T−1归一化需区分。 | BWT不需要；MAA需要已学任务的中间分数。 |
| CL-MoE，CVPR2025 | 各任务最终准确率、AP、AF | [正式CVPR论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.pdf)，[作者正文§5 Dataset and Evaluation Metrics](https://arxiv.org/html/2503.00413)。明确写AF=(1/(M−1))Σ[i=1..M−1](m[i,i]−m[M,i])。 | **不需要。AF使用刚学完减最终，不是历史最大值减最终。** |

不是所有论文都同时报告AP、遗忘、FWT、BWT四项；也不能只凭AF、Forgetting Rate、Forget等名称断言其公式相同。这里不以传统图像分类CL替代MLLM持续微调的直接依据。

## 与本仓库现有实现的区别

令R[s,t]为训练完任务s后，在任务t完整测试集上的分数；N=6；R[0,t]为初始基座分数。当前运行的`result_tables.py`定义：

- AP = mean(R[N,t], t=1..N)。
- BWT = mean(R[N,t]−R[t,t], t=1..N−1)。
- FWT = mean(R[t−1,t]−R[0,t], t=2..N)。
- 当前历史峰值ForgetRate = mean(max(R[s,t], s=t..N−1)−R[N,t], t=1..N−1)。

CL-MoE的AF = mean(R[t,t]−R[N,t], t=1..N−1)。在本仓库BWT使用相同任务集合、同样N−1归一化的前提下，**AF恰好等于−BWT**。它有正式CVPR论文依据，但与当前历史峰值ForgetRate是不同定义；不能直接换公式而保持无说明的同名列，也不能当成与BWT独立的新证据。

## 可据此采用的精简方案（建议，待用户确定）

保留AP、BWT、FWT；如需对齐CL-MoE，可附列AF并明确AF=−BWT。不再要求历史峰值ForgetRate、MAP或MAA，才可避免中间反复回测所有旧任务。

1. 初始评测全部六任务。FWT实际只必需T2–T6的初始分数；为保持已有有效评测和基座对照，这里保留全部六任务。
2. 训练完T1至T5各阶段，仅评测刚学完的任务及下一个任务；下一个任务的结果就是其学习前分数。
3. 训练完T6，完整评测全部六任务；T6的刚学完分数与最终分数是同一项，不重复评测。

共6+5×2+6=**22组任务测试**，而非7×6=42组。按现有六个完整测试池大小，对应每方法每条件98341次逐题生成，原方案184562次，减少约46.7%。每次被安排的任务仍完整测试，不抽测试样本。

精简后的7×6表应保留未测单元格为空值，不补零、不插值、不拼旧预测；它不再是完整矩阵。主表仍可报告六任务最终分数、AP、BWT、FWT；无法提供原计划每条任务曲线七个阶段都有真实测点的完整轨迹。现有聚合器要求42个单元格齐全，因此实施精简时须同步修改评测调度、必需单元格检查、图表及协议记录，不能仅跳过评测而保留原聚合条件。
