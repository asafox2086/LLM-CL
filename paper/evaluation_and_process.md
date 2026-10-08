# 论文如何展示持续学习：指标与全过程

## 先看本仓库已复原的报告

- [学习收益与保留](../summary_cv/learning_gain.md)：同题学前/学后净收益、新增正确题和后续保留，区分恢复与超越基座。
- [Qwen2-VL 全过程](../summary_cv/process.md)：全部 10×9 矩阵、任务与领域曲线、切换冲击、逐题回答演变。
- [外部知识题过程](../summary_cv/knowledge_probe.md)：基座及每个阶段对同一组外部学科知识题的表现与答对/答错转变。
- [T5 全过程](../summary/learning_process.md)：8×7 矩阵、过程曲线和同口径指标重算。

这些报告重用保存的阶段记录。原论文采用的模型、数据和任务评分与本实验不同；可以复用计算结构，不能直接把本地 ROUGE-L 称为原论文的分类准确率。

## 论文实际展示了什么

| 论文与定位 | 展示方式 | 这里的对应报告 |
|---|---|---|
| [MIGU §4.3、Fig.3 / Fig.12](https://aclanthology.org/2024.findings-emnlp.379.pdf)；[本地 PDF](UNLOCK_2406.17245.pdf) | 一边画新能力 HumanEval 的训练过程，一边画独立 HellaSwag / Winogrande / ARC-Challenge 的保留曲线；Table 4 比较早期与近期学到的领域 | 同时给新任务收益、旧任务冲击和外部固定知识题曲线。外部题使用 MMLU 子集，是本地补测，不复刻论文三套题库 |
| [SAPT §5.1.2、Fig.3、Table 3](https://aclanthology.org/2024.acl-long.625.pdf)；[本地 PDF](SAPT_2024_acl-long-625.pdf) | AP / F.Ra / BWT / 特定定义的 FWT，注意力热图，额外未见任务成绩 | 重算 AP / F.Ra / BWT；GEM FWT 单列。SAPT 的独立训练 FWT 和注意力机制可视化需要不同于分数矩阵的数据 |
| [MLLM-CL §5.1、附录指标说明及 Fig.7](https://arxiv.org/html/2506.05453v1)；[本地 PDF](MLLM_CL_2506.05453.pdf) | MFT（刚学完）、MFN（最终）、MAA（各阶段已学任务平均再平均）、BWT、完整增量任务结果；还有单任务 Oracle | 三项过程/最终指标和逐任务变化已复原；Oracle 未训练，差距标 N/A |
| [MedCL-Bench Fig.3 / “Forgetting Dynamics”](https://arxiv.org/abs/2603.16738)；[本地 PDF](MedCL_Bench_2603.16738.pdf) | 各阶段已学任务平均表现、相邻阶段 transition shock、不同任务顺序、任务家族遗忘差异 | 已学任务曲线、任务级变化、分领域曲线和固定旧任务集合的冲击。只有一个顺序，不能重建其他顺序或误差范围 |
| [Replay-free Medical VLM Table 1 / §2](https://lihe50hz.github.io/ML4H_2025_Replay_free.pdf.pdf)；[本地 PDF](Medical_CL_ML4H_2025.pdf) | 区分最后阶段 Last 与学习过程 Average（阶段已学任务均值的均值），展示逐医学数据集结果 | MFN / MAA 与领域曲线；本地 VQAv2 / VQA-RAD 划分不等同其医学任务序列 |
| [O-LoRA §4.1.2](https://aclanthology.org/2023.findings-emnlp.715.pdf)；[本地 PDF](O-LoRA_2023.findings-emnlp.715.pdf) | 最后阶段所有任务的平均准确率 AA | 同一计算结构可由最终矩阵复原；本地同时报告 ROUGE-L、EM 与 Token F1 的均值，明确评分差异 |

### 不同 FWT 不能混为一列

GEM 的前向迁移是在任务尚未学习时，与基座比较：

$$
\mathrm{FWT}_{\mathrm{GEM}}=
\frac{1}{T-1}\sum_{j=2}^{T}\left(R_{j-1,j}-R_{0,j}\right).
$$

SAPT §5.1.2 在任务学习后，与该任务从基座独立训练的成绩 $R_j^{\mathrm{ind}}$ 比较：

$$
\mathrm{FWT}_{\mathrm{SAPT}}=
\frac{1}{T}\sum_{j=1}^{T}\left(R_{j,j}-R_j^{\mathrm{ind}}\right).
$$

阶段记录给出了 $R_{j,j}$，却没有 $R_j^{\mathrm{ind}}$；基座 $R_{0,j}$ 不能替代独立训练分数。因此原论文这一项需要补做独立任务训练，而非从现有矩阵反推。

## 哪些可以完整复原

| 项目 | 状态 | 数据依据 / 限制 |
|---|---|---|
| 全任务阶段矩阵、初始 / 刚学完 / 最终分数 | 已复原 | 每阶段同一测试集 |
| AP / MFN / AA 的平均结构 | 已复原 | 评分分别注明 ROUGE-L、EM、Token F1 |
| MFT、MAA、Last、Average | 已复原 | 对角线、最终行、各阶段已学任务均值 |
| F.Ra、BWT、GEM FWT | 已复原 | 完整矩阵，并保留负值与固定分母 |
| 每任务遗忘、直接学习收益、学习前迁移 | 已复原 | 每列及其训练阶段 |
| 更新冲击、领域轨迹、未来任务变化、回答过程 | 已复原 | 阶段差分及原始回答 |
| 原论文独立训练 FWT、Oracle gap | N/A | 缺独立训练对照 |
| 原论文未见任务 / OOD 表格的原数据分数 | N/A | 没有评估其原始未见/OOD 数据；本地外部探针单列 |
| MIGU 分类 Macro-F1、HumanEval pass@k | N/A | 这里是生成任务记录，Token F1 不是分类 Macro-F1；没有代码单元测试输出 |
| 注意力 / 激活图、t-SNE | 不由评分矩阵直接提供 | 需从 checkpoint 和输入重算内部输出，不能用分数热图冒充注意力 |
| 多顺序 / 多种子均值、标准差 | N/A | 只有一个顺序、一个训练种子 |

## 这些报告支持什么结论

任务矩阵描述已覆盖能力随学习如何变化。外部固定探针观察未参与本次训练的学科题表现，但仍不覆盖所有知识，也不能直接证明某些内部知识被擦除。过程图可以定位退化发生在哪一步；要归因到训练算法或证明统计显著性，还需要独立对照与重复实验。

## 原始数据与实现

[Qwen 指标重算 CSV](../summary_cv/paper_metrics.csv)、[逐任务学习/保留](../summary_cv/task_learning_retention.csv)、[T5 指标重算](../summary/t5_paper_metrics.csv) 保留所有可计算结果；缺失对照项写为空值，在报告中显示 N/A。

运行 `summary_cv/process.py` 导出精确矩阵和过程数据，`summary_cv/reconstruct_metrics.py` 重算指标，`summary_cv/plot_process.py` 绘图。没有重新训练主线或改写原始评分。图表模板、方法配色及 300 DPI PNG / 矢量 PDF 的来源见各文件。
