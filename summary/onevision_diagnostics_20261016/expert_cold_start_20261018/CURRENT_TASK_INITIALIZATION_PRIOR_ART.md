# 不依赖跨域旧LoRA的初始化：已有工作与本研究边界

2026-10-18核验正式主会来源。本页是文献核验，不是新GPU结果，不新增基线、模块或队列。当前仍先完成[五格来源控制](../initialization_source_controls_20261018/v1/PROTOCOL.md)。

## 1. EVA：用当前训练输入构造初始方向

- **探究的问题：** 作者关注低秩初始化的更新方向与快速适应。
- **做的实验／方法：** 对训练激活做增量SVD，用右奇异向量初始化A，并按解释方差分配秩。
- **结果：** 作者报告多类任务上的收敛和参数效率改善。
- **证明了什么：** 论文给出其条件下的期望梯度信号结论；不能据此保证我们的Answer F1上升。
- **可借鉴的改进：** 从当前TRAIN构造方向，不必继承其他领域已训练的输出；这已是已有思路，不能作为我们的独创点。
- **本研究改进效果：** 未实施、未测。单／双更新、图文输入及极少TRAIN下效果未知。

正式标题为 **Parameter Efficient Fine-tuning via Explained Variance Adaptation**，Fabian Paischer等，**NeurIPS 2025主会**。旧标题或ICLR匿名投稿不能替代正式发表信息。来源：[正式摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/41d33bd41fd44bd9dba0e092047cf213-Abstract-Conference.html)、[正式论文，§3及Figure 1](https://proceedings.neurips.cc/paper_files/paper/2025/file/41d33bd41fd44bd9dba0e092047cf213-Paper-Conference.pdf)。

## 2. CorDA：从任务上下文分解预训练权重

- **探究的问题：** 作者关注适配器忽略目标任务上下文和预训练知识保持。
- **做的实验／方法：** 用样本激活协方差引导权重SVD；按目标选择分量，扣除初始低秩乘积以保持初始有效权重。
- **结果：** 作者报告数学、代码等任务的适配或知识保持收益。
- **证明了什么：** 说明上下文可参与初始化设计；没有证明本模型少数据持续学习的性能原因。
- **可借鉴的改进：** 区分初始函数与可学习方向；其补偿思想也属已有工作。
- **本研究改进效果：** 未实施、未测；协方差、求逆、SVD、权重补偿的临时存储及计算均需计入。

Yibo Yang等，**CorDA: Context-Oriented Decomposition Adaptation of Large Language Models for Task-Aware Parameter-Efficient Fine-tuning，NeurIPS 2024主会**。来源：[正式摘要](https://proceedings.neurips.cc/paper_files/paper/2024/hash/83f95bb0ac5046338ea2afe3390e9f4b-Abstract-Conference.html)、[正式论文，§3.2—3.4、Eq.4—5](https://proceedings.neurips.cc/paper_files/paper/2024/file/83f95bb0ac5046338ea2afe3390e9f4b-Paper-Conference.pdf)。

## 当前行动与判据

冷启动弱习得不推出跨域复用有益，也不推出仅靠换初始化就能获得大幅收益。现有五格先比较同域／跨域完整继承和A-only，后者清零B并要求真实初始函数门槛。若旧源方向无益，当前任务数据驱动的方向是可以继续考察的备选；不是已经选定的新方法。

初始化方向与任务相关性、源A实际变化幅度、首步A梯度可达性需分开解释。A-only仍携带源训练得到的方向，不能叫完全无先验。当前两个源／一个目标的结果也不能概括所有领域。

不因文献宣称收益而插队、改超参、增加专家或用TEST选择方案。以后若实施，只使用被冻结TRAIN，固定原目标训练预算，单列初始化额外前向／分解／内存；独立版本验证实际前向与梯度，再测原始Answer F1。现有GPU、基线和源代码不变。
