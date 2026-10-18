# 冷启动解释的已有工作：初始功能与更新几何需要区分

2026-10-18，核验正式主会原始来源。以下是通用PEFT初始化相关工作，**不是新增的持续学习主表基线，也不是本模型的实验结果**。不以其分数预填我们的收益，不据此改当前GPU源码或排队条件。

| 正式论文 | 作者研究的问题与结论 | 对本研究的作用与边界 |
| --- | --- | --- |
| [LoRA-GA: Low-Rank Adaptation with Gradient Approximation，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/62c4718cc334f6a0a62fb81c4a2095a1-Abstract-Conference.html)，Shaowen Wang、Linxi Yu、Jian Li | 用初始化使低秩首步更新更接近全参数更新；Section 3.1以抵消初始低秩乘积保持初始有效权重不变。 | 初始化/收敛问题已有研究；不能将零B对应A首梯度为0当作新理论，也不能把继承收益全归旧知识。作者并未证明我们的1%多领域图文持续学习效果。 |
| [The Impact of Initialization on LoRA Finetuning Dynamics，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d4387c37b3b06e55f86eccdb8cd1f829-Abstract-Conference.html)，Soufiane Hayou、Nikhil Ghosh、Bin Yu | 随机A/零B与零A/随机B都从相同预训练函数开始，但训练动力学、适用学习率及实验表现不同；其分析和实验支持前者在文中配置下更好。 | 同初始函数不等于同优化过程；“某一因子首步无梯度”不能推出这种初始化普遍更差。论文没有证明本模型应该翻转A/B或改变学习率。 |
| [Beyond Zero Initialization: Investigating the Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics，ICML 2025](https://proceedings.mlr.press/v267/li25bm.html)，Shiwei Li等 | 分析非零A/B初始化；文中非零初始化对不理想、尤其偏小学习率表现更稳健。 | 非零初始化不必来自旧任务知识；潜在收益还可能来自优化条件。不能把作者条件下的稳健性推广为我们的单步预算、任意尺度或所有领域必然提升。 |

## 对当前证据的准确表述

我们已在真实T02单步保存梯度和实际参数中核验：冷启动A没有更新，完整继承后A/B均更新；同19阶段/1313 TRAIN且500条调用不变的对照，在155题诊断AP提高4pp。它支持**本冻结训练流中的初始化干预贡献**。目前没有分开先前习得的有效函数、因子基底和非零B带来的优化条件；更没有证明所有现有方法都因同一个机制变差。完整21454及数据量交互仍待。

研究动机可以写为：“少数据/少更新会使新增专家的初始状态和早期可学习方向尤其值得检查；只比较最终容量或trainable参数总数不足以解释实际习得。”不能写为“我们首次发现零初始化LoRA收敛有问题”“首步A不更新证明LoRA无用”或“当前改善已经证明知识充分”。

## 尚未实施的分离思路，不是新增模块或已准入作业

如后续需要进一步分离初始函数与优化几何，可借鉴LoRA-GA的补偿形式：

\[
\Delta W_t=s(B_tA_t-B_0A_0).
\]

在数学上的t=0增量为0，而非零A0/B0允许两个因子获得梯度；它提示可以构造“同初始有效函数、不同因子初始化”的控制。**公式相等不代替BF16实际前向、梯度、初始函数精确门槛。** 同一函数下仍可能改变切空间/先验，不能凭此宣称纯粹分离了全部“知识”因素。

补偿因子、共享dropout/计算路径、额外前反向、CPU/GPU存储和可恢复checkpoint均有成本，不能说只是改初始化而没有额外开销。若实际实现，需要独立可追溯版本和TRAIN预定设置；不能用TEST选尺度、步数或父专家。本页仅记录已有工作和未解决的解释，不启动新GPU、增专家、插队或延迟当前质量调用/既定控制与完整确认。

[当前冷启动证明链](COLD_START_PROOF.md) · [实际首步梯度](first_factor_gradient/RESEARCH_FINDINGS.md) · [19阶段干预与诊断结果](../warm_start_20261018/first19_outcome/RESEARCH_FINDINGS.md)。
