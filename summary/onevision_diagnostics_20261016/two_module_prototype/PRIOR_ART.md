# 设计前的相关工作初查（摘要级，不作创新性结论）

- [Continual LLaVA](https://arxiv.org/abs/2411.02564)：作者摘要已有低秩池选择与跨任务加权组合，不能把“复用旧低秩模块”本身写成首次提出。
- [A-GEM](https://arxiv.org/abs/1812.00420)：ICLR2019的历史约束参考；这里的模块B检查实际有限位移的TRAIN功能，仍需与既有约束方法比较，不能只换名字就宣称创新。
- [Bilinear Optimization Divergence](https://arxiv.org/abs/2609.23594)：2026预印本摘要已讨论LoRA因子约束、双线性有限步和优化后位移，明确存在先例；不是已核实的顶会论文。
- [LiteLoRA](https://arxiv.org/abs/2606.28117)：摘要已有学习复用/新适配器选择；页面注明ColorAI@ICML2026，不能据此称ICML主会。与用户希望自动选LoRA数量有重叠，须进一步核验全文/代码。
- [Hidden Failure Modes of Gradient Modification under Adam](https://arxiv.org/abs/2604.22407)：官方页面明确已撤稿，原因涉及实验设计和评价；不采用其性能数字或主要结论作为方法依据。我们保留自己独立可复核的CPU反例/真实梯度证据。

本次只核对官方摘要/状态，尚未完成全面公式、代码与新颖性评审；没有将这些条目混入原29篇正式顶会收录表。方法是否有效、是否有可发表创新都未证明。
