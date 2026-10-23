# 直接相关先例与不能声明的创新

本轮定向查询用于判断框架是否只是已有模块拼装，不是泛读进展或质量证据。只据原论文；没有重做模型/旧统计。

| 原论文 | 与草案直接重合 | 仍不能从论文推出什么 |
| --- | --- | --- |
| [OSFT，ICLR2026正式摘要](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | 保留已有子空间并约束新任务更新，固定参数规模 | 本次OneVision收益的唯一来源；论文自适应机制不能代替本地固定25%实现事实 |
| [Basis Sharing v1](https://arxiv.org/html/2410.03765v1) | 跨层共享基和层专属系数；再用LoRA恢复已有实验 | 共享基+LoRA是本研究创新，或当前1213FIT足够恢复 |
| [Relaxed Recursive Transformers v1](https://arxiv.org/html/2410.20672v1) | 共享/循环层、层专属LoRA、从原权重初始化与恢复 | 简单层共享加适配器是新框架；其15B/60B token恢复不能作本1%多模态恢复保证 |
| [MobiLlama v1](https://arxiv.org/html/2402.16840v1) | 多层共享FFN，较小模型参数规模 | 从头预训练结果意味着能廉价转换当前多模态基座 |
| [InfLoRA v1](https://arxiv.org/html/2404.00228v1) | 用受约束适应子空间控制任务干扰 | 加正交小适配器是原创，或其ViT分类结论直接给出本多模态生成F1 |
| [Mehta等，AISTATS2021](https://proceedings.mlr.press/v130/mehta21a.html) | 因子字典、因子复用与持续任务容量分配 | 字典复用本身新颖；它的数据驱动容量扩展不等同于固定最终预算归并 |

草案尚未形成与这些先例明确分离的完整算法。特别是“共享基+层小残差+保护”不能仅通过换名宣称原创。潜在研究问题是压缩与持续适应共同占用有限前向容量时，如何有依据地重分配/归并并控制功能损失；目前只有问题，不能声称已经解决或文献中不存在。仍需针对最终具体机制继续查重。
