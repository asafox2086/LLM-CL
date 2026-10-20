# 借鉴什么，哪些不能直接套用

| 论文 | 可以借鉴的观察／方法 | 本轮采用范围与边界 |
| --- | --- | --- |
| [SVD-LLM，ICLR2025](https://mi-zhang.github.io/papers/2025_ICLR_SVD-LLM.pdf)，§3.1／3.2 | 权重截断与真实输入输出误差可能错位；按校准激活处理，并在压缩后补偿。 | 本轮只借鉴功能误差视角：同rank8比较权重方向与校准输出方向。没有实现原白化／闭式逐层恢复，不称原论文复现；原LLM结果不是0.5B多模态持续学习证据。 |
| [Dobi-SVD，ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/218ca0d92e6ed8f9db00621e103dc70c-Abstract-Conference.html) | 分析权重截断、激活截断与压缩秩选择，提示不能只看原权重谱。 | 本轮固定原基线rank8，不训练或搜索秩，不引入可微秩分配。 |
| [Compact Language Models via Pruning and Knowledge Distillation，NeurIPS2024](https://proceedings.nips.cc/paper_files/paper/2024/hash/4822991365c962105b1b95b1107d30e5-Abstract-Conference.html) | 从已有模型剪枝并蒸馏，减少重新预训练的成本。 | 论文训练数据比例以预训练语料计，不能等同我们的1313QA；暂不直接用其结果承诺小数据恢复或端侧优势。 |

当前新实验的明确假说是：**OSFT权重更新的高秩部分，在特定输入上的作用未必同样高维。** 这是待测假说。原K59的8.20%权重能量不是准确率覆盖率；功能空间有改善也不等于完整模型可删、推理会加速，必须后续完整质量与部署确认。

本轮校准／探针来自原FIT的不同图像组，teacher已用全部1313TRAIN学过这些题。只测近似子空间的输入外推，不称未见任务泛化。原四组共同DEV实验独立继续，TEST不选方案。
