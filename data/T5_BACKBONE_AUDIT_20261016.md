# T5与VL-T5基座核查（2026-10-16）

用户最新方向：先实验诊断、分析问题，再提出方法。目前只核查基座，不新增虚构算法，不停止或切换正在运行的Qwen2-VL对照。

## 核实事实

| 名称 | 语言模型规模 | 原生输入 | 与本实验关系 |
|---|---:|---|---|
| T5-Small | 60M | 文本 | 小，但不能直接把原图输入纯T5 |
| T5-Base | 220M | 文本 | VL-T5作者代码所用的语言骨干 |
| T5-Large | 770M | 文本 | 比Base大；模型名称本身不代表视觉能力 |
| T5-3B／11B | 3B／11B | 文本 | T5家族也有大版本，不能统称轻量 |
| VL-T5 | 作者代码基于T5-Base，另有视觉投影等参数 | 图像区域特征＋文本 | 需要视觉预训练权重与特征提取流程；不能用纯T5权重冒充 |

参数规模来自[Google官方列表](https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints)。[T5-Base模型卡](https://huggingface.co/google-t5/t5-base)确认220M；[VL-T5官方代码](https://github.com/j-min/VL-T5)API指定backbone=t5-base。这里没有把语言模型参数量写成完整多模态系统参数量。

FP16纯权重推算（十进制GB）：Small约0.12GB，Base约0.44GB，Large约1.54GB；只计算参数×2字节，不含视觉系统、优化器、梯度、激活和框架缓存。没有实测就不承诺加速倍数、显存占用或训练时长。

## 哪些论文在用

- [VL-T5：Unifying Vision-and-Language Tasks via Text Generation，ICML2021](https://proceedings.mlr.press/v139/cho21a.html)：原始多模态生成框架，支持VQA等，不局限于分类。
- [VQACL，CVPR2023作者代码](https://github.com/zhangxi1997/VQACL)：VL-T5路径明确存在，说明传统持续VQA工作确有该基座；它超出本次2024–2026文献收集的时间范围，作为基座背景参考。
- [CL-MoE，CVPR2025，表1／第5节](https://arxiv.org/html/2503.00413)：同一论文对照VL-T5和LLaVA-7B。自身CL-MoE使用LLaVA-7B；一张表里多行VL-T5是同基座不同方法，不能当成多篇新论文都采用T5。
- [Continual LLaVA，2411.02564，第3.3节](https://arxiv.org/html/2411.02564)：作者明确选择Vicuna作为语言模型，并非T5。

本仓库当前方法源码manifest中SMoLoRA、PASs-MoE、Hyper-LLaVA列为LLaVA/Vicuna；PaRSP列为Llama/Qwen，ALER列为Qwen。该本地记录支持“当前候选并非主要围绕T5”，但不是对所有持续学习论文的系统统计。

## 是否适合五域：待验证的推断

官方[特征提取说明](https://github.com/j-min/VL-T5/blob/main/feature_extraction/README.md)使用bottom-up区域特征，每图36框、HDF5存储。我们的五域没有现成等价的作者特征缓存；适配和提取成本必须计入，不能用COCO缓存替代医学／遥感／图表的原图。若选择新的视觉编码器，要标记为架构适配，不能称作者原始VL-T5。

| 数据集 | 必须测的问题；以下均未声称完成 |
|---|---|
| TextVQA | 原图文字是否被视觉特征保留；错误是否集中在读字 |
| SLAKE | 原生中英问题能否理解，医学影像特征能否支持回答；不得删中文／知识题 |
| RSVQA-LR | 俯视图、计数与空间关系能否被自然图像区域特征表示 |
| FinMME | 图表细节、选项、多选和数值是否可学习；不输入verified_caption等答案相关元数据 |
| CODA-LM | 长答案生成、区域红框与整体场景的视觉信息是否可用 |

这些是依据架构和数据差异提出的检查点，不是已证实的域内性能结论。若初始和单任务训练后成绩都低，首先考察骨干／视觉输入是否能学会，不能直接解释为持续学习遗忘。

## 后续诊断顺序

1. 验证官方多模态预训练权重的获取、完整性及实际参数数目；核对区域特征提取依赖，隔离环境，避免修改现有基线环境。
2. 在预留新GPU1测真实图文前向／反向、CPU/GPU占用、输入与目标token数、生成速度；记录特征提取的一次性成本和训练过程成本。
3. 使用已经冻结的训练ID做各领域单任务可学习性诊断；提前固定诊断评测ID，不按方法结果挑例子。正式比较继续完整原始测试，不改问答构成。
4. 结合Qwen顺序实验区分：模型本身学不会、跨任务遗忘、视觉未被利用、额外参数／内存增长。再确定自研方法要解决的问题。

当前没有下载大权重、没有启动VL-T5训练，也没有把VL-T5追加到正式十方法结果表。
