# 含图像的持续学习实验报告

这轮观察同一个 Qwen2-VL-2B 模型先学习语言，再学习医学语言、自然图像和医学图像后，能学到多少、保留多少。四方法的完成状态和成绩见下表。

## 先看真实示例

以下每任务取固定测试顺序的第一条，展示输入与回答的节选；示例选择不依赖得分。模型回答来自 SeqLoRA 学完全部任务的状态。

| 领域 / 任务 | 输入节选 | 参考答案节选 | 模型最终回答节选 |
|---|---|---|---|
| 新闻摘要（XSum） | Cross, who has admitted to the murders, said he hoped to "die a martyr" after the verdict … | White supremacist Frazier Glenn Cross has been found guilty of murdering three people at t… | A white supremacist who shot dead three people at a Jewish community center in Kansas City… |
| 阅读理解（Quoref） | Passage: It was probably William the Conqueror who gave the city and its castle to Bishop … | Rochester. | William the Conqueror |
| 关系抽取 | radio can be used as the opposite of tv | radio Antonym tv | radio Antonym tv |
| 对话续写（PersonaChat） | Personality: I try to limit how much I eat. I whine a lot. I've a golden retriever puppy. … | I like being at home with the puppy. I eat a lot and should really limit that. | I do not have a job. |
| 因果推理（GLUCOSE） | story: All the towels in the house are wet. There are only four of us living here. There a… | The towels are wet &gt;Causes/Enables&gt; The towels dry out | All the towels are wet &gt;Causes/Enables&gt; Someone is extra dry |
| 医患对话转临床记录（MTS-Dialog） | Doctor: I spoke with Poison Control regarding the possible ingestion of the liquid. They l… | I discussed the case with Poison Control and apparently this is actually relatively small … | No further action. |
| 影像 Findings 转 Impression（IU-Xray） | The cardiomediastinal silhouette is normal in size and contour. Hyperexpanded lungs withou… | Negative for acute abnormality. | No acute cardiopulmonary abnormality. |
| 自然图像问答（VQAv2） | Answer the question briefly using the image. Is it cold outside? | no | yes |
| 医学图像问答（VQA-RAD） | Answer the question briefly using the image. Are the lungs normal appearing? | No | No |

图像示例使用真实图片：[自然图像与医学图像、各答案类别以及四方法完整回答](examples.md)。该文档覆盖 9 个任务和 5 个图像答案类别，逐例对比基线、刚学完和最终回答。

## 大体效果

本轮 SeqLoRA 的最终平均分最高；SAPT-LoRA 的平均遗忘幅度最小。 应同时看学习能力与保留能力。结果来自单个种子、固定顺序及短训练预算。

| 方法 | 状态 | AP ↑ | 遗忘率 ↓ | BWT ↑ | FWT ↑ |
|---|---|---:|---:|---:|---:|
| SeqLoRA | completed | 48.799 | 3.890 | -3.864 | -3.397 |
| MIGU-LoRA | completed | 43.964 | 4.871 | -4.586 | -2.559 |
| O-LoRA | completed | 47.239 | 0.736 | -0.498 | -2.050 |
| SAPT-LoRA | completed | 42.247 | 0.084 | -0.030 | -1.206 |

## 指标怎样理解

所有任务分数均为 0–100；多参考答案时，每项取与参考比较的最大值，再对固定测试样本平均。

| 指标 | 含义 | 阅读方式 |
|---|---|---|
| ROUGE-L ↑ | 生成文字与参考答案的最长公共子序列 F1 | 词面重合越多，分数越高；不等同于医学或事实正确率 |
| EM ↑ | 小写、去标点/英文冠词并合并空白后，是否完全匹配参考答案 | 单题为 0 或 100；任务均值是归一化精确匹配比例 |
| Token F1 ↑ | 答案与参考的词覆盖率与精确率的调和平均 | 容许部分匹配，不要求词序一致 |
| AP ↑ | 全部学完后的各任务 ROUGE-L 等权平均 | 看最终整体水平 |
| F.Rate ↓ | 旧任务在最终阶段前的历史最好分，减去最终分，再平均 | 看平均遗忘幅度；单位为分数点，允许负值 |
| BWT ↑ | 旧任务最终分与刚学完时分数之差的平均 | 正值表示后续学习帮助旧任务，负值表示下降 |
| FWT ↑ | 每个任务尚未训练时，相对初始基座的分数变化，再平均 | 看前序学习是否帮助未来任务；排除第一任务 |

最后一个任务没有后续学习，不参与 F.Rate/BWT。FWT 只比较训练该任务之前的表现。更完整的定义与手算示例见 [语言实验指标说明](../summary/README.md#指标定义)。

VQAv2 的 EM 是多参考答案归一化精确匹配，不能当作官方 VQA 共识准确率。跨任务平均混合了摘要、对话、短回答等形式，应同时查看下面的领域和任务表现。

## 各领域效果

| 方法 | 最终领域 | ROUGE-L | EM (%) | Token F1 (%) |
|---|---|---:|---:|---:|
| SeqLoRA | 通用语言 | 48.524 | 23.000 | 48.451 |
| SeqLoRA | 医学语言 | 28.286 | 3.750 | 27.981 |
| SeqLoRA | 自然图像 | 86.667 | 84.500 | 85.617 |
| SeqLoRA | 医学图像 | 53.336 | 49.500 | 53.010 |
| MIGU-LoRA | 通用语言 | 42.788 | 14.200 | 42.875 |
| MIGU-LoRA | 医学语言 | 21.799 | 1.500 | 21.095 |
| MIGU-LoRA | 自然图像 | 86.417 | 84.500 | 85.367 |
| MIGU-LoRA | 医学图像 | 51.719 | 48.000 | 51.393 |
| O-LoRA | 通用语言 | 47.802 | 21.800 | 47.423 |
| O-LoRA | 医学语言 | 23.522 | 2.500 | 23.029 |
| O-LoRA | 自然图像 | 88.417 | 86.000 | 86.867 |
| O-LoRA | 医学图像 | 50.679 | 46.000 | 50.047 |
| SAPT-LoRA | 通用语言 | 43.278 | 21.200 | 42.487 |
| SAPT-LoRA | 医学语言 | 14.002 | 0.000 | 16.401 |
| SAPT-LoRA | 自然图像 | 86.414 | 83.500 | 85.119 |
| SAPT-LoRA | 医学图像 | 49.416 | 45.500 | 48.714 |

自然图像问答最终 EM 为 83.5%–86.0%；医学图像问答最终 EM 为 45.5%–49.5%。 不同任务的答案长度和参考形式不同，应逐任务分析。

## 各任务效果

| 方法 | 最终任务 | ROUGE-L | EM (%) | Token F1 (%) |
|---|---|---:|---:|---:|
| SeqLoRA | 新闻摘要（XSum） | 23.263 | 0.000 | 23.630 |
| SeqLoRA | 阅读理解（Quoref） | 58.182 | 50.500 | 58.130 |
| SeqLoRA | 关系抽取 | 83.977 | 62.500 | 87.936 |
| SeqLoRA | 对话续写（PersonaChat） | 16.004 | 0.000 | 13.759 |
| SeqLoRA | 因果推理（GLUCOSE） | 61.193 | 2.000 | 58.801 |
| SeqLoRA | 医患对话转临床记录（MTS-Dialog） | 16.133 | 1.000 | 16.021 |
| SeqLoRA | 影像 Findings 转 Impression（IU-Xray） | 40.438 | 6.500 | 39.941 |
| SeqLoRA | 自然图像问答（VQAv2） | 86.667 | 84.500 | 85.617 |
| SeqLoRA | 医学图像问答（VQA-RAD） | 53.336 | 49.500 | 53.010 |
| MIGU-LoRA | 新闻摘要（XSum） | 24.211 | 0.000 | 24.486 |
| MIGU-LoRA | 阅读理解（Quoref） | 51.717 | 44.000 | 51.994 |
| MIGU-LoRA | 关系抽取 | 69.268 | 25.500 | 73.062 |
| MIGU-LoRA | 对话续写（PersonaChat） | 14.380 | 0.000 | 11.718 |
| MIGU-LoRA | 因果推理（GLUCOSE） | 54.365 | 1.500 | 53.114 |
| MIGU-LoRA | 医患对话转临床记录（MTS-Dialog） | 13.974 | 0.500 | 13.071 |
| MIGU-LoRA | 影像 Findings 转 Impression（IU-Xray） | 29.625 | 2.500 | 29.118 |
| MIGU-LoRA | 自然图像问答（VQAv2） | 86.417 | 84.500 | 85.367 |
| MIGU-LoRA | 医学图像问答（VQA-RAD） | 51.719 | 48.000 | 51.393 |
| O-LoRA | 新闻摘要（XSum） | 24.844 | 0.000 | 26.424 |
| O-LoRA | 阅读理解（Quoref） | 56.185 | 50.000 | 56.251 |
| O-LoRA | 关系抽取 | 84.737 | 57.500 | 85.057 |
| O-LoRA | 对话续写（PersonaChat） | 16.081 | 0.000 | 13.586 |
| O-LoRA | 因果推理（GLUCOSE） | 57.165 | 1.500 | 55.796 |
| O-LoRA | 医患对话转临床记录（MTS-Dialog） | 17.329 | 1.500 | 16.936 |
| O-LoRA | 影像 Findings 转 Impression（IU-Xray） | 29.715 | 3.500 | 29.123 |
| O-LoRA | 自然图像问答（VQAv2） | 88.417 | 86.000 | 86.867 |
| O-LoRA | 医学图像问答（VQA-RAD） | 50.679 | 46.000 | 50.047 |
| SAPT-LoRA | 新闻摘要（XSum） | 24.759 | 0.000 | 27.200 |
| SAPT-LoRA | 阅读理解（Quoref） | 53.493 | 45.500 | 53.535 |
| SAPT-LoRA | 关系抽取 | 86.417 | 60.500 | 86.333 |
| SAPT-LoRA | 对话续写（PersonaChat） | 14.560 | 0.000 | 11.776 |
| SAPT-LoRA | 因果推理（GLUCOSE） | 37.163 | 0.000 | 33.590 |
| SAPT-LoRA | 医患对话转临床记录（MTS-Dialog） | 18.076 | 0.000 | 20.758 |
| SAPT-LoRA | 影像 Findings 转 Impression（IU-Xray） | 9.929 | 0.000 | 12.044 |
| SAPT-LoRA | 自然图像问答（VQAv2） | 86.414 | 83.500 | 85.119 |
| SAPT-LoRA | 医学图像问答（VQA-RAD） | 49.416 | 45.500 | 48.714 |

## 报告范围与明细

学习顺序：XSum → Quoref → 关系抽取 → PersonaChat → GLUCOSE → MTS-Dialog → IU-Xray → VQAv2 → VQA-RAD。每任务 1,000 条训练、100 条验证、200 条测试，训练 1 轮；图像与语言使用同一预算。

VQAv2 使用官方验证集的本地子集；VQA-RAD 使用图像/病例分组划分。这轮是先导实验，不能直接与论文的完整官方测试成绩比较，也不能直接与 T5 的不同任务和测试配额比较。

[全部阶段](stage_scores.csv) · [领域分数](domain_scores.csv) · [答案类别与医学部位](category_scores.csv) · [完整实例](examples.json)

已核对 72,000 条测试回答，汇总分数与逐题评分一致，记录见 [verification.json](verification.json)。

## 实现流程与复现命令

原始运行：[本次运行](../exp/CV_result/qwen2vl2b_language5_medical2_vision2/20261007T152511Z)。完整参数和方法适配见 [实验协议](../rules/004_qwen2vl_cv.md)。

初始基座及每个学习阶段结束后，均复测全部九个任务，形成 10×9 矩阵。模型基座和视觉编码器冻结，训练 LoRA；图像的冻结特征缓存复用。每个训练更新保存优化器、学习率调度器、随机状态和样本游标，阶段 checkpoint 另含方法记忆。

各方法目录内的 `config.json`、`data_manifest.json`、`environment.json` 和 `source/` 保留参数及实现指纹；`predictions/` 保留回答，`checkpoints/` 保留训练日志和恢复状态。

在仓库根目录刷新此报告：

```bash
.vision-env/bin/python summary_cv/collect.py
```

启动与恢复命令、实现文件及日志位置见 [实验说明](../exp/README.md#实现流程与复现)。
