# 回答类型不能直接由现有显式关键词可靠分流

原输入构造只拼接任务instruction、context和question，`metadata.question_type`不传入模型。对冻结1%与10%金融TRAIN做一次新的有限检查后，**当前显式词典不支持直接区分单选与多选**。不能把原metadata当成部署时已知标签，也不能用这份词典直接给专家分流。

| TRAIN池 | 原单选／多选／数值数 | 命中明确单选提示 | 命中明确多选提示 | 两者均未命中 |
| --- | --- | ---: | ---: | ---: |
| 1% | 53／21／15 | 0 | 0 | 89 |
| 10% | 527／213／147 | 0 | 9 | 878 |

明确单选词典为“单选、单项选择、single choice、single-choice、choose one”；明确多选为“多选、多项选择、不定项、select all”。大小写忽略，英文multiple choice单列为歧义flag，不能自动代表多个正确选项。本次两个池该歧义flag均0；未在结果后扩词典。

10%的9个多选提示均来自题干，其中8题metadata为multiple_choice、1题为single_choice。这里不认定哪一方标注或语义错误，只说明该字面规则与原类型并非完全一致。共用instruction及全部context均未命中这些cue。原题型UNKNOWN为0，所有行保留。

## 对方法设计的影响

此前三强法的结果已表明：同一1%条件下，PAS能够获得部分多选原F1收益，而SeqFT与OSFT没有。[金融功能差异](FINANCE_STRONG_ACQUISITION_FINDINGS_20261027.md)。本次输入结果补充了约束：不能假设模型已经收到显式题型指令，或者通过原metadata就能免费切换单选／多选专家。

若继续研究小适配器如何承接这项功能，只能使用推理时真实可见的输入；原TRAIN类型可作为训练监督的候选，但不能在推理时提供答案或隐藏类型。另一种机制是否能避免先判题型，也须另行证明。本次没有选择具体模块、训练权重或新实验，更不修改已冻结提示、数据或评分。

## 不能推出什么

- 未命中有限关键词不等于题目语义不可区分，模型仍可能通过句义、选项内容或图像推断回答要求。
- 这没有证明错误来自输入歧义、题型混合干扰或容量不足；也不证明加入提示或辅助损失有效。
- 未建立数值提示词典，数值组只报告这些单／多选cue的出现情况，不能称三类识别器已评估。
- 构造的是chat template与tokenizer之前的文本，非历史逐token重放；原context均通过不触发检索的长度检查。

仅使用两池金融TRAIN，共976条；原答案字段未用于分析，0TEST文件、模型、生成或评分。源码先经独立审阅，原manifest、TRAIN与输入构造源均核SHA。[完整匿名结果](FINANCE_TRAIN_TYPE_CUES_20261027.json) SHA `39247cd9bfda546e25b99d2f085607f4781c5e30d2c8e1df7a9817a8f1b6ab3f`；[源码](../../analysis/finance_train_type_cues_20261027.py) SHA `0f8bd2f9160dcda1f7227fd2092554297521fbc13a6889be0a8f4a8b76894bd9`；[独立审阅](FINANCE_TRAIN_TYPE_CUES_REVIEW_20261027.md)。原QA不复制进公开结果。
