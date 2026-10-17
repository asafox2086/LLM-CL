# 知识写入后的迁移：输出偏移与评分表面形式

只读已完成病例；逐题主评分重新复算一致，原数据／评分／参数不改。以下均为事后诊断，不是新临床准确率。

| 条件 | 步数 | 主EM正确/267 | 已训练value词表中的输出（均为原EM错） | 额外冠词／单复数待核查 | 中文题纯拉丁字母回答 |
| --- | ---: | ---: | ---: | ---: | ---: |
| rank1_correct | 0 | 0 | 0 | 3 | 28 |
| rank1_correct | 8 | 0 | 0 | 3 | 35 |
| rank1_correct | 64 | 10 | 7 | 11 | 48 |
| rank1_permuted | 0 | 0 | 0 | 3 | 28 |
| rank1_permuted | 8 | 0 | 0 | 4 | 36 |
| rank1_permuted | 64 | 3 | 0 | 10 | 47 |
| rank8_correct | 0 | 0 | 0 | 3 | 28 |
| rank8_correct | 8 | 3 | 0 | 11 | 52 |
| rank8_correct | 64 | 8 | 33 | 14 | 62 |
| rank8_permuted | 0 | 0 | 0 | 3 | 28 |
| rank8_permuted | 8 | 4 | 0 | 11 | 55 |
| rank8_permuted | 64 | 0 | 55 | 19 | 34 |

## 不能把严格0分直接解释为完全没有迁移

作者三元组同relation／tail且gold等于同一匹配事实subject的11题中，rank8正确知识64步原EM为0。逐题列出全部11例，避免只挑有利例子；它们与训练时subject→relation→value的问法不同，并依赖图像。

同一11题的负对照也必须同时报告：所有条件／时点原EM均为0/11。下面仅列额外词形待核查数，不能加到主正确数。

| 条件 | 0步 | 8步 | 64步 |
| --- | ---: | ---: | ---: |
| rank1_correct | 0 | 0 | 7 |
| rank1_permuted | 0 | 1 | 6 |
| rank8_correct | 0 | 7 | 9 |
| rank8_permuted | 0 | 7 | 10 |

rank8错配组64步也有10条词形待核查，正确组为9条。因此这组现象不能证明正确事实带来了特异性的关系迁移；以下11例仅展示正确组的具体输出，不作为正负对照结论。

| ID | 参考答案 | 原生成 | 仅冠词／简单s词形差待核查 |
| --- | --- | --- | --- |
| SLAKE:12278 | Lung | lungs | 是 |
| SLAKE:12293 | Heart | The heart | 是 |
| SLAKE:12308 | Heart | The heart | 是 |
| SLAKE:12323 | Lung | lungs | 是 |
| SLAKE:12398 | Lung | lungs | 是 |
| SLAKE:12413 | Heart | The Heart | 是 |
| SLAKE:12428 | Heart | The lungs | 否 |
| SLAKE:12443 | Lung | lungs | 是 |
| SLAKE:12458 | Lung | Stomach | 否 |
| SLAKE:12801 | Lung | The lungs | 是 |
| SLAKE:12811 | Kidney | The kidneys | 是 |

词形规则只识别TRAIN事实subject词表中的单英文器官名，允许一个前导冠词及简单附加s；不合并左右器官、列表、翻译或错误器官。此列不是人工医学判定，不能加到主正确数中，也未用于调参。

## 当前可以与不可以得出的结论

- 记住正向事实后，输出仍会集中到少数器官／系统词；KG问法、关系、语言与输出形式需要分开检查，不能只说知识不足或全部视觉失效。
- 原主EM很低时，同时存在实际错误与表面形式差异；不能从这11题原EM0推断模型完全不会对应关系，也不能把词形待核查当作泛化已解决。
- 词表输出频率变化是现象证据；没有训练信息隔离的干预前，不把它当成唯一原因。正确／错配完整结果和各原生分组见JSON。

全部组别、输出模式、逐题ID与SHA见[JSON](KNOWLEDGE_TRANSFER_AUDIT.json)。
