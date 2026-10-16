# 固定提示词对照结果

未训练基座，不增加训练数据或参数；全部四个预定提示均报告。

| 提示 | 领域 | NLL题数 | NLL↓ | 生成题数 | EM↑ | Token F1↑ | ROUGE-L↑ |
|---|---|---:|---:|---:|---:|---:|---:|
| original | natural | 100 | 5.853 | 10 | 0.000 | 42.487 | 39.879 |
| original | medical | 100 | 9.595 | 100 | 0.000 | 13.204 | 13.132 |
| original | geography | 100 | 3.396 | 10 | 0.000 | 8.000 | 8.000 |
| original | finance | 100 | 3.955 | 10 | 0.000 | 0.000 | 0.000 |
| original | traffic | 100 | 2.303 | 10 | 0.000 | 29.614 | 16.838 |
| answer_only | natural | 100 | 2.498 | 10 | 50.000 | 62.333 | 62.333 |
| answer_only | medical | 100 | 9.456 | 100 | 0.000 | 12.749 | 12.683 |
| answer_only | geography | 100 | 5.380 | 10 | 0.000 | 12.000 | 12.000 |
| answer_only | finance | 100 | 4.035 | 10 | 0.000 | 0.000 | 0.000 |
| answer_only | traffic | 100 | 2.309 | 10 | 0.000 | 30.378 | 17.602 |
| question_language | natural | 100 | 4.830 | 10 | 0.000 | 48.222 | 48.222 |
| question_language | medical | 100 | 7.197 | 100 | 0.000 | 16.113 | 16.113 |
| question_language | geography | 100 | 1.463 | 10 | 20.000 | 20.000 | 20.000 |
| question_language | finance | 100 | 4.165 | 10 | 0.000 | 0.000 | 0.000 |
| question_language | traffic | 100 | 2.302 | 10 | 0.000 | 28.731 | 17.471 |
| both | natural | 100 | 6.845 | 10 | 0.000 | 38.987 | 38.987 |
| both | medical | 100 | 11.169 | 100 | 0.000 | 12.445 | 12.435 |
| both | geography | 100 | 11.912 | 10 | 0.000 | 12.000 | 12.000 |
| both | finance | 100 | 5.027 | 10 | 0.000 | 0.000 | 0.000 |
| both | traffic | 100 | 2.315 | 10 | 0.000 | 28.740 | 17.228 |

医学100题每个提示EM均为0。提示能改变损失和部分重叠指标，但这组固定提示没有使模型稳定输出匹配参考答案的回答；不能据此证明专业知识不足。

医学逐语言分组和配对图像簇bootstrap区间见JSON。分组只按原问题是否含中文字符，不是自动评价生成语言是否正确。自然等领域仅10个生成探针，避免把50%等小样本变化冒充全量准确率。

限制：

- 100 medical generation QA; other domains only10, all diagnostic subsets.
- Image-cluster bootstrap1000 resamples seed42 describes this subset, not full-test performance.
- No prompt or capacity selected by these test scores; knowledge protocol retains predeclared both prompt.
- This tests fixed English instruction additions, not all possible language/format interventions.
