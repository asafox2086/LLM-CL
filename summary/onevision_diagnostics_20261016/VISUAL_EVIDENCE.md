# 基座视觉消融：原图、无图、错图

固定同一500个诊断QA，正NLL优势表示原图更好。输入审计：无图视觉token为0；原图／错图均有视觉token；错图引用路径与原图不同且符合原确定性规则，逐题ID一致。

| 领域 | 题数 | 原图NLL | 无图NLL | 错图NLL | 原图胜过错图题数 | 原图对错图NLL优势及95%区间 |
|---|---:|---:|---:|---:|---:|---|
| natural | 100 | 5.853 | 8.296 | 6.905 | 72 | +1.053 [+0.682, +1.429] |
| medical | 100 | 9.595 | 8.517 | 9.470 | 46 | -0.125 [-0.272, +0.033] |
| geography | 100 | 3.396 | 3.860 | 3.307 | 43 | -0.089 [-0.385, +0.202] |
| finance | 100 | 3.955 | 5.740 | 4.007 | 52 | +0.053 [-0.241, +0.371] |
| traffic | 100 | 2.303 | 2.588 | 2.382 | 85 | +0.079 [+0.064, +0.094] |

医学和遥感域没有显示稳定的正确图像NLL优势，而自然和交通表现不同。这使视觉条件利用、语言／答案行为成为需要与知识缺口一起检验的解释；不能仅据无图损失更低就断言视觉编码错误、数据无效或专业知识不需要。

19个原生任务分组、原图对无图区间及原始文件哈希见JSON；小组样本量明确保留。

限制：

- Untrained base only; no claim trained adapters have the same behavior.
- Teacher-forced NLL on500 diagnostic QA; not accuracy or full21454 test.
- Wrong image is deterministic from same-domain diagnostic pool, not necessarily contradictory content.
- Wrong image audit checks distinct asset paths, not pixel-level or semantic uniqueness.
- Removing an image changes prompt length; not a perfectly isolated vision-encoder intervention.
- Image-cluster bootstrap1000 seed42; native subgroups can have very few samples.
- Raw question/answer/image files remain unchanged; altered input views are evaluation ablations only.
