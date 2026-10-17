# 最终完整旧锚：领域替换与语言分布混杂

作者q_lang/answer_type只用于分层，原始Answer F1、问答与预测完全不改；没有翻译、答案映射或前缀剥离。

| 条件 | 英文QA/F1 | 中文QA/F1 | 中文题预测含汉字 | 中文题空输出 |
|---|---:|---:|---:|---:|
| repeat_anchor | 253/42.490 | 254/27.743 | 251/254 | 0 |
| within_medical | 253/46.047 | 254/22.336 | 251/254 | 0 |
| cross_short | 253/36.087 | 254/1.234 | 25/254 | 8 |

## 跨域−同域分层差异

- cross_short, en: F1差-9.960，95%图像区间[-15.659,-4.023]，QA加权贡献-4.970pp。
- cross_short, zh: F1差-21.102，95%图像区间[-26.680,-16.324]，QA加权贡献-10.572pp。
- 中文差−英文差：-11.142pp，图像区间[-19.159317060605552, -3.9037037037037017]；是分层异质性，不是语言因果效应。

## 原32题同队列轨迹：与全锚起点分开

| 条件/语言 | 题数 | S1/S3/S5 F1 | S1/S3/S5预测含汉字 |
|---|---:|---:|---:|
| repeat_anchor/en | 13 | 26.154/30.769/30.769 | 0/0/0 |
| repeat_anchor/zh | 19 | 14.737/18.421/25.439 | 9/15/19 |
| within_medical/en | 13 | 26.154/23.077/30.769 | 0/0/0 |
| within_medical/zh | 19 | 14.737/10.526/16.667 | 9/8/19 |
| cross_short/en | 13 | 26.154/41.538/33.846 | 0/0/0 |
| cross_short/zh | 19 | 14.737/13.158/2.632 | 9/7/2 |

32题属于完整队列的子集，其语言样本数和题目不同；本表不能补造完整507题S1。全部小样本配对区间见JSON，点估计下降不自动代表显著遗忘。


## 实际TRAIN语言构成

- repeat_anchor: 共同S1之后32目标QA曝光，问题含汉字16次；逐阶段作者语言/ID源见JSON。
- within_medical: 共同S1之后32目标QA曝光，问题含汉字17次；逐阶段作者语言/ID源见JSON。
- cross_short: 共同S1之后32目标QA曝光，问题含汉字0次；逐阶段作者语言/ID源见JSON。

没有汉字不等于回答一定错误；中文题上的英语回答也不能未经语义验证就称知识正确。完整共同S1仍须实测，不能把最终差距全说成已丢失的中文能力。

下一可证伪问题是在相同权重/原问题/同一F1下，既有question_language指令能否恢复输出及成绩；尚无该干预结果，不据此宣称知识充足。

[全部分层、原始输出分布、源SHA](T04_EVIDENCE.json)

- Post-hoc author-language/type strata; all predefined groups included, not a new main metric.
- Han/prefix flags are literal character statistics, not language-following accuracy or semantic correctness.
- No translation, option mapping, prefix stripping, alternative scoring, training/ID/prediction edits.
- QA-weighted contribution is arithmetic decomposition, not causal percentage explained by language.
- Language cohorts differ in questions/answers; image clustering does not establish translation-pair equivalence.
- TRAIN task/domain/language/content shifts are confounded; prompt intervention and common fullS1 still needed.
- Single anchor/seed/sequence; paired uncorrected image intervals condition on fixed models.
- Missing full commonS1 is not replaced by32QA; no full21454 or general method superiority claims.
