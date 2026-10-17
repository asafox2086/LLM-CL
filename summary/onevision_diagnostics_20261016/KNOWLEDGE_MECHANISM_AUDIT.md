# 知识匹配、参数与训练资源审计

本CPU审计不训练、不占GPU、不改知识池。correct仅表示作者原始事实、按TRAIN问题词汇相关选取，不保证覆盖当前QA。

| 数据 | 事实池 | QA | 同关系 | 同关系且固定tail匹配同一事实 | gold等于事实value | gold等于事实subject |
|---|---|---:|---:|---:|---:|---:|
| train | correct | 12 | 6 | 4 | 0 | 1 |
| train | unrelated | 12 | 0 | 0 | 0 | 0 |
| train | permuted | 12 | 6 | 4 | 0 | 1 |
| test | correct | 267 | 71 | 31 | 0 | 23 |
| test | unrelated | 267 | 29 | 0 | 0 | 0 |
| test | permuted | 267 | 71 | 31 | 0 | 23 |

这是字面统计，不是可回答率。是非题的gold可能是Yes/No，反向提问可能回答subject，功能同义词也会漏计。因此不能以gold-value零匹配宣布知识零覆盖；需结合作者原三元组、视觉实体绑定和事实编码核验解释。TRAIN的需求关系与事实池的关系分布不同，原16事实负结果不足以否定专业知识假设。

## 训练完成快照（不代表评测全部完成）

| 干预 | 可训练参数 | 更新/暴露 | 第1/4轮平均训练loss | 新增ΔW范数 | 与旧共享ΔW余弦 |
|---|---:|---:|---:|---:|---:|
| continue_rank8_correct | 540672 | 8/64 | 6.073/2.905 | 0.5457 | 0.0500 |
| continue_rank8_permuted | 540672 | 8/64 | 5.810/4.013 | 0.5549 | 0.0639 |
| continue_rank8_unrelated | 540672 | 8/64 | 5.437/2.840 | 0.5657 | 0.1862 |
| residual_rank1_correct | 67584 | 8/64 | 6.597/5.666 | 0.2571 | 0.0553 |
| residual_rank1_permuted | 67584 | 8/64 | 6.137/5.614 | 0.2508 | 0.0506 |
| residual_rank1_unrelated | 67584 | 8/64 | 5.785/5.122 | 0.2418 | 0.0715 |
| residual_rank2_correct | 135168 | 8/64 | 6.512/5.139 | 0.3436 | 0.0737 |
| residual_rank2_permuted | 135168 | 8/64 | 6.108/5.271 | 0.3429 | 0.0712 |
| residual_rank2_unrelated | 135168 | 8/64 | 5.747/4.691 | 0.3347 | 0.1003 |

实际ΔW=4BA，rank8继续训练使用相对旧共享模块的差分。各知识版本间的完整余弦矩阵、逐例训练前后CPU/GPU具体字节及窗口采样峰值见JSON。原知识runner仅记录梯度范数；不声称保存了每步原始梯度。新基底／预算实验会另存每步原始梯度。

## 完整KG任务按固定字面关系分组

三个分组相同且互斥：同关系及固定tail31题、仅同关系40题、关系不在池中196题。分组来自原三元组与冻结事实池，是事后诊断，不能称独立确认。相同关系/tail也不保证视觉head实体有对应事实。

| 已完成干预 | 固定分组 | QA | EM正确数 | Token F1 |
|---|---|---:|---:|---:|
| context_shared_correct | relation_absent | 196 | 8 | 6.545 |
| context_shared_correct | relation_only | 40 | 0 | 24.500 |
| context_shared_correct | same_relation_fixed_tail | 31 | 1 | 18.280 |
| context_shared_none | relation_absent | 196 | 0 | 6.694 |
| context_shared_none | relation_only | 40 | 0 | 15.458 |
| context_shared_none | same_relation_fixed_tail | 31 | 0 | 15.054 |
| context_shared_permuted | relation_absent | 196 | 8 | 5.798 |
| context_shared_permuted | relation_only | 40 | 0 | 23.333 |
| context_shared_permuted | same_relation_fixed_tail | 31 | 1 | 10.215 |
| context_shared_unrelated | relation_absent | 196 | 2 | 3.430 |
| context_shared_unrelated | relation_only | 40 | 1 | 10.833 |
| context_shared_unrelated | same_relation_fixed_tail | 31 | 0 | 1.613 |
| residual_rank1_correct | relation_absent | 196 | 0 | 5.419 |
| residual_rank1_correct | relation_only | 40 | 0 | 12.167 |
| residual_rank1_correct | same_relation_fixed_tail | 31 | 0 | 12.903 |
| residual_rank1_permuted | relation_absent | 196 | 0 | 5.551 |
| residual_rank1_permuted | relation_only | 40 | 0 | 12.167 |
| residual_rank1_permuted | same_relation_fixed_tail | 31 | 0 | 15.054 |
| residual_rank1_unrelated | relation_absent | 196 | 2 | 7.217 |
| residual_rank1_unrelated | relation_only | 40 | 0 | 12.792 |
| residual_rank1_unrelated | same_relation_fixed_tail | 31 | 0 | 12.903 |

## 原生答案类型的生成表现

OPEN/CLOSED来自作者原标注，不重写问题或筛除样本。语言分组与其他干预完整记录在JSON。

| 已完成干预 | 数据组 | 原答案类型 | QA | EM正确数 | Token F1 |
|---|---|---|---:|---:|---:|
| context_shared_correct | medical | CLOSED | 40 | 11 | 34.409 |
| context_shared_correct | medical | OPEN | 60 | 10 | 23.105 |
| context_shared_correct | medical_KG | CLOSED | 63 | 1 | 2.222 |
| context_shared_correct | medical_KG | OPEN | 204 | 8 | 13.184 |
| context_shared_none | medical | CLOSED | 40 | 6 | 23.381 |
| context_shared_none | medical | OPEN | 60 | 2 | 17.783 |
| context_shared_none | medical_KG | CLOSED | 63 | 0 | 11.746 |
| context_shared_none | medical_KG | OPEN | 204 | 0 | 8.123 |
| context_shared_permuted | medical | CLOSED | 40 | 11 | 34.409 |
| context_shared_permuted | medical | OPEN | 60 | 11 | 25.511 |
| context_shared_permuted | medical_KG | CLOSED | 63 | 1 | 2.222 |
| context_shared_permuted | medical_KG | OPEN | 204 | 8 | 11.012 |
| context_shared_unrelated | medical | CLOSED | 40 | 9 | 31.701 |
| context_shared_unrelated | medical | OPEN | 60 | 12 | 28.185 |
| context_shared_unrelated | medical_KG | CLOSED | 63 | 2 | 5.503 |
| context_shared_unrelated | medical_KG | OPEN | 204 | 1 | 3.966 |
| residual_rank1_correct | medical | CLOSED | 40 | 7 | 25.167 |
| residual_rank1_correct | medical | OPEN | 60 | 2 | 14.623 |
| residual_rank1_correct | medical_KG | CLOSED | 63 | 0 | 9.101 |
| residual_rank1_correct | medical_KG | OPEN | 204 | 0 | 6.743 |
| residual_rank1_permuted | medical | CLOSED | 40 | 6 | 21.952 |
| residual_rank1_permuted | medical | OPEN | 60 | 2 | 12.976 |
| residual_rank1_permuted | medical_KG | CLOSED | 63 | 0 | 10.159 |
| residual_rank1_permuted | medical_KG | OPEN | 204 | 0 | 6.869 |
| residual_rank1_unrelated | medical | CLOSED | 40 | 8 | 28.381 |
| residual_rank1_unrelated | medical | OPEN | 60 | 2 | 11.472 |
| residual_rank1_unrelated | medical_KG | CLOSED | 63 | 2 | 13.333 |
| residual_rank1_unrelated | medical_KG | OPEN | 204 | 0 | 7.285 |

## 训练占用前后（MiB，非模型文件大小）

| 干预 | CPU USS前→后 | GPU allocated前→后 | GPU reserved前→后 |
|---|---:|---:|---:|
| continue_rank8_correct | 2218.18→2218.18 | 1742.17→1742.35 | 7296.00→7296.00 |
| continue_rank8_permuted | 2218.19→2218.19 | 1742.17→1742.37 | 7296.00→7296.00 |
| continue_rank8_unrelated | 2218.18→2218.18 | 1742.17→1742.35 | 7296.00→7296.00 |
| residual_rank1_correct | 2212.91→2218.11 | 1742.42→1742.61 | 7296.00→7296.00 |
| residual_rank1_permuted | 2215.91→2218.15 | 1742.42→1742.63 | 7296.00→7296.00 |
| residual_rank1_unrelated | 2218.12→2218.13 | 1742.42→1742.61 | 7296.00→7296.00 |
| residual_rank2_correct | 2218.10→2218.10 | 1742.68→1742.87 | 7296.00→7296.00 |
| residual_rank2_permuted | 2218.18→2218.18 | 1742.68→1742.89 | 7296.00→7296.00 |
| residual_rank2_unrelated | 2218.15→2218.15 | 1742.68→1742.87 | 7296.00→7296.00 |

同一长进程已包含前面上下文推理缓存；具体前后值可读，但不能把CUDA缓存变化当专家参数增长。新基底／预算及容量实验分别使用新进程来降低跨配置污染。

限制：

- Post-hoc literal overlap, not semantic coverage or clinical validation; no replacement facts selected.
- Yes/no questions and questions asking for the subject need not have answers equal to fact values.
- Relation aliases/synonyms and visual binding are not resolved by exact matching.
- Checkpoint admission proves completed training, not completed evaluation; scores live in separate durable-case report.
- High deltaW cosine may reflect common factor initialization, language, templates or optimizer; not semantic redundancy proof.
- Epoch losses average two different minibatches; not per-fact retention or generalization.
- Only aggregate gradient norms were saved by this original knowledge runner, not all8 raw gradient vectors.
- Current knowledge pool remains immutable; additional causal knowledge coverage/encoding tests are still needed.
