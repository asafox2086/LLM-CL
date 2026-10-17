# 知识干预：格式敏感性与事实写入检查

17/17原知识干预、事实回忆输出均通过完成证明与SHA验证；CPU只读，无新训练。固定单一The answer is前缀规则只作事后辅助，不修改主分数或数据。所有上下文对照使用原实验统一both提示，不混用其他prompt的基座结果。

## 医学100题：原分数上升未必是知识收益

| 干预 | 原EM正确 | 固定前缀辅助正确 | 前缀输出数 | NLL |
| --- | ---: | ---: | ---: | ---: |
| context_base_none | 0/100 | 24/100 | 92 | 11.169 |
| context_base_correct | 9/100 | 11/100 | 28 | 2.983 |
| context_base_unrelated | 7/100 | 8/100 | 20 | 2.923 |
| context_base_permuted | 10/100 | 12/100 | 37 | 3.076 |
| context_shared_none | 8/100 | 15/100 | 28 | 4.665 |
| context_shared_correct | 21/100 | 21/100 | 0 | 2.376 |
| context_shared_unrelated | 21/100 | 21/100 | 0 | 2.376 |
| context_shared_permuted | 22/100 | 22/100 | 0 | 2.452 |
| residual_rank1_correct | 9/100 | 17/100 | 27 | 3.925 |
| residual_rank1_unrelated | 10/100 | 16/100 | 24 | 3.852 |
| residual_rank1_permuted | 8/100 | 15/100 | 30 | 3.982 |
| residual_rank2_correct | 9/100 | 15/100 | 21 | 3.585 |
| residual_rank2_unrelated | 10/100 | 16/100 | 15 | 3.495 |
| residual_rank2_permuted | 9/100 | 15/100 | 21 | 3.620 |
| continue_rank8_correct | 16/100 | 20/100 | 13 | 2.762 |
| continue_rank8_unrelated | 31/100 | 31/100 | 0 | 2.226 |
| continue_rank8_permuted | 23/100 | 26/100 | 5 | 2.598 |

基座none→correct原EM为0→9，但去固定前缀的辅助正确数为24→11。共享LoRA none→correct原EM8→21，辅助15→21；低相关同为21，错配为22。正确事实带来的特异性生成收益仍未被这些对照确证。不能把原EM差全部当知识收益，也不能反过来认定知识一定无用。

## 原生KG267题：同样保留负对照与格式限制

| 干预 | 原EM正确 | 固定前缀辅助正确 | 前缀输出数 |
| --- | ---: | ---: | ---: |
| context_base_none | 0/267 | 17/267 | 201 |
| context_base_correct | 0/267 | 14/267 | 150 |
| context_base_unrelated | 1/267 | 1/267 | 101 |
| context_base_permuted | 0/267 | 12/267 | 198 |
| context_shared_none | 0/267 | 7/267 | 52 |
| context_shared_correct | 9/267 | 10/267 | 16 |
| context_shared_unrelated | 3/267 | 5/267 | 26 |
| context_shared_permuted | 9/267 | 10/267 | 22 |
| residual_rank1_correct | 0/267 | 6/267 | 61 |
| residual_rank1_unrelated | 2/267 | 11/267 | 57 |
| residual_rank1_permuted | 0/267 | 7/267 | 65 |
| residual_rank2_correct | 2/267 | 12/267 | 75 |
| residual_rank2_unrelated | 2/267 | 8/267 | 49 |
| residual_rank2_permuted | 2/267 | 13/267 | 78 |
| continue_rank8_correct | 3/267 | 10/267 | 90 |
| continue_rank8_unrelated | 22/267 | 22/267 | 0 |
| continue_rank8_permuted | 4/267 | 12/267 | 61 |

## 正确事实与负对照的配对区间

| 正确事实干预 | 对照 | 组 | 原EM差95%区间（pp） | 辅助EM差95%区间（pp） |
| --- | --- | --- | --- | --- |
| context_base_correct | context_base_none | medical | +9.00 [+4.00, +14.68] | -13.00 [-21.15, -5.05] |
| context_base_correct | context_base_none | medical_KG | +0.00 [+0.00, +0.00] | -1.12 [-4.62, +2.38] |
| context_base_correct | context_base_unrelated | medical | +2.00 [-1.92, +6.12] | +3.00 [-2.02, +8.26] |
| context_base_correct | context_base_unrelated | medical_KG | -0.37 [-1.20, +0.00] | +4.87 [+2.33, +7.63] |
| context_base_correct | context_base_permuted | medical | -1.00 [-3.09, +0.00] | -1.00 [-4.30, +2.11] |
| context_base_correct | context_base_permuted | medical_KG | +0.00 [+0.00, +0.00] | +0.75 [-1.06, +2.60] |
| context_shared_correct | context_shared_none | medical | +13.00 [+6.25, +20.00] | +6.00 [-2.06, +14.14] |
| context_shared_correct | context_shared_none | medical_KG | +3.37 [+1.49, +5.59] | +1.12 [-1.75, +3.96] |
| context_shared_correct | context_shared_unrelated | medical | +0.00 [-5.83, +5.61] | +0.00 [-5.83, +5.61] |
| context_shared_correct | context_shared_unrelated | medical_KG | +2.25 [+0.00, +4.62] | +1.87 [-0.39, +4.33] |
| context_shared_correct | context_shared_permuted | medical | -1.00 [-4.63, +2.04] | -1.00 [-4.63, +2.04] |
| context_shared_correct | context_shared_permuted | medical_KG | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| residual_rank1_correct | residual_rank1_unrelated | medical | -1.00 [-3.09, +0.00] | +1.00 [-2.08, +4.51] |
| residual_rank1_correct | residual_rank1_unrelated | medical_KG | -0.75 [-1.90, +0.00] | -1.87 [-3.56, -0.40] |
| residual_rank1_correct | residual_rank1_permuted | medical | +1.00 [+0.00, +3.16] | +2.00 [+0.00, +5.00] |
| residual_rank1_correct | residual_rank1_permuted | medical_KG | +0.00 [+0.00, +0.00] | -0.37 [-1.20, +0.00] |
| residual_rank2_correct | residual_rank2_unrelated | medical | -1.00 [-3.16, +0.00] | -1.00 [-3.16, +0.00] |
| residual_rank2_correct | residual_rank2_unrelated | medical_KG | +0.00 [+0.00, +0.00] | +1.50 [+0.36, +3.04] |
| residual_rank2_correct | residual_rank2_permuted | medical | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| residual_rank2_correct | residual_rank2_permuted | medical_KG | +0.00 [+0.00, +0.00] | -0.37 [-1.75, +0.79] |
| continue_rank8_correct | continue_rank8_unrelated | medical | -15.00 [-22.92, -7.37] | -11.00 [-19.39, -2.97] |
| continue_rank8_correct | continue_rank8_unrelated | medical_KG | -7.12 [-10.42, -3.97] | -4.49 [-7.39, -1.58] |
| continue_rank8_correct | continue_rank8_permuted | medical | -7.00 [-12.62, -2.11] | -6.00 [-11.96, -0.97] |
| continue_rank8_correct | continue_rank8_permuted | medical_KG | -0.37 [-1.67, +0.79] | -0.75 [-2.55, +1.10] |

## 事实原模板：去前缀仍未得到严格回忆成功

| 干预 | 参照事实 | 原EM | 辅助EM | NLL |
| --- | --- | ---: | ---: | ---: |
| context_base_correct | relevant_true | 1/16 | 1/16 | 2.945 |
| context_base_none | relevant_true | 0/16 | 0/16 | 8.545 |
| context_base_permuted | relevant_true | 0/16 | 1/16 | 3.151 |
| context_base_unrelated | relevant_true | 0/16 | 0/16 | 4.553 |
| context_shared_correct | relevant_true | 4/16 | 4/16 | 2.255 |
| context_shared_none | relevant_true | 0/16 | 0/16 | 6.626 |
| context_shared_permuted | relevant_true | 0/16 | 0/16 | 2.432 |
| context_shared_unrelated | relevant_true | 0/16 | 0/16 | 4.167 |
| continue_rank8_correct | relevant_true | 0/16 | 0/16 | 2.553 |
| continue_rank8_permuted | own_permuted | 0/16 | 0/16 | 3.720 |
| continue_rank8_permuted | relevant_true | 0/16 | 0/16 | 2.920 |
| continue_rank8_unrelated | own_unrelated | 0/16 | 0/16 | 2.645 |
| continue_rank8_unrelated | relevant_true | 0/16 | 0/16 | 3.552 |
| residual_rank1_correct | relevant_true | 0/16 | 0/16 | 5.429 |
| residual_rank1_permuted | own_permuted | 0/16 | 0/16 | 5.471 |
| residual_rank1_permuted | relevant_true | 0/16 | 0/16 | 5.497 |
| residual_rank1_unrelated | own_unrelated | 0/16 | 0/16 | 4.945 |
| residual_rank1_unrelated | relevant_true | 0/16 | 0/16 | 5.710 |
| residual_rank2_correct | relevant_true | 0/16 | 0/16 | 4.894 |
| residual_rank2_permuted | own_permuted | 0/16 | 0/16 | 5.124 |
| residual_rank2_permuted | relevant_true | 0/16 | 0/16 | 5.072 |
| residual_rank2_unrelated | own_unrelated | 0/16 | 0/16 | 4.470 |
| residual_rank2_unrelated | relevant_true | 0/16 | 0/16 | 5.532 |

rank1／rank2／rank8正确事实LoRA对原16训练模板，原评分及仅去前缀评分均0/16；对应同事实固定改写也均0/16。未证明写入成功不等于证明知识不存在：别名、中文长答案的部分语义回忆及训练预算仍是解释。至少“只差这个英文前缀”不能解释其严格回忆失败。上下文正确事实shared原模板为4/16、改写为2/16，说明本测试也非必然全零。

下一步优先核验训练干预本身是否足量且能拟合原事实，再讨论对医学原生任务的特异收益；不修改旧事实池，不按测试答案挑事实／rank／预算，不先设计新方法。

原医学100题和KG267题有12题交集；原模板与改写共享16事实，均不得视作独立扩样。完整逐条件统计、配对区间及文件SHA见[JSON](KNOWLEDGE_FORMAT_AUDIT.json)。
