# 遥感损失权重对照：完成病例证据

仅纳入完成证明和源码／文件SHA校验通过的病例；完整三组未完成前不下全对照结论。

| 条件 | 更新 | split | 类别 | EM正确/总数 | F1 | ROUGE-L | NLL |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| uniform | 1 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.3338 |
| uniform | 1 | TRAIN | urban | 2/4 | 50.00 | 50.00 | 0.3940 |
| uniform | 1 | TRAIN | rural | 2/2 | 100.00 | 100.00 | 0.2134 |
| uniform | 1 | TEST | all | 76/100 | 76.00 | 76.00 | 0.2958 |
| uniform | 1 | TEST | urban | 23/44 | 52.27 | 52.27 | 0.3905 |
| uniform | 1 | TEST | rural | 53/56 | 94.64 | 94.64 | 0.2215 |
| uniform | 4 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.2397 |
| uniform | 4 | TRAIN | urban | 4/4 | 100.00 | 100.00 | 0.0766 |
| uniform | 4 | TRAIN | rural | 0/2 | 0.00 | 0.00 | 0.5659 |
| uniform | 4 | TEST | all | 44/100 | 44.00 | 44.00 | 0.3737 |
| uniform | 4 | TEST | urban | 44/44 | 100.00 | 100.00 | 0.0796 |
| uniform | 4 | TEST | rural | 0/56 | 0.00 | 0.00 | 0.6047 |
| class_balanced | 1 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.3582 |
| class_balanced | 1 | TRAIN | urban | 2/4 | 50.00 | 50.00 | 0.4357 |
| class_balanced | 1 | TRAIN | rural | 2/2 | 100.00 | 100.00 | 0.2031 |
| class_balanced | 1 | TEST | all | 75/100 | 75.00 | 75.00 | 0.2973 |
| class_balanced | 1 | TEST | urban | 20/44 | 45.45 | 45.45 | 0.4190 |
| class_balanced | 1 | TEST | rural | 55/56 | 98.21 | 98.21 | 0.2017 |
| class_balanced | 4 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.2213 |
| class_balanced | 4 | TRAIN | urban | 4/4 | 100.00 | 100.00 | 0.1787 |
| class_balanced | 4 | TRAIN | rural | 0/2 | 0.00 | 0.00 | 0.3065 |
| class_balanced | 4 | TEST | all | 50/100 | 50.00 | 50.00 | 0.2649 |
| class_balanced | 4 | TEST | urban | 44/44 | 100.00 | 100.00 | 0.1778 |
| class_balanced | 4 | TEST | rural | 6/56 | 10.71 | 10.71 | 0.3333 |
| urban_high_negative | 1 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.3313 |
| urban_high_negative | 1 | TRAIN | urban | 2/4 | 50.00 | 50.00 | 0.3903 |
| urban_high_negative | 1 | TRAIN | rural | 2/2 | 100.00 | 100.00 | 0.2135 |
| urban_high_negative | 1 | TEST | all | 74/100 | 74.00 | 74.00 | 0.2968 |
| urban_high_negative | 1 | TEST | urban | 23/44 | 52.27 | 52.27 | 0.3799 |
| urban_high_negative | 1 | TEST | rural | 51/56 | 91.07 | 91.07 | 0.2316 |
| urban_high_negative | 4 | TRAIN | all | 4/6 | 66.67 | 66.67 | 0.2741 |
| urban_high_negative | 4 | TRAIN | urban | 4/4 | 100.00 | 100.00 | 0.0584 |
| urban_high_negative | 4 | TRAIN | rural | 0/2 | 0.00 | 0.00 | 0.7056 |
| urban_high_negative | 4 | TEST | all | 44/100 | 44.00 | 44.00 | 0.4405 |
| urban_high_negative | 4 | TEST | urban | 44/44 | 100.00 | 100.00 | 0.0574 |
| urban_high_negative | 4 | TEST | rural | 0/56 | 0.00 | 0.00 | 0.7415 |

逐题预测分布、配对差异和区间、参数位移、原始梯度审计及训练内存字节见[JSON](RURAL_WEIGHT_EVIDENCE.json)。不能把改损失权重的收益解释为对所有领域／方法的唯一病因，也不据此提出新CL方法。
