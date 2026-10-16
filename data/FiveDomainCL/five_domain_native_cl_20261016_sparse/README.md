# 五领域19原生任务固定划分

固定seed42；沿用父实验已冻结1%样本，无重新抽样；训练前落盘并设只读。

| 阶段 | 领域 | 原生标签 | 全量训练 | 1%训练 | 完整测试 |
|---|---|---|---:|---:|---:|
| T01 | natural | all | 34602 | 347 | 5000 |
| T02 | medical | Modality | 1070 | 11 | 228 |
| T03 | medical | Position | 1695 | 17 | 375 |
| T04 | medical | Organ | 2543 | 26 | 507 |
| T05 | medical | Size | 650 | 6 | 124 |
| T06 | medical | Abnormality | 1410 | 14 | 311 |
| T07 | medical | Quantity | 437 | 4 | 89 |
| T08 | medical | Plane | 505 | 5 | 116 |
| T09 | medical | Shape | 83 | 1 | 12 |
| T10 | medical | Color | 268 | 3 | 65 |
| T11 | medical | KG | 1174 | 12 | 267 |
| T12 | geography | rural_urban | 572 | 6 | 100 |
| T13 | geography | presence | 17008 | 170 | 2955 |
| T14 | geography | count | 17089 | 171 | 2947 |
| T15 | geography | comp | 22554 | 226 | 4002 |
| T16 | finance | all | 8864 | 89 | 2233 |
| T17 | traffic | general | 4884 | 49 | 500 |
| T18 | traffic | regional | 10727 | 107 | 1123 |
| T19 | traffic | suggestion | 4884 | 49 | 500 |

各阶段内训练按SHA256(42:stage:original_QA_ID)升序；所有方法和主机共用manifest、QA与顺序哈希。
未新增QA，原记录恰好属于一个阶段。FinMME原图组80/20研究holdout不变，其他来源沿用原配置的训练/测试边界。
原生标签来自作者，标签训练顺序是研究固定顺序，不宣称作者官方课程。
完整测试在初始和最终各测一次，中间仅刚学/下一任务；19阶段共80316预测，缺失格为null。
