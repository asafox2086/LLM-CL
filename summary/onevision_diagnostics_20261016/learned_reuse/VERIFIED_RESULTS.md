# 可学习向量空间与LoRA优先复用

完整训练证明：2/2；完整21454题确认：0/2。待测不记零分。

| 版本 | 范围 | 领域 | 生成题数 | Answer F1 |
|---|---|---|---:|---:|
| fixed_cosine_K3 | train | natural | 10 | 56.999 |
| fixed_cosine_K3 | train | medical | 73 | 12.871 |
| fixed_cosine_K3 | train | geography | 32 | 38.958 |
| fixed_cosine_K3 | train | finance | 10 | 0.000 |
| fixed_cosine_K3 | train | traffic | 30 | 30.358 |
| learned_cosine_K3 | train | natural | 10 | 58.571 |
| learned_cosine_K3 | train | medical | 73 | 15.841 |
| learned_cosine_K3 | train | geography | 32 | 46.875 |
| learned_cosine_K3 | train | finance | 10 | 20.000 |
| learned_cosine_K3 | train | traffic | 30 | 31.421 |

| 版本 | 最终LoRA数 | 新增阶段 | 投影实际改变阶段 |
|---|---:|---|---|
| fixed_cosine_K3 | 3 | [1, 2, 3] | [] |
| learned_cosine_K3 | 3 | [1, 2, 3] | [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19] |

完整分配决策、训练NLL前后、实际LR/梯度、逐题得失、内存窗口与资源见[JSON](VERIFIED_EVIDENCE.json)。尚无GPU完成证明时，不宣称路由有效或性能提升。

- Assigned-expert labels and cosine novelty are not proof of task mastery.
- One expert produces no discriminative metric update; report actual gradients/changes.
- Kmax3 is a fixed budget, never a proved optimal expert count.
- Diagnostic500 NLL /155 generated are not full21454; report separate cells.
- Final-only evaluation cannot establish full FWT/BWT or within-expert forgetting.
- No domain/answer/TEST-based allocation; all valid variants get full confirmation.
- nativeAnyRes research is separate from formal common-image V2.
- No memory superiority without matched measurement including router/history/temporary/extra compute.
