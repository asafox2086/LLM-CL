# 可学习向量空间与LoRA优先复用

完整训练证明：0/2；完整21454题确认：0/2。待测不记零分。

| 版本 | 范围 | 领域 | 生成题数 | Answer F1 |
|---|---|---|---:|---:|

| 版本 | 最终LoRA数 | 新增阶段 | 投影实际改变阶段 |
|---|---:|---|---|

完整分配决策、训练NLL前后、实际LR/梯度、逐题得失、内存窗口与资源见[JSON](VERIFIED_EVIDENCE.json)。尚无GPU完成证明时，不宣称路由有效或性能提升。

- Assigned-expert labels and cosine novelty are not proof of task mastery.
- One expert produces no discriminative metric update; report actual gradients/changes.
- Kmax3 is a fixed budget, never a proved optimal expert count.
- Diagnostic500 NLL /155 generated are not full21454; report separate cells.
- Final-only evaluation cannot establish full FWT/BWT or within-expert forgetting.
- No domain/answer/TEST-based allocation; all valid variants get full confirmation.
- nativeAnyRes research is separate from formal common-image V2.
- No memory superiority without matched measurement including router/history/temporary/extra compute.
