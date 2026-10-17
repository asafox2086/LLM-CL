# 严格任务/领域对照：机制是否激活、激活后是否有效

完整轨迹 12/48。只读通过SHA和共同首阶段门槛的完整轨迹；0个完成表示待测。

| 锚/序列/机制 | 旧锚F1 S1→S5 | 当前任务F1 | 回放曝光 | 投影标记 | 非零penalty更新 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T04/repeat_anchor/SeqFT_LoRA | 19.375→27.604 | 22.500 | 0 | 0 | 0 |
| T04/repeat_anchor/ER_LoRA | 19.375→25.000 | 23.958 | 32 | 0 | 0 |
| T04/repeat_anchor/EWC_lambda1 | 19.375→27.083 | 23.958 | 0 | 0 | 15 |
| T04/repeat_anchor/EWC_lambda100 | 19.375→23.958 | 22.917 | 0 | 0 | 15 |
| T04/repeat_anchor/anchor_gradient_noop | 19.375→27.604 | 22.500 | 0 | 0 | 0 |
| T04/repeat_anchor/anchor_gradient_projection | 19.375→27.604 | 22.500 | 0 | 0 | 0 |
| T04/within_medical/SeqFT_LoRA | 19.375→22.396 | 28.589 | 0 | 0 | 0 |
| T04/within_medical/ER_LoRA | 19.375→23.958 | 27.247 | 32 | 0 | 0 |
| T04/within_medical/EWC_lambda1 | 19.375→22.396 | 28.589 | 0 | 0 | 15 |
| T04/within_medical/EWC_lambda100 | 19.375→23.958 | 28.021 | 0 | 0 | 15 |
| T04/within_medical/anchor_gradient_noop | 19.375→22.396 | 28.589 | 0 | 0 | 0 |
| T04/cross_short/SeqFT_LoRA | 19.375→15.312 | 48.958 | 0 | 0 | 0 |

逐阶段EWC最新项/全部项、首批原始新旧梯度、实际四步参数/BA位移、额外TRAIN信息和同SeqFT的旧/新F1差见[EVIDENCE.json](EVIDENCE.json)。

机制激活与最终作用分开：正则项首步0不是遗忘因果；投影标记不是成功保护；回放访问额外旧QA，不能称同总计算量。任何一组保旧结果都要与当前学习并列。

当前没有足够结果时不写因果结论。配对图像区间、序列比较与SeqFT完整锚测试见[原严格结果](../VERIFIED_RESULTS.md)。
