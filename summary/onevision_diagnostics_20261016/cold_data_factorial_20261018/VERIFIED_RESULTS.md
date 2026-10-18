# 独特TRAIN量 × 初始化 × 更新数

有效轨迹完成1/18，每条均报告初始、1步、4步；缺失为待测。

| 原生任务 | 初始化 | 独特TRAIN | 每步曝光 | 原TEST | 初始F1 | 1步F1 | 4步F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| T02 | cold_base | 1 | 44 | 228 | 19.316 | 19.429 | 35.633 |

全部原native任务测试，不是完整五域主表。逐题原ID/金标/Answer F1复算、实际更新/原始梯度摘要和内存见[证据](VERIFIED_EVIDENCE.json)。

预定交互为：(温−冷)少样本 − (温−冷)全原TRAIN，正值表示该对照下小支持集的继承优势更大。所有方向、任务和区间均保留，不按结果选方法或超参；未齐格不能解释为0效应。

[协议与六栏](PROTOCOL.md)

- Single nested TRAIN hash selection, seed42; support amount changes content/answer coverage too.
- Equal exposure/effective batch within anchor, not equal tokens or equal compute across unique supports/tasks.
- Warm uses original nativeT01/history347TRAIN; not the older basis-budget warm control.
- Known3 anchors, no TEST-based case or budget selection; all18/both endpoints retained.
- Intervals are image-cluster/uncorrected and fixed-weight, not multiple-training-seed evidence.
- No fullAP/FWT/BWT or universal knowledge sufficiency from these isolated native tasks.
