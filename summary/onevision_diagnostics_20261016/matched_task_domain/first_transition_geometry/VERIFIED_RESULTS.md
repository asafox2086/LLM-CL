# 严格共同起点：首次跨域梯度是否已经冲突

旧8TRAIN eval梯度在两个独立noop轨迹中逐元素相同，三个SeqFT起点A/B也精确相同；没有移植不同权重的梯度。

| 序列 | 新/旧梯度余弦 | 理想首步Adam旧loss Taylor项 | 真实四更新位移旧loss Taylor项 |
|---|---:|---:|---:|
| repeat_anchor | 0.775663 | -0.888430 | -3.054329 |
| within_medical | 0.512339 | -0.600578 | -2.217591 |
| cross_short | 0.159667 | -0.140492 | -0.740049 |

已完成3条，其中3条首次点积为正。正点积的配对不能用这次首批负梯度冲突解释；较弱正相关也不证明最终损害由梯度夹角造成，需后续更新与真实有限性能控制。

负Taylor项只说明该局部模型预计旧TRAIN loss下降；没有测量这个有限位移下的旧TRAIN loss，更不是F1或语义知识保留证明。

[原始源SHA/数值](EVIDENCE.json)

- Old reference eval-mode8QA versus new train-mode/dropout2QA; defined probes, not all-task gradients.
- Shared gradient used only at proven identical canonical weights, never later EWC/other states.
- Positive cosine only rejects a negative dot for this first pair; later conflict remains possible.
- Ideal Adam is known algebra; actual only four-update endpoint displacement saved here.
- Taylor terms are local linear predictions, not measured finite TRAIN/test NLL or F1 effects.
- Single Organ anchor/seed and frozen examples; not a domain-count law or novelty claim.
