# 同一任务重复训练：保护是否需要、是否生效

六组共同首阶段参数/输出精确相同；原8条训练题重复5阶段，每阶段4更新，共40目标QA曝光。
此处没有跨领域切换；F1唯一主指标，NLL辅助。

| 方法 | 锚F1 S1/S3/S5 | 锚NLL S1/S3/S5 | 当前题阶段均值F1 |
|---|---:|---:|---:|
| SeqFT_LoRA | 19.375/23.438/27.604 | 5.8568/1.6987/1.3551 | 22.500 |
| ER_LoRA | 19.375/21.875/25.000 | 5.8568/1.6266/1.3260 | 23.958 |
| EWC_lambda1 | 19.375/21.875/27.083 | 5.8568/1.6973/1.3610 | 23.958 |
| EWC_lambda100 | 19.375/19.886/23.958 | 5.8568/1.7199/1.4487 | 22.917 |
| anchor_gradient_noop | 19.375/23.438/27.604 | 5.8568/1.6987/1.3551 | 22.500 |
| anchor_gradient_projection | 19.375/23.438/27.604 | 5.8568/1.6987/1.3551 | 22.500 |

## 机制与效果分开

- noop/projection全部5个参数状态、首批原始梯度和三次锚评测精确等于SeqFT；16次后续更新点积非负，投影未触发，不是一次有效投影的负结果。
- EWC两系数的stage2首批梯度精确等于SeqFT，但stage2末参数已不同；后续机制确实改变了轨迹，不能说EWC始终不生效。
- 共同首阶段的旧8TRAIN eval梯度范数为69.175216；这不是旧训练损失已经收敛的证据。
- 固定stage1 Fisher的二次距离沿SeqFT轨迹离开0，同时锚NLL下降；该中心距离不能直接解释成实际旧功能损失。正二次距离是已知代数，不是新发现。

## 相对SeqFT的最终配对差异

| 方法 | F1差/95%图像区间 | NLL差/95%图像区间 |
|---|---:|---:|
| ER_LoRA | -2.6042 [-10.4167,+2.2222] | -0.0290 [-0.1470,+0.0920] |
| EWC_lambda1 | -0.5208 [-1.6667,+0.0000] | +0.0059 [-0.0034,+0.0151] |
| EWC_lambda100 | -3.6458 [-10.7527,+0.0000] | +0.0936 [+0.0513,+0.1340] |
| anchor_gradient_noop | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |
| anchor_gradient_projection | +0.0000 [+0.0000,+0.0000] | +0.0000 [+0.0000,+0.0000] |

EWCλ100的NLL劣于SeqFT，但其F1差区间包含0；不宣布显著F1损害。ER的额外回放也不代表本控制内稳定优于SeqFT。

当前只证明：保护未触发与触发后未获主指标收益是不同情况；没有域切换也能出现约束与继续习得的张力。跨域是否进一步加剧，须等同域/跨域完整配对。

全部原始SHA、参数等价、曝光及区间见[JSON](T04_EVIDENCE.json)。

- Repeated same8QA is a no-domain-switch control, not the actual19-task1313QA CL setting.
- Fresh AdamW and boundary placement are held fixed, not independently ablated.
- The anchor has32QA; paired outputs record actual image clusters. Only SeqFT has full507/228 final.
- NLL is auxiliary, F1 remains the sole primary. F1 intervals touching0 do not establish significant harm.
- Positive displacement penalty is expected algebra, not new mathematics or proof forgetting occurs.
- Old TRAIN gradient is at the shared stage1 state; no own later EWC TRAIN gradients invented.
- Conditional paired image intervals uncorrected; one anchor/order/seed, no general method ranking.
- All six variants included; no TEST selection, code/threshold/data/checkpoint edits or new GPU jobs.
