# 任务／领域控制：逐题转移、额外计算与梯度干预

24 completed default-backend controls,23 original anchor QA; not full-test CL metrics or strict causal proof

同一23题的S1→S5拆分：早期正确后错误、早期错误后正确分别记录。净分上升不等于没有旧题退化；分母很小，不将两三题的保持率泛化。

| 序列 | 机制 | 始终正确 | 正确→错误 | 错误→正确 | 始终错误 | ΔEM [95%] | ΔNLL [95%] |
|---|---|---:|---:|---:|---:|---:|---:|
| repeat_anchor | SeqFT_LoRA | 2 | 0 | 6 | 15 | +26.087 [+9.091,+45.455] | -4.481 [-5.382,-3.591] |
| repeat_anchor | ER_LoRA | 2 | 0 | 4 | 17 | +17.391 [+4.348,+33.333] | -4.356 [-5.172,-3.515] |
| repeat_anchor | EWC_lambda1 | 2 | 0 | 6 | 15 | +26.087 [+9.091,+45.455] | -4.444 [-5.341,-3.541] |
| repeat_anchor | EWC_lambda100 | 2 | 0 | 7 | 14 | +30.435 [+9.091,+52.000] | -4.332 [-5.179,-3.469] |
| repeat_anchor | anchor_gradient_noop | 2 | 0 | 6 | 15 | +26.087 [+9.091,+45.455] | -4.418 [-5.332,-3.523] |
| repeat_anchor | anchor_gradient_projection | 2 | 0 | 6 | 15 | +26.087 [+9.091,+45.455] | -4.438 [-5.322,-3.567] |
| within_medical | SeqFT_LoRA | 2 | 0 | 7 | 14 | +30.435 [+9.091,+52.000] | -4.367 [-5.183,-3.527] |
| within_medical | ER_LoRA | 2 | 0 | 7 | 14 | +30.435 [+9.091,+52.000] | -4.254 [-5.013,-3.464] |
| within_medical | EWC_lambda1 | 2 | 0 | 8 | 13 | +34.783 [+13.636,+56.000] | -4.281 [-5.077,-3.468] |
| within_medical | EWC_lambda100 | 2 | 0 | 8 | 13 | +34.783 [+13.636,+56.522] | -4.200 [-4.994,-3.370] |
| within_medical | anchor_gradient_noop | 2 | 0 | 7 | 14 | +30.435 [+9.091,+52.000] | -4.384 [-5.216,-3.539] |
| within_medical | anchor_gradient_projection | 2 | 0 | 7 | 14 | +30.435 [+9.091,+52.000] | -4.319 [-5.111,-3.506] |
| cross_short | SeqFT_LoRA | 1 | 1 | 3 | 18 | +8.696 [-9.091,+25.000] | -0.473 [-1.004,+0.061] |
| cross_short | ER_LoRA | 0 | 2 | 7 | 14 | +21.739 [-4.545,+46.154] | -4.113 [-4.918,-3.275] |
| cross_short | EWC_lambda1 | 1 | 1 | 5 | 16 | +17.391 [+0.000,+36.364] | -0.689 [-1.270,-0.110] |
| cross_short | EWC_lambda100 | 1 | 1 | 3 | 18 | +8.696 [-9.091,+25.000] | -0.534 [-1.066,-0.004] |
| cross_short | anchor_gradient_noop | 0 | 2 | 4 | 17 | +8.696 [-13.043,+29.167] | -0.513 [-1.016,+0.032] |
| cross_short | anchor_gradient_projection | 0 | 2 | 5 | 16 | +13.043 [-9.091,+34.783] | -1.833 [-2.456,-1.159] |
| cross_five_domains | SeqFT_LoRA | 0 | 2 | 4 | 17 | +8.696 [-13.043,+29.167] | -0.530 [-1.013,-0.031] |
| cross_five_domains | ER_LoRA | 0 | 2 | 7 | 14 | +21.739 [-4.545,+46.154] | -4.066 [-4.951,-3.154] |
| cross_five_domains | EWC_lambda1 | 0 | 2 | 4 | 17 | +8.696 [-13.043,+29.167] | -0.535 [-0.997,-0.063] |
| cross_five_domains | EWC_lambda100 | 0 | 2 | 3 | 18 | +4.348 [-13.636,+21.739] | -0.391 [-0.722,-0.052] |
| cross_five_domains | anchor_gradient_noop | 0 | 2 | 3 | 18 | +4.348 [-13.636,+21.739] | -0.388 [-0.855,+0.099] |
| cross_five_domains | anchor_gradient_projection | 0 | 2 | 3 | 18 | +4.348 [-13.636,+21.739] | -1.339 [-1.780,-0.914] |

## 额外数据读取与计算

每个方法均为40次当前QA暴露、20更新。ER额外32次回放SGD；EWC额外20次Fisher QA；投影及no-op均额外128次锚QA梯度计算。后两者的读取次数相同，可以检查投影操作，但默认数值与原型小样本混杂仍存在。40次重复锚只含8个不同QA，不能写成40个新样本。

| 序列 | 干预−控制 | ΔEM变化差 [95%] | ΔNLL变化差 [95%] |
|---|---|---:|---:|
| repeat_anchor | ER_LoRA−SeqFT_LoRA | -8.696 [-22.727,+0.000] | +0.126 [-0.043,+0.281] |
| repeat_anchor | anchor_gradient_projection−anchor_gradient_noop | +0.000 [+0.000,+0.000] | -0.021 [-0.090,+0.047] |
| repeat_anchor | EWC_lambda100−EWC_lambda1 | +4.348 [+0.000,+12.000] | +0.112 [+0.020,+0.206] |
| within_medical | ER_LoRA−SeqFT_LoRA | +0.000 [+0.000,+0.000] | +0.112 [-0.001,+0.243] |
| within_medical | anchor_gradient_projection−anchor_gradient_noop | +0.000 [+0.000,+0.000] | +0.065 [-0.002,+0.139] |
| within_medical | EWC_lambda100−EWC_lambda1 | +0.000 [-13.043,+13.043] | +0.082 [+0.010,+0.148] |
| cross_short | ER_LoRA−SeqFT_LoRA | +13.043 [-4.545,+30.435] | -3.640 [-4.384,-2.937] |
| cross_short | anchor_gradient_projection−anchor_gradient_noop | +4.348 [+0.000,+13.636] | -1.320 [-1.580,-1.054] |
| cross_short | EWC_lambda100−EWC_lambda1 | -8.696 [-21.739,+0.000] | +0.154 [+0.057,+0.247] |
| cross_five_domains | ER_LoRA−SeqFT_LoRA | +13.043 [+0.000,+26.923] | -3.536 [-4.311,-2.816] |
| cross_five_domains | anchor_gradient_projection−anchor_gradient_noop | +0.000 [+0.000,+0.000] | -0.951 [-1.173,-0.718] |
| cross_five_domains | EWC_lambda100−EWC_lambda1 | -4.348 [-13.636,+0.000] | +0.144 [-0.012,+0.298] |

变化差=(干预S5−S1)−(控制S5−S1)。它缓解起始分数不一致的展示问题，不能修复起始权重不一致或非线性训练路径差异。

## 固定任务数下的序列对比

以下为SeqFT；其余机制的全部同类对比见JSON。跨短序列不含交通长答案，但内容／视觉输入仍不同，不能将差异纯归为领域数量。

| 序列−参照 | ΔEM变化差 [95%] | ΔNLL变化差 [95%] |
|---|---:|---:|
| within_medical−repeat_anchor | +4.348 [+0.000,+12.000] | +0.115 [-0.011,+0.250] |
| cross_short−within_medical | -21.739 [-39.130,-4.545] | +3.893 [+3.176,+4.647] |
| cross_five_domains−within_medical | -21.739 [-39.130,-8.333] | +3.837 [+3.166,+4.548] |
| cross_five_domains−cross_short | +0.000 [-13.043,+13.043] | -0.057 [-0.247,+0.166] |

## 原始TRAIN梯度的核验

仅阶段2–5第一个minibatch有原始当前／参考梯度：CPU重算点积、投影及实际优化器记录的梯度范数，所有32个检查通过。JSON保留各阶段原余弦与重建投影余弦、负点积／投影频率、目标token数与额外QA读取。重建向量不是历史保存的投影后向量，负余弦本身也不是最终遗忘的因果证明。

补充优化器限制：真实CPU Adam的二维反例确认，梯度点积为0时其参数更新仍可能使旧梯度一阶项为正。因此原投影控制只能称Euclidean原始梯度投影，不能声称Adam实际更新必保旧损失。实际四步参数位移的一阶项与首步Adam公式重建均记录，前者不是实测loss变化，后者不是历史保存的单步位移。

限制：

- Stage1 weights differ across cases despite same seed; subtracting S1 does NOT remove this numerical confound.
- One weakly acquired Organ anchor, single order/seed; few initial correct answers make retention fractions unstable.
- Task/domain interventions also change content, target lengths and visual inputs.
- ER has extra SGD replay; Fisher/projection have different extra compute. Only projection/noop share reference-read counts.
- Raw gradient audit covers first minibatch at stages2-5 only; reconstructed projection is not a saved postprojection vector.
- Read-only CPU norms use FP64 after flat FP32 reduction mismatch; saved training gradients and original model reproducibility gates unchanged.
- Reconstructed first Adam step is an idealized fresh-optimizer formula; actual four-step displacement linear term is Taylor evidence, not measured loss or a saved first-step update.
- Bootstrap uses image-path groups, exploratory95% intervals, no multiple-testing correction.
- Correct-to-wrong transition is reference-EM loss, not a proof of semantic/clinical knowledge deletion.
- Sparse anchor endpoints cannot provide full benchmark AP/FWT/BWT or historical-peak ForgetRate.
