# EWC在极少更新设置下何时开始起作用

CPU只读实际8个EWC条件、32个阶段首批梯度和对应检查点／Fisher，全部sidecar SHA验证。另核验4序列×6机制×3锚时点的72个已完成输出。原运行和评分不变。

## 可直接验证的机制性质

EWC项为λ/2 Σ F(θ−θ_old)²。在刚进入新任务时，最新旧任务的中心恰等于当前参数，因此其梯度为0。实际32/32个阶段首批均验证这一点；进入第二阶段仅有这一项，因此8/8条件的整个EWC梯度均为0。

用实际保存的相同起点、同一个首批目标梯度，在严格确定性CPU模式独立构造fresh AdamW：λ=0／1／100的第一步参数逐元素相同（8/8条件）。这是条件固定的梯度／优化器反事实，不是原GPU轨迹完全复现或准确率因果证据。首次未设确定性的CPU断言失败已保留；当时未记录具体失败条件，后续检查未稳定复现，不能据此认定确定性开关就是其成因。严格相等标准未放宽，原GPU源文件未改。

**这意味着：只有一次更新的新任务，不能指望以最新参数为中心的二次EWC项在该第一步阻止偏离。** 但更多旧任务的中心可能已不同，进入第三阶段后它们可立即产生非零梯度；其他正则化方法未必具有同一性质。不能概括为EWC或LoRA毫无作用。

## 同时核查：旧任务是否已到驻点

另读取noop／projection八个控制各自阶段1权重后、阶段2首批更新前保存的原8条Organ TRAIN均值梯度。全部非零：范数约63.96–72.50。相同点若构造以它自身为中心的二次项，梯度仍为0。因此这些有限预算检查点的真实旧任务一阶项未消失，中心二次项本身并不匹配它。

这是各控制自身权重的性质，不把其梯度当作另一个EWC条件的梯度，也不把任意非零梯度当作“新任务一定损害旧任务”。新／旧梯度可能相容；实际准确率因果影响未测定。原4×6条件未达到相同权重／确定性复现要求，不能用跨条件小差异作因果证明。

## 后续阶段并非全部失活

| 序列 | EWC λ | 阶段 | 首批EWC／目标梯度范数 | 最新任务项范数 |
| --- | ---: | ---: | ---: | ---: |
| repeat_anchor | 1 | 2 | 0.000000 | 0.0 |
| repeat_anchor | 1 | 3 | 0.001032 | 0.0 |
| repeat_anchor | 1 | 4 | 0.008067 | 0.0 |
| repeat_anchor | 1 | 5 | 0.028591 | 0.0 |
| repeat_anchor | 100 | 2 | 0.000000 | 0.0 |
| repeat_anchor | 100 | 3 | 0.089840 | 0.0 |
| repeat_anchor | 100 | 4 | 0.572262 | 0.0 |
| repeat_anchor | 100 | 5 | 0.889553 | 0.0 |
| within_medical | 1 | 2 | 0.000000 | 0.0 |
| within_medical | 1 | 3 | 0.002648 | 0.0 |
| within_medical | 1 | 4 | 0.001585 | 0.0 |
| within_medical | 1 | 5 | 0.009404 | 0.0 |
| within_medical | 100 | 2 | 0.000000 | 0.0 |
| within_medical | 100 | 3 | 0.246116 | 0.0 |
| within_medical | 100 | 4 | 0.102912 | 0.0 |
| within_medical | 100 | 5 | 0.415954 | 0.0 |
| cross_short | 1 | 2 | 0.000000 | 0.0 |
| cross_short | 1 | 3 | 0.001973 | 0.0 |
| cross_short | 1 | 4 | 0.019436 | 0.0 |
| cross_short | 1 | 5 | 0.003862 | 0.0 |
| cross_short | 100 | 2 | 0.000000 | 0.0 |
| cross_short | 100 | 3 | 0.234520 | 0.0 |
| cross_short | 100 | 4 | 0.961939 | 0.0 |
| cross_short | 100 | 5 | 0.286871 | 0.0 |
| cross_five_domains | 1 | 2 | 0.000000 | 0.0 |
| cross_five_domains | 1 | 3 | 0.002131 | 0.0 |
| cross_five_domains | 1 | 4 | 0.001454 | 0.0 |
| cross_five_domains | 1 | 5 | 0.236689 | 0.0 |
| cross_five_domains | 100 | 2 | 0.000000 | 0.0 |
| cross_five_domains | 100 | 3 | 0.210097 | 0.0 |
| cross_five_domains | 100 | 4 | 0.055183 | 0.0 |
| cross_five_domains | 100 | 5 | 6.338026 | 0.0 |

原试验每阶段4步，而非只做1步；第二步起相对于最新中心的偏移可触发保护项。记录的正则标量可达约0.24，不能把整段训练解释成正则项从未生效。首批已存raw梯度包含正则项，data-only梯度由保存的Fisher／中心解析相减，不是额外保存的原GPU数据梯度。

## 固定前缀敏感性：原锚结果是否只是格式

| 序列 | 机制 | S1原／辅助 | S3原／辅助 | S5原／辅助 |
| --- | --- | ---: | ---: | ---: |
| repeat_anchor | SeqFT_LoRA | 2/3 | 8/8 | 8/8 |
| repeat_anchor | ER_LoRA | 2/3 | 9/9 | 6/6 |
| repeat_anchor | EWC_lambda1 | 2/4 | 8/8 | 8/8 |
| repeat_anchor | EWC_lambda100 | 2/3 | 9/9 | 9/9 |
| repeat_anchor | anchor_gradient_noop | 2/3 | 9/9 | 8/8 |
| repeat_anchor | anchor_gradient_projection | 2/3 | 9/9 | 8/8 |
| within_medical | SeqFT_LoRA | 2/4 | 8/8 | 9/9 |
| within_medical | ER_LoRA | 2/4 | 3/3 | 9/9 |
| within_medical | EWC_lambda1 | 2/3 | 8/8 | 10/10 |
| within_medical | EWC_lambda100 | 2/4 | 8/8 | 10/10 |
| within_medical | anchor_gradient_noop | 2/3 | 8/8 | 9/9 |
| within_medical | anchor_gradient_projection | 2/4 | 8/8 | 9/9 |
| cross_short | SeqFT_LoRA | 2/3 | 4/5 | 4/5 |
| cross_short | ER_LoRA | 2/4 | 6/6 | 7/7 |
| cross_short | EWC_lambda1 | 2/3 | 3/4 | 6/7 |
| cross_short | EWC_lambda100 | 2/3 | 3/4 | 4/4 |
| cross_short | anchor_gradient_noop | 2/4 | 3/4 | 4/5 |
| cross_short | anchor_gradient_projection | 2/3 | 4/5 | 5/5 |
| cross_five_domains | SeqFT_LoRA | 2/3 | 3/4 | 4/5 |
| cross_five_domains | ER_LoRA | 2/4 | 7/7 | 7/7 |
| cross_five_domains | EWC_lambda1 | 2/3 | 4/5 | 4/5 |
| cross_five_domains | EWC_lambda100 | 2/3 | 3/4 | 3/4 |
| cross_five_domains | anchor_gradient_noop | 2/3 | 3/4 | 3/4 |
| cross_five_domains | anchor_gradient_projection | 2/3 | 6/7 | 3/3 |

以上分母均23。原／辅助只区分固定英文前缀，不代表语义人工评分；跨域SeqFT最终原EM4/23、辅助5/23，同域最终两者均9/23，说明这一个前缀不能消除全部差距。

## 同样任务数下的格式敏感性对比

| 机制 | 序列差（S5−S1变化之差） | 原EM pp [95%] | 固定前缀辅助 pp [95%] |
| --- | --- | --- | --- |
| SeqFT_LoRA | within_medical − repeat_anchor | +4.35 [+0.00, +12.00] | +0.00 [-13.64, +12.00] |
| SeqFT_LoRA | cross_short − within_medical | -21.74 [-39.13, -4.55] | -13.04 [-31.82, +4.35] |
| SeqFT_LoRA | cross_five_domains − within_medical | -21.74 [-39.13, -8.33] | -13.04 [-31.82, +4.35] |
| SeqFT_LoRA | cross_five_domains − cross_short | +0.00 [-13.04, +13.04] | +0.00 [-13.04, +13.04] |
| ER_LoRA | within_medical − repeat_anchor | +13.04 [+0.00, +27.27] | +8.70 [-8.70, +25.00] |
| ER_LoRA | cross_short − within_medical | -8.70 [-27.27, +8.33] | -8.70 [-27.27, +8.33] |
| ER_LoRA | cross_five_domains − within_medical | -8.70 [-27.27, +8.33] | -8.70 [-27.27, +8.33] |
| ER_LoRA | cross_five_domains − cross_short | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| EWC_lambda1 | within_medical − repeat_anchor | +8.70 [+0.00, +20.83] | +13.04 [+0.00, +33.33] |
| EWC_lambda1 | cross_short − within_medical | -17.39 [-36.36, +0.00] | -13.04 [-31.82, +4.55] |
| EWC_lambda1 | cross_five_domains − within_medical | -26.09 [-43.48, -9.09] | -21.74 [-40.91, -4.55] |
| EWC_lambda1 | cross_five_domains − cross_short | -8.70 [-22.73, +0.00] | -8.70 [-22.73, +0.00] |
| EWC_lambda100 | within_medical − repeat_anchor | +4.35 [+0.00, +13.64] | +0.00 [-13.04, +13.04] |
| EWC_lambda100 | cross_short − within_medical | -26.09 [-44.00, -9.09] | -21.74 [-41.67, +0.00] |
| EWC_lambda100 | cross_five_domains − within_medical | -30.43 [-50.00, -13.64] | -21.74 [-43.48, +0.00] |
| EWC_lambda100 | cross_five_domains − cross_short | -4.35 [-13.64, +0.00] | +0.00 [-13.64, +12.00] |
| anchor_gradient_noop | within_medical − repeat_anchor | +4.35 [+0.00, +12.00] | +4.35 [+0.00, +12.00] |
| anchor_gradient_noop | cross_short − within_medical | -21.74 [-39.13, -8.33] | -21.74 [-40.91, -4.55] |
| anchor_gradient_noop | cross_five_domains − within_medical | -26.09 [-43.48, -9.09] | -21.74 [-40.91, -7.69] |
| anchor_gradient_noop | cross_five_domains − cross_short | -4.35 [-13.64, +0.00] | +0.00 [-13.04, +13.04] |
| anchor_gradient_projection | within_medical − repeat_anchor | +4.35 [+0.00, +12.00] | +0.00 [-13.64, +12.00] |
| anchor_gradient_projection | cross_short − within_medical | -17.39 [-36.36, +0.00] | -13.04 [-34.62, +9.09] |
| anchor_gradient_projection | cross_five_domains − within_medical | -26.09 [-43.48, -9.09] | -21.74 [-41.67, +0.00] |
| anchor_gradient_projection | cross_five_domains − cross_short | -8.70 [-21.74, +0.00] | -8.70 [-21.74, +0.00] |

区间按原图像路径组bootstrap10000次、seed42，未校正多重比较。差上差不能消除S1参数数值差，也不能消除任务内容／答案长度差；不能据此作领域数唯一成因或方法优劣的严格因果结论。

## 因果边界

仍需确定性后端、相同初始权重及任务／内容／token预算控制来判定这种首步失活是否造成最终性能问题。原默认4×6对照S1权重已有数值差异，不能以此报告细小方法优劣；单锚也不能证明领域数是唯一原因。本审计直接证明的是更新机制的启用条件，尚非它对生成准确率的因果影响。

完整首批范数、正则值、CPU反事实、72格输出和SHA见[机器可读记录](EWC_ACTIVATION_AUDIT.json)。
