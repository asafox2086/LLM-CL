# 原权重视觉／训练拟合对照：完成证据

all original native T12 TEST100 and frozen TRAIN6; only five fixed checkpoints, no optimizer

全部输出／参数／源码SHA核验；四个既有终点的原图TEST生成与原基底预算结果逐题复核，400条prediction和scores必须完全相同，NLL误差单列。相同文字输入的候选margin常量门槛通过。无新训练。

| 权重 | split | 视图 | 主EM | 固定前缀辅助 | 两裸候选辅助 | gold NLL | 裸候选margin均值／标准差 |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| past_before | TRAIN | image | 2/6 | 2/6 | 2/6 | 0.5838 | -1.0829 / 0.4306 |
| past_before | TRAIN | wrong | 2/6 | 2/6 | 2/6 | 0.7073 | -1.0829 / 0.4306 |
| past_before | TRAIN | text | 4/6 | 4/6 | 4/6 | 0.5482 | +0.1291 / 0.0000 |
| past_before | TEST | image | 56/100 | 56/100 | 56/100 | 0.4737 | -1.1969 / 0.4397 |
| past_before | TEST | wrong | 56/100 | 56/100 | 56/100 | 0.5343 | -1.1969 / 0.4397 |
| past_before | TEST | text | 44/100 | 44/100 | 44/100 | 0.5126 | +0.1291 / 0.0000 |
| random_A_zero_B_update1 | TRAIN | image | 0/6 | 3/6 | 2/6 | 5.9399 | -3.5816 / 0.6702 |
| random_A_zero_B_update1 | TRAIN | wrong | 0/6 | 3/6 | 2/6 | 6.1243 | -3.5816 / 0.6702 |
| random_A_zero_B_update1 | TRAIN | text | 0/6 | 0/6 | 2/6 | 8.0314 | -6.9824 / 0.0000 |
| random_A_zero_B_update1 | TEST | image | 0/100 | 64/100 | 56/100 | 5.2129 | -4.0224 / 0.7187 |
| random_A_zero_B_update1 | TEST | wrong | 0/100 | 58/100 | 56/100 | 5.4248 | -4.0224 / 0.7187 |
| random_A_zero_B_update1 | TEST | text | 0/100 | 0/100 | 56/100 | 6.7552 | -6.9824 / 0.0000 |
| random_A_zero_B_update4 | TRAIN | image | 3/6 | 3/6 | 3/6 | 0.3787 | -0.4162 / 0.3995 |
| random_A_zero_B_update4 | TRAIN | wrong | 1/6 | 1/6 | 1/6 | 0.4856 | -0.4162 / 0.3995 |
| random_A_zero_B_update4 | TRAIN | text | 0/6 | 0/6 | 4/6 | 0.7705 | +0.0066 / 0.0000 |
| random_A_zero_B_update4 | TEST | image | 62/100 | 62/100 | 65/100 | 0.3359 | -0.5232 / 0.3314 |
| random_A_zero_B_update4 | TEST | wrong | 58/100 | 58/100 | 59/100 | 0.3784 | -0.5232 / 0.3314 |
| random_A_zero_B_update4 | TEST | text | 0/100 | 0/100 | 44/100 | 0.7055 | +0.0066 / 0.0000 |
| past_AB_update1 | TRAIN | image | 4/6 | 4/6 | 4/6 | 0.3338 | -0.0836 / 0.3793 |
| past_AB_update1 | TRAIN | wrong | 2/6 | 2/6 | 2/6 | 0.4128 | -0.0836 / 0.3793 |
| past_AB_update1 | TRAIN | text | 4/6 | 4/6 | 4/6 | 0.3259 | +1.2510 / 0.0000 |
| past_AB_update1 | TEST | image | 76/100 | 76/100 | 73/100 | 0.2958 | -0.1588 / 0.3983 |
| past_AB_update1 | TEST | wrong | 50/100 | 50/100 | 47/100 | 0.3558 | -0.1588 / 0.3983 |
| past_AB_update1 | TEST | text | 44/100 | 44/100 | 44/100 | 0.4045 | +1.2510 / 0.0000 |
| past_AB_update4 | TRAIN | image | 4/6 | 4/6 | 4/6 | 0.2397 | +1.8961 / 0.3914 |
| past_AB_update4 | TRAIN | wrong | 4/6 | 4/6 | 4/6 | 0.3293 | +1.8961 / 0.3914 |
| past_AB_update4 | TRAIN | text | 4/6 | 4/6 | 4/6 | 0.3107 | +2.1257 / 0.0000 |
| past_AB_update4 | TEST | image | 44/100 | 44/100 | 44/100 | 0.3737 | +1.7916 / 0.3666 |
| past_AB_update4 | TEST | wrong | 44/100 | 44/100 | 44/100 | 0.4273 | +1.7916 / 0.3666 |
| past_AB_update4 | TEST | text | 44/100 | 44/100 | 44/100 | 0.4650 | +2.1257 / 0.0000 |

两候选概率只来自TRAIN固定的rural/urban，按完整字符串含EOS计算；不能替代原自由生成分数，不是医学／金融等其他领域的泛化成绩。margin跨图变化本身不证明正确视觉理解；需要与确定性错图对照，保留区间。

## 原图相对错图：配对差

| 权重 | split | 错图−原图gold NLL [95%] | 原图−错图目标带符号margin [95%] |
| --- | --- | --- | --- |
| past_before | TRAIN | +0.1236 [-0.0511, +0.2707] | +0.5411 [+0.1462, +0.7906] |
| past_before | TEST | +0.0606 [+0.0200, +0.1010] | +0.2548 [+0.1323, +0.3773] |
| random_A_zero_B_update1 | TRAIN | +0.1844 [-0.4034, +0.9516] | +0.4481 [-0.1664, +1.1043] |
| random_A_zero_B_update1 | TEST | +0.2119 [-0.0114, +0.4423] | +0.4657 [+0.2719, +0.6597] |
| random_A_zero_B_update4 | TRAIN | +0.1069 [+0.0068, +0.1843] | +0.4999 [+0.1667, +0.7703] |
| random_A_zero_B_update4 | TEST | +0.0426 [+0.0191, +0.0668] | +0.1924 [+0.0974, +0.2861] |
| past_AB_update1 | TRAIN | +0.0790 [-0.0146, +0.1471] | +0.3744 [-0.0003, +0.6449] |
| past_AB_update1 | TEST | +0.0600 [+0.0348, +0.0845] | +0.2823 [+0.1698, +0.3910] |
| past_AB_update4 | TRAIN | +0.0896 [+0.0182, +0.1758] | +0.5000 [+0.1043, +0.8541] |
| past_AB_update4 | TEST | +0.0536 [+0.0331, +0.0745] | +0.2975 [+0.1987, +0.3962] |

## 相同终点权重下训练拟合与测试变化

| 起点 | split | NLL4−NLL1 [95%] | 主EM4−EM1 pp [95%] |
| --- | --- | --- | --- |
| random_A_zero_B | TRAIN | -5.5613 [-6.6755, -4.3927] | +50.0000 [+16.6667, +83.3333] |
| random_A_zero_B | TEST | -4.8771 [-5.1471, -4.6200] | +62.0000 [+52.0000, +71.0000] |
| past_AB | TRAIN | -0.0941 [-0.3342, +0.1599] | +0.0000 [-66.6667, +66.6667] |
| past_AB | TEST | +0.0778 [+0.0092, +0.1463] | -32.0000 [-47.0000, -16.0000] |

这里TRAIN NLL为真实终点1／4下重新评测，区别于原训练日志的pre-update loss。仅6条训练QA，配对区间不是独立训练seed方差。若训练NLL下降、测试退化可定位目标拟合与泛化差，但不足以唯一归因于样本比例；没有对标签比例或优化器做新的因果干预。

## 坍缩不等于图像信号消失

旧AB的4步TEST输出全部urban，但原图相对错图仍降低gold NLL，目标带符号margin也更好。必须区分图像对答案分布仍有信息与自由生成决策是否正确。

1→4步裸候选margin的整体偏移、逐图差值标准差、跨图相关性及按金标类均值见JSON。正值偏urban；这是事后机制定位，不是校准器或测试调参。不能仅凭整体偏移唯一归因于训练类比例，或断言视觉能力完全不变。完整字符串含EOS且两裸候选长度不同，wrapped候选的分数另外保留，不能择优替换主指标。

输出频率、bare/wrapped候选、全部配对区间、参考复现误差与SHA见[完整JSON](RURAL_COLLAPSE_EVIDENCE.json)。
