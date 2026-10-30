# 四组固定点对照：完整诊断结果，暂不扩展方法

两个原S19 teacher均完成一次GPU诊断；80历史TRAIN拟合，20条不参与本次拟合的历史TRAIN读出（五域各4条）。这20条仍是teacher训练时见过的数据，不是TEST／未见teacher样本。固定block12／仅已有down_proj.weight／相同student图、新增参数0。

A：全部条件投影按原全局平均gate合并，不修复。B：teacher特征拟合teacher down输出。C：student特征拟合teacher down输出。D：同student特征，拟合teacher完整block出口减去student残差入口。共同lambda=.01×teacherGram平均对角，统一FP64求解后FP16回写；不是F1或完整24层方法验证。

## 原始完整均值

| teacher | 臂 | block原始MSE ↓ | 最终全词表KL ↓ | teacher top1一致率 ↑ |
| --- | --- | ---: | ---: | ---: |
| PASs_MoE | A | 0.208955669 | 0.002596328 | 0.994944620 |
| PASs_MoE | B | 0.208971907 | 0.002630980 | 0.994295270 |
| PASs_MoE | C | 0.255940087 | 0.002655043 | 0.994796751 |
| PASs_MoE | D | 0.062664108 | 0.001316695 | 0.994291909 |
| CL_MoE | A | 0.065626164 | 0.000942934 | 0.999710983 |
| CL_MoE | B | 0.065592480 | 0.000947541 | 0.999026051 |
| CL_MoE | C | 0.058850790 | 0.000730690 | 0.998232192 |
| CL_MoE | D | 0.016433762 | 0.000510877 | 0.999350649 |

## 支持与否定的解释

- **支持的局部性质：拟合完整残差出口，比只拟合该down分支更有效。** C→D保持同student输入、同Gram／lambda／先验、同一个可改weight；PAS block 20/20、KL19/20下降；CL block20/20、KL16/20下降。D同时比不修A的两项总体均值更低。
- **不支持把teacher→student输入重配本身当共同根因。** B→C，PAS block20/20更差、KL总体均值更差；CL才两均值都下降。因此“前缀错位统一解释了两法问题”未成立。
- **不能称准确率提升或质量等价。** D相对A，两法teacher top1一致率均略降；CL遥感域block更好而KL从0.000057400升至0.000114438。无新生成、CE、F1、TEST或最终CP。
- **局部统计不能直接预测整网保真度。** 原80题局部差值/增量平方比PAS低于CL；本20题A的最终KL却PAS0.002596328高于CL0.000942934。输入支持、精度和真实隐藏轨迹并不相同，不能据排序反转归因某个因素；尤其不能把0.010457%说成整模误差。

## 范围与后续决定

仅两个同骨干1%专家方法、单block、单种子、历史TRAIN答案位置（含EOS）。相同残差结构的代数关系不是新发现；本结果是这两个实例的实际对照。PaRSP／OSFT／FullFT未由本诊断代表，10%未验证；不能作为普适性证明。遵循人类新指示，保留为诊断，不自动开始24层修复、方法训练、调层或调lambda。下一步优先查跨扩展／非扩展方法的实际学习事实。

student图登记神经参数均893,675,552；诊断临时GPU峰包含teacher+student+Gram/solve，不能作部署峰，不填主表Ours。

## 完成与故障公开

两teacher分别真实EXIT0／gone。原控制器在全部模型完成后，因tuple经JSON成为list，最终SLOT发布相等检查失败；原controller EXIT1与FAILURE／CLAIM／COMPLETE全保留。仅原SLOT字节归档后通过有限审计发布released；透明兼容readout只放行这个已审故障，保持原paired统计、同marker／同输出；唯一消费成功，未重跑模型或控制器。

完整归档：[DURABLE_ARTIFACTS.json](FINAL_RUNTIME_ARTIFACTS/DURABLE_ARTIFACTS.json)。完整配对汇总：[PILOT_PAIRED_SUMMARY.json](FINAL_RUNTIME_ARTIFACTS/summary/residual_exit_pilot_old_runtime_20261030/PILOT_PAIRED_SUMMARY.json)，SHA `877eb612e88b849a20a436d6e0ac03a4a73d7c8d0bd87e342e0ec9d5cfcba26b`。源审／有限恢复原件均在同一归档；不得重放release、controller或consumer。
