# 小活动更新空间与强方法成本：已有证据的准确范围

本页只读已有完整结果、保存图布局JSON及冻结源码，0模型／QA／新评分／资源测量。**“方向没有及时对准”仍是待检验假说；更大的主optimizer坐标数与更高分并不能证明这一原因。** 主optimizer持有的参数元素、临时校准梯度、完整保存权重、实际GPU峰分别记账。

## 同LLaVA的主optimizer空间

| 方法／实现 | 主optimizer名义坐标 | 唯一参数张量 | FP32梯度的维度字节数 | 两份FP32 Adam moments的维度字节数 | 状态位置／证据范围 |
| --- | ---: | ---: | ---: | ---: | --- |
| 正式1% PASs-MoE | 140,771,328 | 672 | 563,085,312 | 1,126,170,624 | 原HostAdamW；moments平时在CPU，逐参数搬回计算；此处字节为形状推导，非正式训练GPU驻留实测 |
| 正式1% OSFT | 113,766,144 | 504 | 455,064,576 | 910,129,152 | 原HostAdamW，亦有activation offload；字节为形状推导 |
| 普通shared q/v-r8 | 540,672 | 96 | 2,162,688 | 4,325,376 | 匹配FIT/GPU实现实测完整Adam state为4,325,760，另含96个FP32 step标量共384字节 |

坐标来源是原正式S19保存图的 `model_registered_layout`：只取 `_parameters && requires_grad`，按 `(storage_key, storage_offset, shape, stride, dtype)` 去掉同一视图的注册别名。三法这些视图均为FP32。主optimizer按 `requires_grad` 收集参数，与此名义预算相符；**这不是每步非零梯度数、独立功能自由度或有效更新方向数**。PAS为24层七位置、每位置两组r128 A/B；OSFT为七位置低侧U/S/V；小r8只在48个q/v位置。

PAS／OSFT分别是r8的260.363636／210.416193倍主坐标；比值仅约束这一列。两份moments均为 `8×坐标数`，未把HostAdam的step小标量混入；也不把CPU moments当GPU常驻。`host_adam.py:HostAdamW.step`逐参数将moments移到参数device，更新后再转CPU，不能据静态总字节直接计算训练峰或速度。

直接来源与定位：

- [完整保存账本](../final_weight_reduction_20261024/checkpoint_weight_storage_v1/EVIDENCE.json) 的 `rows[backbone=onevision,method=...]`，承接正式proof及私有布局SHA。PAS／OSFT／SeqFT布局SHA依次为 `d463c7c1…b3d358c`、`ee971bb8…6566b7f`、`a1f2f1bf…08ec6f6`；本页只汇总这些既有JSON，未再读checkpoint。
- [r8 OUTCOME](../plain_r8_gpu_control_20261025/v1/OUTCOME.json)：`train_coordinates`、`gradient_bytes`、`optimizer_bytes`；[S16证据](../plain_r8_gpu_control_20261025/v1/qv_STAGE_16_COMPLETE.json)：`optimizer_state_bytes=4325760, all_optimizer_state_GPU=true`；`qv_grad_S16_001.json.gradient_bytes=2162688`。
- 原 `exp/two_backbone_native_cl_20261017_v2/runtime_utils.py:make_optimizer`、`host_adam.py:HostAdamW.step`；`pass_adapter.py:PASMethod`及`osft_adapter.py:OSFTMethod`说明原实现与位置。PAS与OSFT的保护／投影、数值及LR还不同，不能把质量差单独归因坐标规模。

## 当前方向候选另有全q/v校准成本

普通r8没有这项额外步骤。当前local方向对照的**主optimizer**仍持有540,672个A/B坐标，但 `calibration.py:estimate` 临时让48个q/v稠密权重 `requires_grad=True`，共22,020,096个FP32坐标；梯度G为88,080,384字节，即84 MiB。此步骤只估计梯度，不执行optimizer更新，随后恢复dtype／mask、清梯度；G不进入最终图。

这84 MiB不是整个额外峰：校准时的稠密权重FP32化、`p.grad`与克隆G短时共存、激活、SVD／QR工作区另计。因此应写“主更新预算固定r8、借用当前任务的稠密梯度信息”，不能写“全程只有540,672个可求导坐标”或“校准没有训练成本”。两臂均16条当前FIT校准反传，分别69条主FIT反传／5次主更新；两臂合计32校准＋138主反传＝170，10次主更新。

本local对照从共同S15 fold后再开r8，累计相对W0的更新集合并非全程单r8；两臂共享这一变化。GPU校准峰／耗时须看该新运行真实证据，本页没有补测或预填。

## 完整保存权重与容器

| 同LLaVA记录 | 完整保存系数（含必要持久状态） | 必要权重字节 | 相对原基座权重字节增加 | CP容器字节 |
| --- | ---: | ---: | ---: | ---: |
| 正式1% PASs-MoE，原表示／dtype | 1,034,447,216 | 2,350,437,760 | 563,086,656 | 3,051,292,733 |
| 正式1% OSFT，原未fold USV | 990,913,568 | 2,891,956,288 | 1,104,605,184 | 3,028,589,237 |
| 正式1% SeqFT-r8，原未fold | 894,216,224 | 1,789,513,792 | 2,162,688 | 1,926,007,972 |
| 已核FIT普通r8，实际fold／strict reload | 893,675,552 | 1,787,351,104 | 0 | 1,787,622,158 |

前三行来自保存账本 `rows[].final_saved_storage_coefficients / final_required_saved_weight_bytes / incremental_weight_bytes / checkpoint_file_bytes`；末行来自r8 `OUTCOME.complete_checkpoint.layout.weights`及`.bytes`。原基座就是893,675,552系数／1,787,351,104字节。末行是1213FIT／100已探索DEV，不是原正式1313TRAIN／完整TEST轨迹；不混排它们的质量。

PAS的140,771,328神经增量均FP32，另336个FP32 mask＝1,344字节。原基座FP16：完整系数仅比folded小图多15.752%，必要权重字节多31.504%，不是260倍。原537.001 MiB权重增加量中，268.5 MiB来自该增量用4而非2字节的算术溢价；这不是已经转换或保证无损的FP16 PAS。OSFT还保存冻结的高基／低基完整表示，不能把所有保存系数当可训练坐标。

CP容器还含方法历史、processor、RNG、元数据／旧视觉缓存等。PAS既有提取记录的CPU保护历史692,318,288字节单列，不是实测RSS，也不是推理学习权重。派生RoPE／position buffers等按原账本排除，原OneVision注册派生部分9,032字节另保留。

单shared BA与OSFT线性USV都有常规fold回稠密基座形状的机会；不能用OSFT未fold的2757.984 MiB对已fold的1704.551 MiB，宣称本候选独占结构压缩。OSFT同口径fold后质量及资源尚未实测。PAS的输入相关混合不能直接套单个常量BA的fold删除银行；本页也未证明其保存图已是最优部署。

## 固定五窗GPU锚：只填真实测过的图

| 域 | 正式PAS原完整图 peak allocated，bytes | FIT普通r8实际fold图 peak allocated，bytes |
| --- | ---: | ---: |
| natural | 3,103,171,584 | 2,271,321,088 |
| medical | 2,909,733,376 | 2,140,254,720 |
| geography | 2,611,419,136 | 1,947,183,616 |
| finance | 2,867,997,184 | 2,140,249,600 |
| traffic | 3,022,102,528 | 2,271,311,872 |

来源：[PAS memory.json](../strong_pas_resource_20261025/v1/memory.json)与[r8 qv_memory.json](../plain_r8_gpu_control_20261025/v1/qv_memory.json)，逐项 `domain, peak_allocated`；对应完整证明为PAS `VERIFIED.json` SHA `c987ec2a…753d08`、r8 `VERIFIED.json` SHA `3b172989…e3bc66`。最大值分别2959.415039／2166.100586 MiB。两者已绑定OLD同物理GPU1、同五条raw QA／准备后输入、同32新token与无视觉缓存路径，窗口含prepare／vision／decode；不是整个TEST最坏峰、NVML／CPU总内存或生产吞吐。**OSFT没有这套同卡五窗推理峰证据，保留缺测。**

实际原表示／dtype下的小图低峰成立；不能把差额全归架构或方向方法，也不能按FP32静态字节直接从GPU峰相减。[既有精度范围审阅](../edge_resource_comparison_20261025/v1/PRECISION_FAIRNESS_REVIEW.md)已记录该限制。原正式SeqFT未fold图的1%五窗也未测；这里没有拿探索fold图冒充其正式实测。

## 跨Qwen只能另列

全正式1%最高Qwen SeqFT本身也是简单q/v-r8，并非复杂专家：主A/B 1,089,536，完整保存2,210,075,136系数／4,422,329,344字节，原基座2,208,985,600系数。它的4954.603516 MiB原图五窗最大峰已测，但骨干、原生输入展开不同，不能归给PAS或本候选的路由／方向设计。更不能把Qwen基座大小差当同LLaVA方法压缩。

因此现有证据支持一个明确的研究取舍：**强PAS／OSFT的主更新状态明显更大，小图已有原实现成本空间；是否能靠当前任务方向信息弥补小图新习得缺口，仍须固定对照的功能结果。** 260倍只是参数化训练坐标比，不是260倍部署内存、独立知识容量或已证因果。
