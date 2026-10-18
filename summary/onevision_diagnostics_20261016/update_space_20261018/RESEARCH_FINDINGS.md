# 相同训练参数数量，不等于相同可更新空间

只读原native自然域后同一起点和T02/T11/T12三条已封存TRAIN单步更新；不读TEST、不执行模型前向或新优化器。

| 量 | 全48个q/v之和 |
| --- | ---: |
| trainable_coordinates | 540672 |
| free_local_weight_dimension | 537600 |
| protected_local_weight_dimension | 150528 |

free局部维数 r(m+n-r)；protected高空间秩k=min(m,n)-r后，允许块维数(m-k)(n-k)。矩形v存在额外零空间，不能把全部矩阵误算为r²。合成三形状以实际Jacobian秩独立核验。

| TRAIN任务 | QA | 实际BA位移范数 | protected能量% | 各向同性均值% | free能量% | 各向同性均值% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T02 | 11 | 0.103087 | 0.7706 | 0.6847 | 2.5014 | 2.4425 |
| T11 | 12 | 0.103082 | 0.7952 | 0.6869 | 2.5296 | 2.4446 |
| T12 | 6 | 0.103106 | 0.7621 | 0.6827 | 2.4776 | 2.4405 |

各向同性参考是每层维数/(m×n)的精确方向均值，再按原实际位移能量加权；不是随机显著性检验。小子空间本来就只能覆盖少量任意方向，不能只凭低能量比例判定特殊不匹配或有害。

这提供一个需要GPU对照检验的风险：缩小SVD余量同时加保护会限制可更新方向。不能据此认定该限制已造成准确率损害，也不能把原LoRA方向当作最优方向。当前两组GPU对照保持原样，等待实际习得／最终F1与资源证据。

局限：
- Known local rank-manifold and orthogonal-projection algebra, not novel theory.
- One original native LoRA start and three sealed one-step TRAIN updates only; no population claim.
- Projected finite LoRA displacement is a representability diagnostic, NOT the SVD optimizer step or loss descent.
- Protected/free SVD GPU starts and trajectories have NOT been measured by this script.
- Native effective coefficient uses original W plus scaled BA; separate BF16 matmuls need not equal one merged forward.
- Low retained energy does not prove bad directions, poor F1, or that removing protection improves performance.
- Small projected energy is expected in a low-dimensional space; an analytic isotropic mean is reported, not a stochastic significance test.
- Local differential dimension is not statistical capacity, memory, trainability, or independent semantic knowledge.
- No TEST predictions/answers/scores, model forward, GPU, optimizer, route or hyperparameter selection.
- No edits to frozen queued sources/protocol/settings; both predeclared SVD conditions and full confirmations retained.

## 按问题记录

1. **探究的问题：** 同样训练参数数量时，保护是否会限制新任务可用更新空间？
2. **做的实验：** 合成方形／宽／高三种实际Jacobian秩验证；原模型48矩阵的同一起点分解；三个完成TRAIN控制的实际有限BA更新投影。计划／证明含冻结评测ID元数据，但没有读取TEST预测、答案或评分。
3. **结果：** 540672坐标对应free537600／protected150528局部权重维数，保护减约72%；三TRAIN更新protected能量0.762%—0.795%，各向同性均值0.683%—0.687%；free2.478%—2.530%，均值2.440%—2.445%。
4. **证明了什么：** 精确匹配训练坐标仍不匹配局部可更新维数；限制的存在已量化。没有证明该限制导致低Answer F1，也没有证明权重奇异值能衡量任务重要性。原LoRA方向并非最优方向，局部代数不是新的因果理论。
5. **改进的方法：** 当前仍用已冻结free／protected同参数化对照检验保护的实际效应；后续训练信息驱动的方向／位置选择只是候选，需要新证据，当前不扩条件或按TEST选秩。
6. **改进的效果：** GPU训练与最终21454尚未运行，此项无准确率改善可报告；两个条件均保留，不因几何结果取消负条件。
