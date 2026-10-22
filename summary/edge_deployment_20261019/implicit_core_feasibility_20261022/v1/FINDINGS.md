# 隐式核心：具体表示仍有未解代价，停止实现这一分支

问题：完整允许集合的较少训练坐标，能否自然换来更简洁且初态对齐的表示？本轮只执行整数计数和固定源码阅读，零张量/模型/QA/GPU。COUNTS.json绑定实际脚本、旧形状和问题文件SHA；没有新质量结果。

设权重m×n、h=.75min(m,n)，p=m−h、q=n−h。分别对U_high及V_high转置作QR；反射数组保留上三角R和下部reflector，不另存R。精确算术下W=Q_u blockdiag(R_u diag(S_high) R_v^T,C) Q_v^T，自由核心C为p×q。两高基并非任意有限精度下严格正交，不能把R当符号单位阵。此为自行表示推导，不是模型前向等价证明。

| 所计目标张量 | 原USV | 直接QR/core | 原实际FP16起点+零增量QR/core |
| --- | ---: | ---: | ---: |
| 永久state bytes | 1820258304 | 1650859008 | 2366171136 |
| state+FP32梯度 bytes | 2275322880 | 1935842304 | 2651154432 |
| 两Adam矩CPU bytes | 910129152 | 569966592 | 569966592 |

直接形式永久state仅少9.31%，初态实际FP16重构未证相同；保留原实际起点的offset形式state反增29.99%，state+gradient增16.52%。这两种具体实现不足以支持继续算子适配；停止该实现分支，不是隐式表示普遍下界。71245824核心坐标是旧结论，不重复当新发现。

这些是目标矩阵的静态字节账，不是实际GPU/CPU峰值；共享基座、激活、重构矩阵、工作区、传输、allocator和I/O另在范围外。安装Torch2.6头文件OrmqrBackward0声明input2_/input3_/self_/result_；原训练有非重入checkpoint和CPU offload，不能把168个保存矩阵相乘当真实显存峰。

来源：[LAPACK隐式Q](https://www.netlib.org/lapack/lug/node128.html)、[Torch2.6文档源码](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/torch/_torch_docs.py)、[导数声明](https://raw.githubusercontent.com/pytorch/pytorch/v2.6.0/tools/autograd/derivatives.yaml)。[HOFT v1](https://arxiv.org/html/2505.16531v1)已读，其学习左右Householder正交变换；不同于这里冻结补基+自由核心，Householder本身不是原创点。未采用其实现或性能数字。

下一行动改变：采用原表示也存在更简单的可检验问题——固定完整短侧补基，训练长侧及S；无需实现QR/ormqr或改变原前向。其独立证明、必要回调处理和有限配方审阅另列，不从本计数声称F1/端侧收益。
