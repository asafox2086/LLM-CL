# 跨方法部署功能审计：学习时的结构不等于部署时的必要结构

2026-10-29，只读源码、既存质量/权重证据及两个末态CP的stat/小元数据；0模型加载、0CP载荷读取、0Torch执行、0QA正文/新评分、0新GPU实验。研究对象是端侧部署背景下方法已学功能如何承载，不是特定PAS优化，也不限制训练更新空间。FullFT新增接入草稿本轮保持，不扩实现。

## 共同问题与边界

应区分三种状态：(a)只为学习服务的统计/保护/回放状态；(b)推理时输入无关、可静态物化的表示；(c)推理仍依赖输入的条件功能。存储银行很大不能直接证明(c)有价值，不能从(c)存在直接断言它不能被静态函数近似；相反，无路由不意味着训练功能弱。FOREVER仅约2.0625MiB新增且可以常规数学合并，必须公平比较。训练参数数目不作为部署目标。

下文“可fold”区分数学函数等价与有限精度实际等价：普通LoRA将FP32分支并入FP16基座可能改变舍入和矩阵乘加次序，不能自动继承原F1。OSFT的forward本身先产生固定FP16稠密W，物化同W较接近原算术；仍须独立重载/完整质量与资源证明。所有普通合并与编译收缩均不单独宣称创新。

## 八法实际eval依赖

路径基准：`exp/two_backbone_native_cl_20261017_v2/`；FOREVER另见`exp/two_backbone_forever_20261019_v3/forever_adapter.py`。各方法不是仅按名字分类，下表依实际实现。

| 方法与实际类/forward | 只服务训练的主要状态 | 输入无关部分 | 真正不能直接常量fold的eval依赖 |
| --- | --- | --- | --- |
| SeqFT_LoRA：`methods.SeqLinear.forward` | optimizer/scaler；dropout在eval关闭 | `W x + scale B A x`，常规BA数学合并 | 无条件router；FP32分支→FP16基座合并需实测数值 |
| FOREVER：`forever_adapter.ForeverMethod(Method)`，推理继承同`SeqLinear` | `ForeverStage`漂移/回放选择、TRAIN库、作者replay；end_stage后core清空 | 同一套q/v-r8；无推理历史专家库，可常规合并 | 无新增推理router；回放训练费用不能省略，也不能把低新增参照改名FullFT |
| CL_MoE：`clmoe_adapter.CLMoELinear.forward`及`CLMoEMethod.prepare_row/feature` | previous专家拷贝、训练收集total/count、seen及阶段间momentum等 | 每专家BA；若所有混合固定，则数学可合并 | token级learned linear-router softmax，加0.5权重的question-feature nearest-anchor所得历史task_weights；question feature需关闭adapter后额外decoder前向。不是PAS的A能量路由 |
| PASs_MoE：`pass_adapter.PASMethod`→作者`CoINMOELoraLinear.forward` | learner `_I_agg/_A_ref/_B_ref/_cur_I_*`、rank-energy统计和梯度正则；eval不读取这些 | 两专家各自BA；常数混合可数学合并 | 每token的A投影平方能量softmax，原实现168投影、每处2套r128；A同时定义expert和router。两个专家都计算，不是稀疏执行 |
| PaRSP：`parsp_adapter.PaRSPMethod`，原decoder线性层forward | attribution/localization、region梯度hook、历史covariance/nullspace projectors、probe IO | 已直接更新原稠密W，没有adapter银行，无需额外fold | 没有依输入选择推理参数的router。原选中过的attention/MLP被转FP32并保留，影响权重字节；不能把这部分混成额外参数或无损半精度已验证 |
| OSFT：`osft_adapter.OSFTLinear.forward` | reSVD、梯度/参数投影及阶段保护规则 | 每次重构同一`reconstruct_weight_matrix(svd, FP32, output_dtype)`再F.linear；终态可静态物化这个W | 无输入条件router；USV保存态不是不可避免的部署增量，旧CPU逐层等价不等于GPU全TEST已验证 |
| SpaRTA：`sparta_adapter.Router/ SpaRTAMethod.make_hook`→作者`DATAInjectedLinear.forward` | progressive Gram-Schmidt分配、orthogonal penalty、stochastic restore初始副本 | 两个rank2/rank8分支各为BA，scale3保持1；固定两个scale后数学可合并 | 每4层共享router，从该组入口序列算两个[B,1,1]标量，经sigmoid后加权两rank分支。不是逐token专家softmax；prefill全L求和和decode L=1的尺度语义不同，实际影响未测 |
| SMoLoRA：`methods.AuthorSMoLinear.forward`→作者`SMoLoraLinear.forward` | 首阶段后gate冻结、训练offload保存激活等 | 每专家BA；固定instruction分支中部分选择可静态确定 | visual/input分支按token序列均值作top1、instruction embedding按当前instruction index作另一个top1，随后两个分支输出再经learned fc产生逐token两路softmax混合。仅固定instruction不足fold全函数；当前方法确实使用instruction索引，不能误称纯task-free |

CL配置总rank64被作者`CLMoEMOELinearA/B`按8专家均分，所以每专家rank8，不是8套rank64。SMo不能据表面r128/8等字段凭空填总部署参数，须依实际保存布局核验。

## 现有正式质量与部署账本

以下为同LLaVA正式1%五域最终宏原值，不把新的100题探索probe和完整TEST并排冒作同设置。权重列来自既存`summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/EVIDENCE.json`的required-saved口径，包含必需未注册推理状态；未fold保存态不是最优部署下界。基座893675552系数、1787351104B。

| 方法 | 完整必要保存系数 | 保存态额外系数（非最小部署增量） | 必要权重B / MiB | 原完整最终五域F1 |
| --- | ---: | ---: | ---: | ---: |
| SeqFT_LoRA | 894216224 | 540672 | 1789513792 / 1706.613 | 42.185908 |
| FOREVER | 894216224 | 540672 | 1789513792 / 1706.613 | 40.830165 |
| CL_MoE | 930877024 | 37201472 | 1936156992 / 1846.463 | 42.774019 |
| PASs_MoE | 1034447216 | 140771664（含336 mask标量） | 2350437760 / 2241.552 | 46.619976 |
| PaRSP | 893675552 | 0 | 2383774784 / 2273.345 | 43.888271 |
| OSFT | 990913568（USV保存态） | 97238016（不是不可避免部署增量） | 2891956288 / 2757.984 | 45.937438 |
| SpaRTA | 904077344 | 10401792 | 1828958272 / 1744.231 | 43.261311 |
| SMoLoRA | 正式末态布局未核，不填 | 未测 | 未测 | 尚无完整正式终态质量，不拿S14中间CP冒充 |

CL的必需anchors/task_weights仅170240B，不能把这小部分元数据声称为整个银行内存来源；注册神经增量为37158912，必需未注册系数另42560。PaRSP新增系数0却多596423680B，体现参数、dtype与训练辅助状态三本账不能互相代替；是否允许改变dtype须独立质量证明。

### 已完成且当前确实存在的不同机制干预资产

本轮仅OLD stat与small JSON，不哈希/读取payload：

- CL-MoE：`/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/CL_MoE/training_20261017_161424/checkpoints/stage_19.pt`，2100250526B，原meta SHA8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4；`runner_source/clmoe_adapter.py` SHA33946eddfc6424638ded0675ded68860273859c2f71993a6a60bd22556341be9，与NEW源一致。
- SpaRTA：同根`SpaRTA/training_20261018_173800/checkpoints/stage_19.pt`，2007664585B，原meta SHA4e3c9fe9fffb349573835a5635c7c8bb6bf25a500886b7a6866892ffec3bb476；adapter SHA8a00505b40663e5a233ddceb34e295d9b7b73d55f0679ac3d2b93318e77d7a44，与NEW源一致。

## 最有信息量的后续判别（均未运行、未立新GPU点）

1. **CL作为第二种条件机制的近似验证对象。** PAS的A能量双专家与CL的learned token router+question anchor双层条件选择不同；若共同的“训练保留复杂适应、部署静态承载”在二者都成立，才支持跨机制观察。应保留同一已训练CP，事前固定TRAIN-only常数规则，先原动态/未fold静态，再同静态的数值fold，比较实际功能而非entropy。CL的8专家常数求解不能直接抄2专家闭式c，也不能根据TEST选混合。若只去掉anchor先验而保token router，回答的是其层级选择必要性，不是整个专家银行可删除；两种问题不要混称。
2. **SpaRTA保留动态功能的代数收缩，区别于PAS去条件。** 当前`get_scale`里K先逐行normalize，然后`einsum('blk,kd->blk', a_q,K)`使K仅通过行和进入；最终`sum(einsum('blk,kd->bld',sim,W),(1,2))`使W也只通过行和进入。固定config.project=0/nomlp=False分支下，完整K/W的eval作用可在实数算术中由行和承载；A与sigmoid及输入条件性继续保留。LLaVA 24层/6router、两分支、每支152个组件、宽896，保守将K/W换成各自row-sum向量，算术可少3264960个FP32系数/13059840B=12.454834MiB。当前active组件区间必须按checkpoint保存的task_id取m=8(task_id+1)，set_instruction是no-op，不能按评价任务改变；S19现有配置对应m=152。此是该实际实现与配置的代数观察，不是作者算法普遍bug，也没有GPU逐值/F1/整模资源证据。浮点reduction次序与独立重载须实际检验；即使完全保功能，也只是可删的表示自由度，不能单独当论文新方法。
3. **SpaRTA请求内路由粒度的函数观测。** 它的两个scale最终sum所有prefill token，而单token decode时L=1；同一QA的阶段切换可能改变适应强度。先固定请求直接记录真实prefill/decode scale及其对应函数贡献，再决定是否有必要做保持请求级选择、只固定decode尺度的控制。不能由公式推断已饱和/错路由/导致质量差，也不能把不同L必然变化当充分论文结论。不新扫层/温度/样本；若实际没有值得解释的差异应关闭这一疑点。

## 审计结论

现在有两类不同的、可证伪的跨方法对象：CL测试条件功能是否必要，SpaRTA测试同一动态功能是否被过度参数化。OSFT/Seq/FOREVER作为已知静态可物化参照，PaRSP用于揭示零新增与实际权重字节不等价。源分类本身不是科学成功；需要实际功能保持及完整质量—部署成本实测，才能形成论文结论。没有据此启动实验、改变冻结科学源/设置或把已知LoRA合并冒作创新。

## 跨方法共同的待检验准则（非新实验绑定）

对仍有线性专家增量的条件方法，统一问题可写成：冻结已训练专家输出u_e(x)和原有效混合G_e(x)，求每层常数c，使历史TRAIN答案预测位置上逐QA等权的 `||sum_e (G_e(x)-c_e) u_e(x)||²` 最小，约束`c_e>=0, sum_e c_e=1`。PAS的二专家闭式是E=2特例；CL的E=8必须包含原prepare_row的nearest-question-anchor与token-router组合，不能只替换一半条件路径却称原函数已静态化。

均匀混合`c_e=1/E`是必要简单对照，区分去条件本身的价值与校准的增量价值。局部拟合优于均匀的平方残差不保证全模型CE或F1更好。该定义是待固定的跨方法设计原则，不是已运行8专家求解器、全方法通用压缩保证或修改当前PAS冻结系数的许可。SpaRTA保条件的充分表示收缩属于另一种干预，不能混称同一拟合方案。

独立research_review已核本审计的八法分类与SpaRTA代数边界；现有证据不足以宣称任意一种新增实验成功。
