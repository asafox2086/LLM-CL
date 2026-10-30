# 固定block12残差出口修复pilot — 科学协议草稿（未准入）

仅PAS与CL原S19 teacher，各独立进程串行、各原CP一次load；一个student graph按A/B/C/D切换原existing down W。人类block12明确定义为decoder.layers[11]；只拟合该block.mlp.down_proj权重，bias固定。其它全部168投影由已发布READOUT的全局mean gate FP32 fold后cast原FP16构造，不校准新c、不域路由、不QP。不是全24block验证，不是整方法已成功，也不恢复原CL专家混合四臂。

## 问题与四个必要控制

A=全部meanfold，不修；B=teacher原Xt_down→teacher原Yt_down；C=student原Xs_down→同Yt_down；D=同Xs_down→[teacher完整block输出−student未归一skip残差]。若拟合参数仅W，四臂的固定bias须从各目标减去。skip为注意力残差相加后、post_attention_layernorm之前的hidden，绝不是norm后MLP输入。teacher完整block输出与其原skip+down输出实际bitwise闭合必须通过。只改最后down W时，Xs_down和skip不依赖此W，故同一student A前缀可供C/D拟合。真实FP16下D仅是拟合目标，必须另外量实际block输出/最终logit KL，不能称代数保证有限精度零误差。

B/C/D共享meanfold Wprior、固定bias和完全相同λ=0.01×meanDiag(teacher equal-QA Gram)；不按结果换λ/层/数据或加重试。按QA内预测位置均值、再80QA等权。矩阵及Cholesky/solve均GPU FP64，失败不加jitter/调参；求解W后立即cast原FP16后评估。记录原normal-equation残差、λ、权重pre/postcast更新平方范数、舍入误差和变化的FP16系数数。闭式拟合更新参数，不称完全未训练；无反向梯度、optimizer或新增部署系数。

## 冻结数据与来源

calibration=原pas-static-selection-v1中固定80历史TRAIN（每域16），原选择SHA710f9a7c15a5b7830eea6c16e56b09e8c155a1ddf48b7ad04dd27a55aa089f2b；mean gate来自唯一已发布READOUT SHA55e95fd9dde9a76b34aa02bdd12b507e03506feb2137a706ba31067c37da6ef1。readout=原FIT1213减去这80后的1133池，按每域sha256('residual-exit-readout-v1:'+id)排序各取4，共20。原FIT已排除旧100；再次硬核readout与80/旧100集合不交。只读取原TRAIN文件/图像与小metadata，不读TEST/score/旧raw矩。样本仍原模型训练已见，readout仅未参与本pilot拟合，不是训练未见泛化。

按原causal loss移位的答案预测位置（含EOS）配对teacher/student。不能把答案输入位置当预测位置。原输入/图像prepare与processor及冻结原native源码SHA不变；teacher原CL question-feature/anchor路径保留，额外前向成本单计。学生无method/router，只用原base输入格式。不会score答案、生成答案或导出新CP。

## 实际观测

拟合前后teacher权重/buffer/source摘要一致。student除block11.down W外所有参数数值/dtype/别名不变；全部168原条件投影已转dense、无旧专家/路由/feature-cache依赖，893675552单骨干神经参数。构造student时用deepcopy memo替换条件模块，保持其余参数别名，并拒绝student与teacher共享参数storage。

20读出QA按A/B/C/D一一报告实际block/residual流平方误差、完整词表teacher→student logit KL及top1 agreement；不是F1或label CE。KL按token在QA内均值，再QA等权，GPU FP64 log-softmax可分token chunk但不截断词表。每臂的Xs_down和skip必须与A bitwise相同；只有block12 down W改变。保留全部QA读数，不按结果挑例。teacher/student实际最终dtype均原FP16，拟合和fold的FP64/FP32仅临时，非端侧权重精度优势。

D仅fit好而heldout差就关闭此固定配方。D胜C却劣A，或block误差下降但最终KL变差，不支持扩大。C≈B而D在heldout胜C且A，只支持该试点收益来自residual target修正，而非仅换Xs配对。单block失败不能否定全部顺序修复，成功也不证明24block/最终F1。BRECQ已有残差block输出重构，CrossMean/RegMean++已有prefix校正；本项目仅验证单CL条件teacher→原dense骨干的出口修复适配。

## 资产与执行状态

本草稿没有binding/启动许可。NEW GPU3仅在原QwenOSFT按完整阶段checkpoint真正暂停、worker gone并独立核验合法空槽后才可用；NEW0/SMo/其它UID不动。两个原teacher最终CP永久保留OLD。NEW副本拟仅5,151,543,259B放独立tmpfs，非耐久final证据；小metadata/source/proof/results放persistent。不得挤占baseline下一CP持久空间。源25/29项以原各config实际映射完整核验，不能通过改原源码或放宽ROOT guard来搬运行。模型和矩阵计算均GPU；CPU仅资产I/O及数学参考审查，不CPU模型实验。复制/运行各自独立准入、无自动retry；此刻未启动任何pilot模型。

## 执行前度量明确化

A本身是FP32 meanfold再cast FP16，故A与teacher的差异同时含路由替换和fold/dtype/加法次序舍入，不能把所有prefix差归因router，也不为隔离它增开新模型臂。学生构造以meta Linear再赋值既定权重避免无意义随机初始化，另外保存/恢复CPU/CUDA RNG并记录构造期间是否推进，不能只口头称未动RNG。

拟合报告由已存equal-QA G/R/原prior残差平方量计算precast和cast权重的理想线性残差r²−2〈Δ,R〉+〈Δ,GΔ〉、共同λ正则项、完整目标与FP64消减尺度；这些不包含实际FP16 GEMM，heldout原模型输出另测。cast权重必须全finite。heldout除大residualstream归一外，保留raw block MSE、相对A变化和teacher block实际增量(blockout−blockinput)能量尺度；不得只展示被巨大residual denominator稀释的比值。

## 模型启动前入口/计量合同

冻结frozen_vision.install中的Python视觉缓存闭包deepcopy后仍可能指向teacher。student必须恢复其自身绑定的原PackedVision.forward（原函数支持相同prepared 2D features）；断言wrapper.model/lm_head和visual.core/tower/projector/image_newline均对应student.core、原绑定__self__为student.visual且无closure。teacher不改、prepare依旧bind teacher，不为student改global ACTIVE。构造后非168替换子树的原参数/buffer在GPU逐同名torch.equal核实际数值，并保原shape/dtype/alias合同。

原冻结checkpoints.load完整payload SHA分支必须实际执行且Python assertions有效，记录loader源SHA/原CP SHA/resolve路径/耗时，不用tmpfs stat代替。注册神经参数及weight bytes按remove_duplicate=True实际计数，A/B/C/D同一registered graph，仅旋转原W。GPU set_device后、load前重置peak，结束同步后记录allocated/reserved峰、GPU UUID/name、whole-worker wall及Linux ru_maxrss。该峰包括teacher+student+fit/哈希，仅为本pilot临时开销，绝不是部署推理峰/速度优势，不能填主表部署指标；未来真正部署资源仍需独立无teacher重载测量。
