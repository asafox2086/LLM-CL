# 原生生成前缀仍保留PAS／CL局部条件变化差异

## 这次观测回答了什么

**是：原先“PAS专家参数更多、相对固定混合的局部变化反而更小”的排序，在同80条历史TRAIN的question-only原生greedy前缀仍成立。** 两种固定参照、prefill及共同decode支持均同排序；五域各16题也同排序。不是仅有gold answer进入teacher forcing时才出现的现象。

**未验证：专家是否可删除、替换后F1能否保持、10%／其它方法是否相同，均没有本实验支持。不能由本结果推出这些结论。** PAS与CL来自各自原生隐藏轨迹／生成历史；本次没有替换模块、生成质量评分或10%同类验证。局部R/base也都很小，不能把CL较大的R/U直接解释为必须保留其银行。

## 固定口径与完整结果

同原LLaVA1% S19完整method-graph CP、相同80历史TRAIN及168投影。原question-only输入、原greedy生成参数、原任务cap／EOS／cache；只附加前32个实际预测位置后的停止规则。每模型80次原生生成，PAS717／CL752个预测forward；prefill各80题，两模型共同decode支持也是80题。EOS保留，无缺测补零、无筛选QA。

两个参照均不注入模型：旧teacher-forced全局c原样运输；native全局c先每QA在其最多32个实际位置平均，再80QA等权。各phase／域／共同支持都使用同一对应全局c，不分组重拟。decode先QA内平均实际位置，再QA等权；不按token数池化。

| 原模型 | 参照 | 阶段（QA） | R/U，平方比% | R/base，平方比 |
| --- | --- | --- | ---: | ---: |
| PASs_MoE | oldTF | prefill末位（80） | 0.011833256 | 6.667661994e-09 |
| PASs_MoE | native_global | prefill末位（80） | 0.011815812 | 6.657832683e-09 |
| PASs_MoE | oldTF | 共同decode（80） | 0.007066971 | 3.204445424e-09 |
| PASs_MoE | native_global | 共同decode（80） | 0.006953718 | 3.153092296e-09 |
| CL_MoE | oldTF | prefill末位（80） | 1.919410639 | 8.654867812e-08 |
| CL_MoE | native_global | prefill末位（80） | 1.901813932 | 8.575522012e-08 |
| CL_MoE | oldTF | 共同decode（80） | 3.472215263 | 9.024561631e-08 |
| CL_MoE | native_global | 共同decode（80） | 3.454770005 | 8.979220030e-08 |

**阶段事实：两种参照下，PAS的decode R/U均低于prefill；CL均高于prefill。** 这只是各自原生轨迹上的局部平方比随阶段不同，不是共同token干预、长度因果或route本身变强／变弱的证明；分子与动态增量分母都可能随阶段变化。

teacher-forcing旧读数与本次native读数涉及不同位置／隐藏轨迹／历史，不能将差值直接当作训练或gold的因果影响。输入端未传gold、未计算CE／F1；所有读数都是原投影增量相对固定gate参照的局部差异。R/U为平方比，不是幅值比或能量保留率。

## 核验与边界

父及双worker均完整结束；原binding、CONTROL／SLOT、三原身份gone、每模型80QA收据、每QA168层×实际预测位置、phase／position文件SHA与原生cache位置全部闭合。全部8个主格超FP64尺度负残差均为0；最大layer-QA raw／稳定表达差PAS为4.1083e−15，CL为1.2490e−16。原始负值／roundoff诊断保留，未clamp、未归一gate、未投影协方差。稳定表达使用c_aug=[c,sum(c)−1]，保留实际有限精度gate-sum偏差。

首次源写入NEW persistent遇ENOSPC，只留下0B失败文件；原文件及失败记录保留，完整已审源先写tmpfs再持久化新版本名。原81d消费者default在已回收systemd unit的ExecMainPID字段拒绝、未打开QA流／未创建readout_v1。最小948兼容只在LoadState=not-found时使用SHA固定的976终态收据及754父kernel身份事件，核全部START／EXIT／CONTROL／SLOT链；统计代码未变。754没有systemd退出码字段，不冒称其证明退出码。随后default READY，readout_v1仅一次--execute，真实EXIT0及COMPLETE。没有模型或生成重跑。

RESULT SHA `8e7e509b31501fdea20ebf1de35c0453f1648ee591b7b3bdfe92bb8b72213a93`；COMPLETE SHA `47eb46902fca5d25c279415076d38123a77e2b05db353ddf42ab1b35c837645d`；服务器间耐久清单SHA `eda34007fadd9cd26e06ab47ec177f8c073a98dd840f07d26afa78fa9f15036f`。完整结果含perQA分布、五域、两参照和全部输入SHA；原QA-phase／position流留OLD，未下载checkpoint或QA流到Windows。
