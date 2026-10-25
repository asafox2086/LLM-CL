# 原生响应选择：先检验干预是否可辨

规则033要求相对公平参照真实减少完整最终权重或结构，没有规定每个候选必须小于原未适应基座。保留全部OneVision而避免PAS式永久专家，仍可能满足目标；但OSFT常规fold后的893675552系数必须作为公平参照，不能拿其原USV保存态制造优势。旧native E/Q/R、旧单臂能量beta、spatial dtype故障点均保持关闭。这里透明定义曾未执行的动态/静态功能问题，不宣称已有框架或新发明。

后续可能比较：完整OneVision+共同宽q/v（4399104坐标）+24个零起点beta；原FFN phi=SiLU(Gx)*Ux，h=Dphi，按通道索引奇偶定义h_diff=D(sign*phi)，新项beta*d(x)*h_diff。动态d=(mean(G_even²)−mean(G_odd²))/(sum+1e-8)，静态d=c_l>0；c为合法S1等QA RMS，不搜索温度、分组或幅度。它不是纯空间或预训练知识因果；24beta之外仍有共同宽q/v。两臂都保显式分支时才有固定表示的动态/静态比较；静态fold若另做须承认FP16舍入差异。额外Down、校准与所有权重/峰计费，最终质量仍需完整对照。

本次只进行有界GPU诊断，不准入后续训练。先判断d和h_diff是否真有非零且变化的输入相关信号；若退化，直接停止，不为无效对照编写完整训练管线。

## 数据与计算

- 原未适应OneVision完整FP16权重/SDPA、原processor与面积401408、原问题/上下文/原answer_loss/answer512上限+EOS不改。无LoRA、beta或forward替换。使用既有325个S1 FIT组的固定`vision-interface-20261025:`哈希顺序前32，各组仍取字典序首FIT QA；不据质量/梯度选样。原TRAIN容器含DEV行但仅按既有FIT IDs筛选，不使用DEV/TEST行或未来域模型输入。
- 每QA一次GPU视觉处理及一次原answer_loss语言前向，32真实回答NLL前向；仅检查有限，不分析/选择NLL值。0新生成、F1、反向、优化器、teacher模型或模型压缩。每QA在原预测位置[prompt_length−1,total_length−2]均匀取min(64,target长度)个位置，逐层gate/down的hook只读取、不修改任何返回值。
- 原Gx、phi和h实际dtype逐层记录，不推定张量一定FP16。统计在GPU FP32；用原down权重在共同FP16 autocast下对抽样phi额外计算一次D(sign*phi)，总24*32个附加抽样Down请求，全计费。动态分数及二阶统计不是模型原forward的数值转换。
- QA等权计算每层d均值、d²均值/RMS及合并位置方差、h²/h_diff²/(d*h_diff)²均值；保逐QA标量、抽样位置、输入tensor与图像SHA，记录完整model state和全部buffer前后digest、参数冻结/无梯度、实际dtype、GPU allocated峰与进程时间。CPU仅普通数据/文件I/O、索引、标量收据，无CPU模型或优化器。

## 唯一判据与行动

若任层d RMS为0、h_diff或d*h_diff二阶量为0或出现非有限，则本固定分组不可用于拟定全24层功能对照，关闭不追温度/组/采样；若24层d都完全常数，则动态干预退化，关闭。是否常数由实际样本d的min/max比较确定，不能以FP32二阶矩相减的微正方差替代；方差仅描述、微小负值按舍入下界0记录。不以阈值搜索制造变化。其余情况只保留“干预非退化、静态尺度已确定”的实施资格，不能说信号有用、功能已承接或专家可删，也不证明loss梯度非零；任何训练仍须独立完整方案/源码审阅。

固定OLD任一真正空卡一进程，首次STARTED起600秒，包括加载、32前向、附加Down、统计与digest，故障/超时关闭、不静默重启。无CPU门、算子探针、模型结构修改、额外QA或F1评分。先发布冻结源/审阅/READY再激活，不碰基线/他人GPU。正负均写实；若本诊断过，下一决定是是否值得成对原Answer F1验证，而非填Ours。
