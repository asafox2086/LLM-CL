# 原生响应必要性先导：CPU准入未通过，真实实验未启动

## 问题

原协议要比较同一紧凑载体的共同q/v锚Q、新增FFN专家包E和原生输出重组R，检验某些新增权重是否可以避免。它是参数必要性研究，不是已经承接PASs的RW/RS或已形成新框架。进入真实模型前，托管E与作者直调参考的必要一致性检查必须通过。

## 实验与结果

六源已独立静态审阅并发布。固定12项资产通过PC默认TLS取得，官方LFS/Git blob、旧配置SHA和safetensors头核验后，经验证SSH传NEW，12文件SHA全部匹配。PC只作资产缓存；两次NEW网络/证书运输失败日志保留，没有修改CA或关闭TLS。头部507482304 BF16系数是序列化信息，不是活体模型去重计量或质量证据。

v1 CPU门在tiny模型构造时因视觉不支持全局SDPA失败。独立v2修复固定视觉eager/语言SDPA，不搜索backend，不重下载或覆盖旧源/旧门。

v2唯一CPU进程877717已exit1。tiny构造和实际attention配置断言已越过，E初态对父模型logits及参考初态state digest检查已越过；随后作者直调参考与托管E的参数梯度精确一致断言失败。Q/R小图路径尚未进入。没有CPU_GATE通过记录、READY、ADMISSION、研究service或真实载体STARTED；0原QA曝光、0真实0.5B模型训练、0GPU模型工作、0新AnswerF1。

原CPU_GATE_FAILURE.json中“raw gradient”标签需在此收窄：失败比较实际位于**backward及作者post_backward之后、clip与当前optimizer步骤之前**。若在第二阶段，梯度可能含RS手加项；具体stage/step没有封存，不能仅定位为autograd或第一步。原sealed JSON不改字节。

同一迭代先前的loss标量与RNG相等断言通过，但loss与梯度的有限性检查位于失败断言之后，未获证实。没有失败参数名、差值/有限性、实际saved tensor布局或精确已完成synthetic更新数；TemporaryDirectory已由原上下文清理。不能补称零更新、四更新、微小末位差或实际NaN。每条E路径此前优化调用仅由循环上界约束在0–3，精确费用缺测；CPU准入时间也未完整落盘，不以SSH耗时充当CPU RUN。

## 一次静态反向契约核对

主/子agent只读既有六源、NEW当前Torch saved-tensor实现与原作者forward/callback，没有模块执行、张量/QA或旧评分重跑。

确定的执行差异是：左侧Hosted包含save_on_cpu(pin_memory=True)和禁autocast上下文；右侧参考直接调用原父类forward，绕过这些上下文。当前Torch pack按size/dtype/layout重新分配再copy，没有显式传入原stride；作者a路由使用A_e.t()视图。因而不能先验假定两路的反向保存表示和完整数值路径相同。但未记录本次实际保存布局及失败参数，**不能把它认定为此次失败原因**。

左侧还对梯度作scale=1除/乘，右侧直接调用learner；在这次小图尺度的理想数学下没有新正则目标。两路fresh构造、作者专家别名建立、阶段reset/tau、record→forward→recordoff→backward→post_backward，以及clip/Adam/scheduler/tau和阶段历史更新顺序同源。未找到足以解释该失败的确定数学或状态错位。当前optimizer观察器尚未执行，不能用当前更新后的落盘解释之前的比较；此前步骤未知。

loss标量/RNG相等不等于所有logits、隐藏状态和反向表示逐值一致；有限性缺测也不能反向推断确有非有限值。根因保持未解。该核对已经完成，不重复它或重门来取得缺失数值。

## 结论

本次实施未取得进入真实实验的资格。不能宣布CPU观察器数学精确、原生载体质量通过、专家可避免、R胜出或任何压缩质量阴性；也不能把实现/运输/参数头当科研收益。原科学问题未被这次失败推翻，当前实现路线关闭。

## 行动与效果

保留v1/v2所有源、review、marker、资产receipt和私有日志；不放宽容差、不重门、不滚动v3、不激活E/Q/R、不恢复旧视觉/PAS/r200/ShortFixed。两机正式基线与其它GPU作业不受影响。

效果是阻止一个尚未通过必要反向一致性检查的参照进入昂贵实验，并明确缺测范围；没有新的质量—权重成果。后续研究须形成能改变判断的功能承接或参数必要性问题，不能以更多兼容版本、重复本文静态核对、泛读或台账维持活跃。科研总目标仍未实现，不应标完成，也不存在等待此CPU/GPU作业的资源阻塞。
