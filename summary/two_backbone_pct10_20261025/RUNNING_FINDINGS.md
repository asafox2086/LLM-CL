# 10%主实验：实际启动与结论边界

本次要检验的是：原1%环境下观察到的新习得不足、方法相对次序和最终质量／内存取舍，是否在更大训练集仍出现。新增主表不是新方法成果；尚无本10%的完整性能结论。

已冻结TRAIN13105／完整TEST21454、19任务、seed42、单轮828次当前任务更新。1%题目逐task包含在10%中，TEST原文件不变；所有组合fresh预训练开始。增加数据也增加单轮更新预算，因此两表能检验设置推广性，不能单独识别“数据量”与“优化步数”的独立因果。

## 已实际运行

- 预注册发布为`453f93a93a21d8042d27f472d2c1a1f6438d6901`；原1%正式主表整块逐字节保留。
- `LATEST_OLD_OBSERVATION.json`的unix1792938818.253909快照：OLD四卡各一组正式运行，LLaVA／Qwen的SeqFT-LoRA及PASs-MoE；四组GPU预检和独立重载均成功退出，正式phase处于stage0初始完整评价，尚无完整19阶段结果。实际GPU进程身份见该JSON，不把本时点写成永久live。
- 其余八个native组合排队。两SMoLoRA仍待NEW原生BF16硬件；NEW原1%队列继续，GPU0不由本队列分配。
- 两FOREVER在任何模型尝试前明确暂缓：旧核心只覆盖不超过24次当前更新，旧预检也假设S2仅一次更新；本10%分别为217／7。不能删除断言冒充原方法扩展，详见`FOREVER_COMPATIBILITY_FINDING.md`。

## FOREVER暂缓与CPU接管

原三冻结Python源及所有已运行native科学源不修改。已审阅并发布`defer_forever.py`，仅在原dispatch持原lock后读取队列时，把两个尚未运行的FOREVER改为明确等待。原phase入口、模型、损失、数据和现有GPU进程不变。

`CONTROLLER_HANDOFF.json`记录原CPU4044638→4049826：先核UID／startticks／unit MainPID与KillMode=process，再只向旧CPU发送SIGTERM，未向任何GPU worker或其wrapper发信号。原OneVision PAS重载worker4049282已在unix1792938739.8661385自然exit0，早于handoff，后续正式worker4049854是正常进入下一phase。其余已活跃正式worker接管前后身份相同。旧unit终态不代表GPU实验停止，新unit以实际proof为准。

早期SOURCE_REVIEW接受了入口／数据／接口扩展，但未识别FOREVER更深层1%假设；后续收结果路径审阅发现后立即在首次FOREVER模型前暂缓。保留这项审阅遗漏，不把先前静态接受当所有16方法科学路径已验证。

## 结果消费与资源

正式训练完整退出后先标`completed_unverified`，完整19阶段／74格／80316请求、逐ID原评分、源码／数据／有效更新及S19完整权重校验后才填表。新习得两列取原74格对角，不新增测试请求。原1%硬编码消费者不能直接用于10%；新条件的独立核验配置还需实现并审阅。

完整系数／dtype、去重权重与必需状态、CP容器和固定窗口GPU峰分别报告。没有10%实测就留空，不抄1%数值，不把训练峰或小文件当部署峰。

OLD新输出位于RAM tmpfs，不能称耐久副本。NEW收集器已运行并完成小证据同步；只为各run保留最新已闭合CP／最终CP的一份校验副本，旧源端保留策略不变，不能声称SeqFT19份阶段CP均已耐久归档。第一轮尚无完整阶段CP。归档是必要数据保存，不是论文收益。
