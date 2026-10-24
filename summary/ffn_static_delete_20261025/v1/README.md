# 固定8个FFN静态删除：实施与当前状态

**最新终态：实施／数值完整性失败，本固定点关闭，删除质量缺测。** OBSERVED_CURRENT实查unit failed/exit1，worker3758372不存在。故障发生在定位反向后的三项联合断言：BI覆盖24层、门梯度全有限、loss有限。未保存分项值，无法识别具体原因，不归因FP16、overflow、某域或学习机制。

实际已加载模型、合并q/v并保存S19完整父包1787591658bytes；至少一次FIT请求已执行loss.backward，实际QA／反向次数未持久化，不能按0或计划37计。尚无selection.json、实际删除包、DEV评价或COMPLETE；真实模型0DEV生成／0物理删除，CPU tiny删除另计。STARTED至失败退出观测239.761606秒是观测区间上界，不是精确RUN；前两次模型前实施、CPU门、运输、等待和整个原绝对窗口费用另保留，不把重叠时间相加。

独立research_review接受上述终态解释：不支持或否定FFN静态删除，不从联合断言拼门梯度机制，不改源重跑。原协议、两次前模型偏离和失败全部保留。本轮没有科研性能成果；下方“当前实查”仅为此前运行快照。

科学问题、冻结选择规则和正负出口见PROTOCOL.md，源码审查见REVIEW.json。原r200同时改变全24层矩阵，本点保留未删矩阵，只把8个FFN残差分支替换为无参数恒零支路；不改注意力/KV。设计完整权重节省11.704%，实际质量和GPU峰值尚未证明。父为合法q/v FIT-only轨迹，不能以其保质代替战胜最强基线。

## 已发生及不可隐藏的偏离

1. CPU小Qwen2删除、gate=1、严格重载、tie存储去重检查通过，耗时0.144478秒；独立审查发现FP16标量门梯度精度问题，改成FP32门乘法并增加FP16检查，必要完成门0.091518秒，两次监督3.562471／3.100955秒分别包含内部计时、不相加。两次均0真实QA，首次检查不是最终源码通过。
2. 首次worker3755090在original_train读取T02时缺文件退出，尚未加载checkpoint／真实模型。第二次worker3756096在38条计数断言退出，同样0真实模型／QA／定位反向；processor加载、数据检查和两次运输成本非零。attempt1_before_model和attempt2_before_model保留原STARTED/FAILURE/ACTIVATION，OBSERVED_FIRST保存首次真实终态。
3. 原paired隔离目录只为旧61条配对诊断运输部分数据；新调用需要全19 TRAIN及原1313 QA的图像闭包。原manifest/SPLIT所需1113文件全部验SHA，补1049个缺失的原字节链接；不改既有文件，不选新图像。ASSET_REPAIR_DEVIATION明确承认一次模型前运输例外，不抹去首次失败。
4. 静态DATA_GATE确认T9仅1条FIT，其余任务至少2条。**原哈希排序[:2]已唯一确定37条**；不换题、不扩大数据。独立research_review修正先前关闭判断，接受仅改断言、计数38→37并绑定实际37ID，域平均/贪心规则、8层幅度、全部DEV请求均不变。COUNT_STORAGE_AMENDMENT显式记录与原38条计划的差异，T9仅1条定位的不稳定性保留。READY_INITIAL保留最初准入，不追改原PROTOCOL冒充零偏离。
5. OLD账户硬配额700GiB、实际701GiB，虽然整盘还有2.5TiB空闲，写目录仍Errno122。没有删除旧数据。本实验代码与输出使用RUN_LOCATION所列独立/tmp盘，ROOT/REPORT/WORK仅作存储路径参数化；全部数据及权重仍绑定原位置，核心数学规则不变。临时盘不是永久成果存储，完整产物须及时转存NEW后核对，不能把/tmp留存说成可靠归档。

第一次STARTED unix1792859398.6338136仍为总预算起点，绝对截止1792870198.6338136（北京时间2026-10-25 03:29:58附近）。当前unit RuntimeMaxSec只取该截止剩余量，不重获10800秒；首次两次实施费用/等待均占原窗口，不延长。第三次执行STARTED仅是执行记录，不是新科学预算。

## 当前实查

OBSERVED_CURRENT：worker3758372/start755838143/UID1084/PGID3758372，OLD GPU1，unit active/running，GPU已有约1960MiB，selection_ids.json已生成，尚无COMPLETE/FAILURE。说明已到加载阶段，不是质量成果；后续必须重新核真实身份与证明。

最大工作量为37次标量门定位反向，0优化器/训练更新；最多6完整图×100原DEV的NLL和原Answer F1，另每图五个固定输入恰32新token的GPU内存窗口，最多630新生成。若两定位集合相同，不重复图，按实际请求计费。模型包真实不含被删FFN并严格重载，才谈实际减重；GPU allocated/reserved固定窗口与全流程/构造/反向内存分开。

## 原刷新对照的重要故障

STAGE_REFRESH_TERMINAL_OBSERVED证明原stage-refresh已failed/exit1，两个原PID均无，Refresh.log有Errno122 Disk quota exceeded。故障和EXIT收据写成0字节.tmp，queue仍running不能当活跃。prefix/Fixed完整保留，Refresh未完整，0新VERIFIED；不消费半份质量，不宣布任何刷新机制正负结论，不重启旧固定点。资源故障不是质量阴性，科研总目标仍未完成。
