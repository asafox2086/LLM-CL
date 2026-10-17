# 旧卡1%任务显存分配器对照恢复

用户要求充分使用五张1%卡。原OSFT在旧GPU0初始化188/196层后OOM，SpaRTA在旧GPU1第17原生任务前向OOM；日志分别有347/485MiB reserved-but-unallocated，并非已经证明任何分配方式都无法容纳。

在空闲旧GPU0实际小张量测试中，CUDA snapshot确认expandable_segments生效且无不支持警告。因此只给旧机队列增加`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，不改算法、精度、数据、训练预算或执行源码。23个源码SHA双机一致；两个CPU队列仅process重启，原四个基线GPU进程及GPU1研究进程startticks保持。旧GPU0/1各重新启动OSFT/SpaRTA的全19任务预检，仍须通过独立checkpoint重载前后等价与有限反向门槛，才能进入正式训练。

本次是有分配器干预的新尝试，不把原失败预测拼接入新结果；失败证据与状态备份保留。分配器改变应在性能／耗时报告披露，不承诺它必然解决后续更大图像或参数占用。实际状态读取队列及PID，不能把“预检运行”称为正式训练完成。

配置：旧用户服务drop-in `20-allocator.conf`；执行仍由原nohup服务与pipeline管理。详见[审计](ALLOCATOR_RECOVERY_20261017.json)。

## 实际结果与第二种分配器

expandable_segments实际生效；OSFT完成196/196层初始化，随后第一任务finite loss/gradient断言失败，未获准训练，已转24GB等待独立新预检。SpaRTA仍在第17任务OOM，虽reserved-but-unallocated降至121MiB，原有效分配10.42GiB仍接近上限；不将其称为已经修复。

另在空闲2080Ti实际分配／同步测试确认`backend:cudaMallocAsync`可用，随后只对未来旧机新进程设置该环境变量。PaRSP（历史默认分配器曾在第16阶段OOM、有1.11GiB未用保留）和SpaRTA各重新由原队列通过全预检，旧失败训练不恢复、不拼接。当前属于预检运行，须查看实际门槛结果；若仍失败，不盲目循环相同配置。

OSFT此次有限性失败的loss或gradient具体哪项非有限尚未区分，不据此断言算法无效或硬件一定导致溢出。既有四个GPU作业环境保持，只有未来新进程使用新分配器。
