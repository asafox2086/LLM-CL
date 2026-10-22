# 同条件独立v2：首步梯度保存修复后实际训练

原v1因首步保存目录缺失在0optimizer/0DEV处失败，详见[失败记录](../v1/FAILURE_FINDINGS.md)与[修复边界](REPAIR_SCOPE.md)。不是质量阴性；失败加载/门槛/16QA前后向成本保留，文件mtime区间55.937秒不当完成的RUN测量。不恢复或拼接v1。

五源及PROOF/PROTOCOL逐项审阅接受，ShortFixedOSFT/optimizer数学AST不变。必要小HF准入3.648578秒，从不存在的run/gradients经worker共同保存helper写入并严格重载键/dtype/梯度值；初态/共享梯度/重冻/投影及HostAdamW原门槛也通过。READY SHA33cc1c4ad7a4b3e909fefb9a23be8dc0f941645bb91815118a2d6b751bd22d68冻结，原作者核与科学条件未改。

2026-10-22 03:45:09UTC，唯一v2服务llmcl-onevision-osft-short-factor-control-20261022-v2.service实际active，CPU7045/start90261，GPU1 worker7170/start90779，新boot f801c9c5-7362-47c3-afc3-1454c1e9d29d。19FIT门槛完成，原TRAIN已有2条更新事件；首步完整梯度文件358047050bytes已写出，越过原故障点。尚无完整F1，不读partial DEV，不把本快照当后续live。

同一1213FIT/100已探索DEV/19阶段87更新/原LR1e-5/原FP16外层与FP32 USV，真实布局168模块89485056训练坐标，保留全部USV存储。原质量与成本判据、完整核验、2h边界、失败独立版本及不自动PAS/1313/21454均不变，详见[协议](PROTOCOL.md)。当前只工程恢复与实际训练启动，Ours/正式表不变，端侧仅背景，科研目标未实现。

正式队列重启恢复见[恢复记录](../../reboot_recovery_20261022/v2/FINDINGS.md)：当前CL4962/SMo4874保全，CPU5963、publisher5965、FOREVER等待5967；PAS/PaRSP旧attempt无权重，保留历史后原条件新鲜排队。旧机完整作业不恢复。上述身份仅该次恢复证据，以后看实际进程与proof。
