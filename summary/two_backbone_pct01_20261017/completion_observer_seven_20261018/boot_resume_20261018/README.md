# 重启后恢复只读正式结果发布器

2026-10-18实查旧服务器OSFT原运行已完成19阶段，原PID572431正常退出。新服务器原七方法发布器为inactive/Result=success/MainPID0，无needs_attention、锁空闲；只恢复原冻结CPU watch服务，未改训练／评分／路由或重启GPU。

新CPU51266核验通过3个方法－骨干完整结果：原SeqFT／PaRSP及新增OneVision OSFT。每格均核验原19有效训练阶段、74必要评测格、21454最终QA、逐题原Answer F1及checkpoint SHA。外侧README仅由原冻结发布器更新已核验行；研究155结果不进入正式表。

[前状态](BEFORE.json) · [保全进程及服务激活](ACTIVATION.json) · [逐题核验索引](../VERIFIED_INDEX.json)。PID为快照，服务和实时完整证明才是后续依据。未恢复任何旧研究调度。两机正在训练的五个基线保全。
