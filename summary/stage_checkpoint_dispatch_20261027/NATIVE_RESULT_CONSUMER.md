# 新保存版本的完整结果核验

`analysis/stage_checkpoint_native_complete_verify_20261027.py`是单次调用工具，覆盖新1%和10% native runner。两机已复制相同源码；不注册新服务、不自动评分、不修改旧主表或旧proof。

它固定复用原消费者的评分函数及完整检查：19阶段、74格、80316条已有回答、最终21454题，1%93次正学习率更新，10%828次调用（822正学习率、6次合法零学习率）。仅适配新目录、来源绑定及评分调用计数。最终完整checkpoint流式核SHA，不加载模型。

新运行需要单独的binding，包含`condition`、固定新`frame`、`sources`、`policy_sha256`、`build`、`ready`、`completed_queue_snapshot`；10%另需`terminal_exit`。文件引用为`{path: ROOT相对路径, sha256: ...}`。1% BUILD来自该runner的BUILD_RECEIPT，READY来自NEW_ADMISSION；10%使用总BUILD与stage-checkpoint READY。完成队列必须另存稳定快照，同时核活队列同一run的终态。真实最终退出码缺失时拒绝，不能把None当0。

命令参数为`--root ABS --run ROOT_REL --host old2080|new3090 --binding ABS --output ABS [--write]`。默认只检查；只有显式`--write`开始评分。输出使用独立的condition/run散列目录，评分前独占创建VERIFY_STARTED，失败保留记录、不可自动重试；已有匹配COMPLETE时读取原proof而不再评分。同一run沿用同一输出目录，禁止换目录重算。

允许按人类授权清理旧中间权重，但要求保留全部19阶段的checkpoint元数据、阶段记录及最终完整权重。暂停链核对原请求、保存收据、显式恢复许可、源与配置，以及更新日志不重复。原暂停收据没有中间PID/真实退出信息，因此不声称核验了这些未记录事实，也不认证跨暂停进程的内存峰值。

源码SHA：`1434c424dcc7d003494da33054bf6ef823c3b49d7e3710805ff360856bf80cc2`。作者与独立子agent审阅完成；root复现了原AST装载、假整数计数器、假checkpoint暂停/恢复及缺permit/缺final拒绝检查。没有读取真实QA、调用评分或加载模型。尚无新版本完整run供实际端到端验证，不能把静态通过写成真实实验通过。

旧消费者继续负责旧运行。两个比例FOREVER由独立的[保存版FOREVER消费者](FOREVER_RESULT_CONSUMER.md)核验，不包含在本native工具中。
