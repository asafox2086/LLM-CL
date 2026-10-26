# 队列接入评判

结论：可通过一个新的外层wrapper继承现有position→dynamic→native/FOREVER链；不需要第二个独立分配器。不修改原position四源、原分配器、科学训练源或已有任务记录。本文是本地源码评判和新wrapper实现说明，未执行交接、模型、评分或服务器访问。

## 已核接口与根目录

- 原coordinator root保持 `/dev/shm/llmcl-pct10-1084-zu7e3hb_`，继续使用原 `queue_old.json`、`dispatch_old.lock`、position admission/slot及native/FOREVER任务。
- 科学worker root独立为 `/dev/shm/llmcl-conditional-1084-20261026`。新wrapper本身置于此root的 `exp/conditional_shared_budget_20261026/v1`，只有其研究controller子进程的env/cwd切到科学root。父进程 `LLMCL_ROOT` 始终是原coordinator root。
- controller实际CLI为 `--root SCIROOT --output ABS --gpu N [--python ABS]`，六phase串行，同一supplied GPU，不自行分配；终态提供 `SUPERVISOR_COMPLETE.json` 或 `SUPERVISOR_FAILURE.json`，失败含 `last_attempted_worker`。
- 原position wrapper先调用原pipeline的idle，再独占一次position slot；原dynamic层继续处理FOREVER专用preflight/reload与native原路径。子phase的 `__file__` 重入仍沿原position入口，不必复制或重新包装其科学源。

## 新外层的最小职责

新 `queue_adapter.py` 加载SHA绑定的原position wrapper，对其 `Slot.wrap` 作外层组合：先获得完整原position idle链，再包唯一conditional槽。原position slot尚未真实 `released=True` 时，不允许conditional claim；因此position排队或保留期间不占第二张研究卡，其余卡仍供原基线。

position释放后，仅原idle判断为可用的同一张卡可以触发新controller。claim先以exclusive文件落盘，再Popen；六phase间CUDA空档也继续保留。没有新增GPU空闲阈值、旁路GPU搜索、信号已有GPU进程或另改native/FOREVER队列。

release必须同时看到唯一终态、原controller身份已消失、全部已记phase与失败的last-attempted worker身份均不在、对应NVML计算PID均不存在。完整和失败终态均可释放，但一次claim账本保留，绝不重启。缺终态或身份不确定只保留原槽，不将它重新分配。

实际原 `dispatch.main` 只在有waiting/ready基线job时调用idle，而且所有基线终态后直接return。因此仅包idle不足以保证研究任务被启动或继续监督。新wrapper在原main**正常返回后**，取得同一原 `dispatch_old.lock`，继续调用已经捕获的同一pipeline/position idle链，沿原OLD卡序与8500MiB阈值完成pending-slot收尾；原main和收尾不并行，原main异常不启动收尾。没有同时存活的第二分配器。

## 最短CPU交接

root先发布并逐字节复制新wrapper及已定研究源，准备独立output父目录；不覆盖原position source/admission/slot。交接前按实时queue核原CPU coordinator的PID/startticks/UID/PGID、原unit的KillMode=process/Restart=no，并保存原四卡正在运行的worker/wrapper身份及position slot/controller身份。旧脚本中的历史PID和“slot不存在”断言不能直接复用。

只在原CPU coordinator处于循环等待、没有进行launch/queue写入时终止这个精确CPU身份；不停止其worker/position controller、不向GPU进程发信号。确认旧CPU身份消失后，启动唯一新CPU入口，继承同一baseline root和position admission。原动态handoff已有receipt只核原来源，不重新改FOREVER状态。

新入口取得原锁并采用原queue。交接后对照原worker身份和slot账本，允许自然完成并保留其原退出证明，不能把自然退出误写成GPU被停止。新unit仍无自动重启。异常只检查这一个CPU身份与已有槽，不额外启动另一个分配器。上述动作由root执行；本审阅没有实施这些操作。

## 已审源与参与披露

只读检查了原position `queue_adapter.py`、`controller.py`、`POSITION_RUN_ADMISSION.json`、原dynamic/dispatch，以及本地历史position handoff脚本的可执行接入部分；历史脚本中的嵌入运输payload未执行。原position wrapper SHA为 `23a7a6fff3f33766a22d636c632476008560023c62da1b0c504f7b9913a9f833`，原dynamic SHA为 `6c5006c6ece0df9550b30eabbe1949ac50a89df4ca7ff3ff0aec656bb4cdc690`。

review作者同时实现新wrapper，需独立交叉审阅实际字段，不把作者自审当独立准入。新源仅作AST解析，未运行main或模型；其最终SHA由交叉审阅后绑定。此接入不证明方法有效或GPU更省，相关科学结果仍需原三臂完整运行和唯一评分。
