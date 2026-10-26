# FOREVER 10%：待空卡的具体接续

本次仅本地源码／小证明审阅与新增 stdlib 监督文件，未连接服务器、运行模型、CPU 模型门或评分。原 core、adapter、run、pipeline、policy、BUILD_RECEIPT、已运行 native 源及其队列均未修改。

已经完成的部分不重做：独立动态 runner 的 metadata build 已封存；`preflight_replay` 已实现 385 条 T13 TRAIN／25 次 current 更新、原历史选择、自然 dynamic@24、继续 current25、final 回放、外层 optimizer／scheduler／scaler 状态检查；原 run 后接 T19 TRAIN probe 与完整 CP 保存，原 checkpoint_probe 负责 fresh-process reload／有限 backward。此前 S1 core 未释放的集成错误已在源冻结前修正。它们尚未实际在 GPU 上通过。

新增文件：

- `exp/two_backbone_pct10_20261025/forever_gate_supervisor.py`：一次首次 2400 秒绝对预算，顺序调用原 `run.py --method FOREVER --condition pct10 --preflight-only` 和原 `checkpoint_probe.py --run-dir …`。不重写训练数学或 probe；失败不重试。输出 `GATE_STARTED`、两子进程 STARTED／EXIT、`GATE_FAILURE` 或 `GATE_VERIFIED.json`。核原 19 个诊断、385/25、dynamic→current25→final、完整保存／重载、实际回放曝光和源链。原 probe 删除 disposable CP，正式阶段从 fresh 另起。原 peak reset 导致整体 GPU 峰缺失明确记录，不造完整峰。
- `exp/two_backbone_pct10_20261025/dispatch_dynamic.py`：导入原 stdlib dispatcher，仅替换 FOREVER 目录及其预检／重载调用，设置原模块 `__file__` 指向新 wrapper，故后续子进程也进入同一路径。预检作为原分配器的占卡任务运行完整 gate；原 ready_reload 槽位只消费已经完成的独立重载证明，0 新模型，避免再次重载。训练仍调用原 pipeline.launch 的 fresh 路径。native phase、ownerless／free／util guard、队列格式和原 launch 不变。两监督源均置于 coordinator 目录，动态 runner 的 `*.py` 集合及 BUILD_RECEIPT 原字节不变。

首次 CPU 交接前由 root 准备 `summary/two_backbone_pct10_20261025/FOREVER_DYNAMIC_DISPATCH_READY.json`，必需字段：

```json
{
  "original_dispatch_sha256": "原 exp/two_backbone_pct10_20261025/dispatch.py 摘要",
  "dynamic_sources": {"每个新动态目录的 *.py 文件名": "最终 SHA256"},
  "supervisor_sources": {"dispatch_dynamic.py": "SHA256", "forever_gate_supervisor.py": "SHA256"},
  "dynamic_policy_sha256": "原样独立 policy 摘要",
  "build_receipt_sha256": "原 BUILD_RECEIPT 摘要"
}
```

该 READY 只绑定静态源／政策／协议，可事前发布，不为队列变化反复改写。原 CPU coordinator 停止后、所有 GPU worker 保全，由 root 一次生成独立 `FOREVER_DYNAMIC_HANDOFF_INPUT_old.json`：

```json
{
  "prior_controller": {"pid": 0, "startticks": "实际值", "uid": 1084, "pgid": 0},
  "initial_queue_sha256": "停旧 CPU 后 queue_old.json 的实际摘要",
  "ready_sha256": "已冻结 FOREVER_DYNAMIC_DISPATCH_READY.json 的摘要"
}
```

再以同原环境和 `LLMCL_ROOT` 运行 `…/.vision-env/bin/python -u …/dispatch_dynamic.py --machine old`。首次交接核该输入的 READY 摘要并在 handoff 收据记录输入 SHA，独占保留原 queue 字节，只将两项 `waiting_forever_pct10_scheduler_review` 改为 `waiting`，其余 job 记录逐值不动。初始 queue SHA 只用于这一次交接，不是以后启动永久条件；已有 handoff 收据时不再读首次输入／旧 queue SHA，也不再次解冻失败任务。新代码不会向原 GPU worker 发信号，也不会独立抢卡。应先核原 CPU unit 的停止行为及进程身份；代码本身拒绝原 controller 仍匹配时交接。

下一空卡前仍需实际完成：① root 将原样动态目录、两监督文件及原 helper source 运输到 OLD overlay 并封摘要／准入；② 按上述意图做唯一 CPU coordinator 交接；③ 每个骨干由同分配器在真正空卡上执行各自一次 2400 秒 gate，只有全部原 proof 与 GATE_VERIFIED 成功才转该骨干 fresh 正式训练。一个骨干过门不冒认另一个已过，失败不自动救跑或续 diagnostic CP。当前没有任何新的 GPU 门结果，不能说已获模型准入。

完整结果链仍缺**独立 FOREVER 10% completion observer**：native observer 明确仅 14 组，固定旧 native FRAME／transport，不能冒认两 FOREVER。应沿原 raw token F1 锁，对新正式 run 的 19 阶段、828 current 调用（822 正 LR／6 合法 warmup 零 LR）、全部实际 dynamic／final replay、74 cells／80316 QA／21454 final TEST、S19 rolling commit 和最终完整 CP 做一次缓存式核验；不沿用旧 1% 的零动态触发／固定 memory 数断言。该消费者不影响 native 现有 14 组，也不能先填正式表。

已核实 archive 无需重做：`analysis/pct10_final_model_retention_20261026.py` 当前 SHA `cbb47ae4545db749cb9ca4b26a68f6d38ce3f4bbeb772ae9c6cdf10d7c387192`，其 `load_base` 已加入动态 FOREVER 目录，并保留完整 final graph 的 SHA-checked hard link；模型质量核验仍独立。滚动 latest／final 留存不等于所有阶段 CP 耐久。

因此，缺口是合法实际 gate、受控 CPU 队列接续及新 FOREVER 正式消费者，不是再写一次动态训练算法。两项 SMoLoRA 的 NEW BF16 槽位交接是另一个已知调度条件，本补丁不处理它，不声称此刻 16 组同时运行。
