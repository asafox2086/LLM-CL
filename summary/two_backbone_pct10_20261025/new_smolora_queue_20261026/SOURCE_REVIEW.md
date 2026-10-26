# 独立源码复核

结论：接受这次 CPU 调度接入，未发现必须修改的接口阻断；这不是服务交接或 GPU 预检已经通过的证明。

- 包装器：`analysis/baseline_pct01_pct10_capacity_20261026.py`，SHA256 `28614141a9f75bb4fdc5c2762993004fe0792a6968776b2207e9fd5d69d732c9`。
- 交接脚本：`.deployment/pct10_new_capacity_handoff_20261026.py`，SHA256 `d2bcc323eb19c8c5202e259f72ee321148b678c334ab7c7c3ba7d288485e8426`。

实际读取上述两源、原 capacity/dispatch/pipeline 和已取回的 NEW 小型源码/BUILD/policy 元数据。NEW BUILD 的 native target 确为 `/data1/liyapeng/program/LLMCL/exp/two_backbone_pct10_native_20261025`；其 25 个 `.py` 的 basename→SHA 映射与已取回 native_sources 完全相同，原 pipeline.sources() 正是同目录同后缀映射，故请求和两份预检 proof 使用的源字典口径一致。原 phase-request 在默认队列主循环前返回，调用原 pipeline.launch；训练/预检仍走原 run.py，保留原 SMoLoRA native-BF16 guard、dtype 和数据配置。

原 capacity 主循环、原四把锁、HOSTS、1%同一内存 states 对象及其优先级保留。新增两项只通过原 idle 机会进入 GPU1/3；GPU0 和原 GPU2/FOREVER 规则不扩占。10%三阶段分别保留真实 wrapper/worker 身份与 EXIT；缺 EXIT 或疑似活 worker 保留占位，不自动恢复或重试。没有调用 `--machine new` 默认创建16项的路径。

交接脚本绑定原 CPU PID/startticks/UID/PGID、原 unit MainPID、KillMode=process、Restart=no 与空 ExecStop，仅替换该 CPU 服务入口；前后核原 GPU worker 身份及科学源码/policy 字节，未向 GPU worker 发信号。实际交接成功与后续 phase 成功仍须看执行收据。

仅做源码、已封小型元数据与 stdlib AST/SHA 检查；未运行模块 main、服务、服务器命令、模型或评分。两项完成后仍需受限发现入口复用原单 run 正式核验；当前固定14项的 native discover 不能直接用于仅两项的 NEW 队列，此缺口已披露，不能提前填表。
