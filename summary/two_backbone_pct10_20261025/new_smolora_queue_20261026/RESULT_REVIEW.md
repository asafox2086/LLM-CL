# 交接结果与既有核验入口复核

接受本地收据支持的 CPU 交接结果。`HANDOFF_STARTED` 记录原 CPU 1221743/start26017836；`ACTIVATION` 与后续 `AFTER` 一致记录新 CPU 1739063/start35800472、unit active/running、KillMode=process、Restart=no。原 GPU worker 8260/start60876 与 8265/start60882 的 UID/PGID/身份前后一致。冻结 wrapper 仍为 `28614141a9f75bb4fdc5c2762993004fe0792a6968776b2207e9fd5d69d732c9`；两项10% SMoLoRA 在该 AFTER 快照均 waiting，不称已训练或完成。`ONCE_CHECK` 为 returncode0/no_launch，队列未改；无模型或评分由该观察启动。

交接脚本最终 SHA256 为 `17ccd721ac55a92a811c91f7791ab1ef8434d4316d2796926a784e291f984ea3`。实际反向撤销两处改动——空 ExecStop 取值改 `.get`、允许已观察到的 Python poll 等待 wchan——精确恢复此前审阅 SHA `d2bcc323eb19c8c5202e259f72ee321148b678c334ab7c7c3ba7d288485e8426`。两处均在 HANDOFF_STARTED/stop 之前，不改变调度 wrapper、训练、GPU 身份或科学设置。root 报告此前两次分别在这两个前置检查失败；不把它们写成模型失败或交接首试通过。

更正此前消费者入口表述：原 `analysis/pct10_verified_baseline_report_20261025.py`（SHA256 `2b2a3202fa049c911324081aacb31d0e7c7508b1020a2727893f1f85ab15c376`）已有可用的 `--host new3090 --mode verify --run <ROOT相对完整训练run> --write`。main→refresh(only_run) 仅 candidate 指定 run，随后对已有 proof 也逐 run candidate，不调用 discover；candidate 只读取相应 `backbone:method` 的真实 training EXIT0/完整终态。故两项队列完成后可直接复用该入口及原唯一评分缓存，无需新消费者或新的发现包装。固定14项索引问题仅影响不指定 run 的 discover/once/watch 路径。原 transport/source/数据/74格/完整CP 等断言均继续保留，不由此绕过。

本次只读本地收据与源码、做 stdlib AST/SHA 检查，未连接服务器、执行消费者或评分。收据 SHA256：ACTIVATION `54d3636dd6d8eae445866d3b9629dfa328343d5e20145e05d5ea8f4ab66a0345`；AFTER `ded1526705916f65686e923c364d0ea1de2b615bcd9d28bcc00101e29698beb4`；ONCE_CHECK `0ec3c9ccd29f8da410957d6eb9a6286e6c0f51fa1f2a612dfe1c0c8da27a05fe`；HANDOFF_STARTED `d2cbbf3ec8904dcaf1b18168281e281d5ae8141d91251629a0b9d8ba40c2b166`。
