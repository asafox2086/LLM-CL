# 原分配器已接入，等待基线阶段保存

科学源发布 4bfe6a4b2b02e5e20bb73ad24b99ec3d5d2c4cd1；继承源绑定修正发布 1257ff2098d5bd062d31b44e06cada59415699ec。有效 BINDING SHA 0707c07082ae3f14febca80364a4643146bc62312452b0101af055ecbe8ac12d。固定科学五源不变；来源修正说明见 SOURCE_BINDING_CORRECTION.md。

OLD 独立科学目录 `/dev/shm/llmcl-task-direction-1084-20261027` 已通过输入/来源元数据准备；不能把此项当 GPU 通过。

原 CPU 外层成功接管为 PID1894645/start780419791/UID1084，unit `llmcl-pct10-task-direction-old-20261027.service`。基线 ROOT 不变，全部原 argv 与队列保留，0 GPU 信号。接管耐久回执在 OLD `.deployment/task_direction_activation_20261027/`；执行脚本及 admission 位于 `.deployment/task_direction_activation_sources_20261027/`。

unix1793105809 对真实仍运行的 PaRSP worker1855248/start780152670 提交单次阶段暂停请求，SHA8477bc34b7ec67301a9752ee4f85390bc7a2d8b14f4c0242392759441956d2ff。它将在当前阶段完整结束及 checkpoint 保存后退出，不是立即中断训练。研究结束进入分析时须恢复这份基线最新 checkpoint，不能复用旧 S1 恢复请求。

sleep60 秒后，unix1793105901 实查上述 CPU 与基线 worker 身份均匹配；PaRSP 完成 S2、正执行 S3。研究 SLOT 尚不存在，0 STARTED，模型预算尚未开始。此为时点事实，不是永久运行声明。后续通过 `.deployment/task_direction_observe_old_20261027.py` 检查真实身份/阶段；异常不得盲目重启。

独立审阅修复 verifier 对训练末 requires_grad=False 的错误预期，并补齐实际 helper/policy 来源。源码检查不等于质量结果；当前仍无本方案 F1 或显存实测，不填 Ours。执行预算、失败不重试与完整留存规则见 PROTOCOL.md。
