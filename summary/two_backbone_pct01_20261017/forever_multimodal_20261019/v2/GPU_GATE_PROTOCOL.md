# FOREVER v2 接入修复门槛

继承 v1 的 [全部接入规范](../v1/GPU_GATE_PROTOCOL.md)：共同 1313 TRAIN／19 阶段／21454 TEST、原始 QA、图像 401408／缓存128MiB、FP16／共同 LoRA、主训练一轮 floor warmup／cosine 与独立作者两轮回放／warmup5／linear。无源、数据、LR、样本、训练次数、回放选择、正则或评价修改。主表和正式准入仍未完成。

唯一实现修复在回放训练完成后移除不可 pickle 的 AMP forward 包装，使用安装库明确提供的 `Accelerator.unwrap_model(model, keep_fp32_wrapper=False)`。不调用默认 True 参数，不复制或变更模型参数。原 v1 的失败源码、日志、partial checkpoint 和所有实际回放证据保留。

新增实际 GPU 门槛：移除包装前后同一模型对象、所有参数身份／存储／版本、训练参数 SHA 精确相同；同一条原始历史 TRAIN 的共同显式 autocast NLL 精确相同，前后恢复 RNG。额外 2 次 TRAIN 前向和 CPU 摘要计入该回放资源窗口，因此该窗口包括训练、保存清理与两次 NLL，不冒称纯模型训练峰值。

CPU 已复现原报错、成功序列化重载并验证参数／RNG／autocast 输出精确；v2 CPU 实际8次HF更新与原控制相同。GPU仍须新鲜模型通过原19 TRAIN图文正反向、额外T02一次更新、真实历史303条回放，再由另进程重载；两 backbone 均须通过。v1 预检权重不能复用，CPU通过不等于GPU成功。

调度仍只旧 GPU0，空闲与他人进程检查保留；旧CL/PAS/Spa和新CL/SMo／研究进程不停止。旧 v1 CPU 与 GPU 已以失败退出，不恢复、不盲重试；新的 v2入口、源码与本协议分别由新 `GPU_GATE_PLAN.json` 绑定。正式训练及发布需要后续明确准入，不能因真实回放已经完成而跳过 checkpoint 门槛。
