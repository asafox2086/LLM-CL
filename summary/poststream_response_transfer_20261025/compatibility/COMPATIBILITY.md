# 一次盲态加载修订，不延原截止时间

原 v1 的 CE 已真实完成 320 次反向、20 次更新，未评价；KD 在 student 构造的 SigLIP `erfinv_()` 报 CUDA driver invalid argument，尚无 KD 训练、教师调用或任何新 DEV 回答／评分。故障点已知，CUDA 根因未被识别。旧源、READY、CE 包和故障全部保留；旧 CE 不参加任何质量比较。

两位子 agent 认可在当前人类自主研究授权内明确修订 root 自设的 first-failure/no-auto-version 条款一次；不将缺测当科学阴性，也不隐瞒偏离。仅使用已反复实测的标准 `from_pretrained` GPU 权重 I/O，随后完整覆盖同一 sealed 静态状态，避免初始化马上会被丢弃的随机矩阵。没有 CPU 模型 forward；不改精度、删图、memory、损失、rank、LR、训练步数或评价。

加载后必须匹配原 CE 初态所有持久和非持久 buffers／FP32 RoPE、alias、layout、attention，以及原 CE 导出包的 config／generation 配置。配置仅排除行政 `_name_or_path` 字段。原 CE 包只 mmap 读取配置元数据，不把其训练权重交给新臂。适配器统一在静态加载后 reset42 再初始化；这改变了相对旧废弃运行的初始化 RNG 流，两个新臂必须实际摘要一致，训练前仍 reset42。原 SeqLinear 前向、CUDA 数学和训练源码均逐字节保留，不重复 GPU operator 检查。

**不延时：** CE/KD 仍分别受原首次 STARTED+1200/+1800 秒的绝对截止约束，修复等待在内；各 eval 首次450秒。总3900模型秒包含原 CE 完整单调钟耗时和 KD STARTED→实际退出墙钟代理（不称精确完整 RUN）。外层原 activation+4200 秒也不延。剩余不足不启动，任何本次再失败或截止不足则终结此对照，不再更换加载路径或初始化重试。

只有两新臂均完整成功时，计划有效比较640次反向／40更新；连同废弃原 CE，实际总960次反向／60更新。当前不得把计划当已发生。独立评价前不读任何质量，200新评分、210新生成及正负行动门不变。修复不是科研发现，所有额外 I/O／实现／审阅和失败成本单列。
