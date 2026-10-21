# 个人电脑 CPU 最小兼容准入

这是一项工程准入，不是新科学实验、方法收益或部署基准测量。

问题：现有完整包仅在服务器执行，不能用服务器训练内存或适配器字节代替目标个人电脑成本。先确认已有 K61 dense OSFT 包能在本机原数值条件下恢复和运行，再准备最高点估计 PASs 的公平完整包，最终在同设备配对比较。此处不先重复五题性能测试。

干预：仅将已封存的 CPU FP16/SDPA 完整包迁至 Windows CPU；Torch 2.6.0 CPU build、Transformers 4.46.3 及其他主要依赖固定服务器版本。CPU 与 CUDA build 差异公开。独立私有环境，不改全局或 Codex 环境，不使用当前占用中的本机 GPU。模型两数值线程、interop1、BelowNormal；完整 FP16 参数保持，不新增精度或量化路线。

只复用原医学 TRAIN 校准请求 SLAKE:5162、原图和原输入/损失函数。teacher 已见本题；不访问 TEST/DEV，不生成回答，不更新参数。不更改原服务器实验、READY、协议或结果。

顺序：依赖安装与包传输各自保存记录；所有包文件及原 helper 源逐文件 SHA 与已有 prepare_COMPLETE 一致；fresh 进程完整参数 digest 精确；随后一次原请求 NLL。记录完整加载/请求耗时和本进程 RSS 峰值，但仅作为兼容成本，绝不称 TTFT、tok/s、受控加速或完整设备基准。

判定：digest 不一致、数值不有限、CPU FP16 不支持、安装/传输失败或超预算即停止并保留失败，不改 dtype、不盲重试。跨平台 NLL 若与服务器 0.6714380383491516 不精确，记录 PLATFORM_DIFFERENCE 并停止后续测量，不放宽旧精确门槛，不沿用旧质量。

预算/出口：依赖安装至多 45 分钟；传输至多 30 分钟；模型加载加一次 NLL 至多 30 分钟，单模型、0 GPU/反向/优化器。模型及运行依赖分别计完整磁盘字节，至少保留 15 GiB 磁盘余量。成功仅支持本机运行的下一步，不启动大轮 benchmark；PASs canonical 包需动态专家/路由与原精度的 fresh 恢复和功能核验后，才能与可合并 OSFT 一起做同机配对资源比较。失败则记录该 runtime 路线的缺口，继续既有质量研究。

官方安装依据：https://docs.pytorch.org/get-started/previous-versions/ 的 v2.6.0 Linux/Windows CPU wheel。Windows 无 PSS/USS 统一支持时填 NA，不记 0。
