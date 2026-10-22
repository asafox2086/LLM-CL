# 独立v2：只修首步梯度文件目录依赖

v1在真实STARTED后加载原模型、完成19FIT图文门槛及首批16FIT图文前后向，首个raw梯度torch.save因gradients父目录不存在停止。原因是READY前删除了先写raw JSON的重复语句，同时无意删除了其隐式建父目录的作用；此前静态/小图门槛没有覆盖真实记录路径。这是实现错误，不能用它判断固定短侧的质量。完整源、READY、STARTED、FAILURE和日志原封保留；V1_FAILURE_SEAL绑定当前完整失败artifact。

v1没有clip/optimizer更新、训练事件、checkpoint或DEV生成，但加载/38NLL+38短生成及16QA前后向都有成本，不能称零耗费。文件mtime时间区间另列，不冒充ResourceMonitor完成的RUN。不得恢复v1、拼接状态/梯度/预测或删除失败。

v2仅新增共同的gradient_directory/save_first_raw_gradients路径处理，worker在run建目录时显式创建gradients，首步保存也经共同helper确保父目录存在。CPU准入复用原tiny backward，从不存在的run/gradients开始，经worker同一helper保存，然后严格重载核验全部梯度键/dtype/值。checkpoint/validity及JSON路径已有创建逻辑，不追加无关门槛。

新REPORT/private/READY与systemd unit为v2独立身份。原ShortFixed类训练/投影/重SVD数学函数、19/87 TRAIN AST、原基座/FIT/DEV/seed/LR/数值/停止判据/原score/质量与成本口径全部保持。v2从原基座重新执行唯一同条件先导，v1历史费用另计；不是追加科学条件、按DEV追分或放宽门槛。原PROOF/PROTOCOL逐字节继承，原作者核和正式基线源码不改。

机器又发生了独立重启：当前boot f801c9c5-7362-47c3-afc3-1454c1e9d29d。v1目录错误在旧boot发生，不归因重启；旧PID不是存活证明。正式队列保全当前CL/SMo，PASs/PaRSP旧中断attempt无可恢复权重，保留后新鲜原条件排队；发布器/FOREVER CPU按核实身份恢复。研究仍只GPU1一流、最多三卡，不用GPU0/旧卡；新源/READY冻结后失败须再次独立版本，不热改或盲重试。
