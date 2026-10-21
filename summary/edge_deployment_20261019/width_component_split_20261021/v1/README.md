# 新CPU对照：分别检验视觉、语言半宽的功能损失

**状态：已冻结、实际启动prepare；两个新条件尚无完整F1结果。** 控制器2190288/start27210630，prepare2190313/start27211785，两次观察CPU时间增长。当前CPU真实完整参考NLL仍待门槛，不把准入或启动写成性能提升。最新状态应读服务器CONTROLLER_STATE及实际进程，本文PID为启动记录。

| 问题 | 已有结果 | 本轮新增干预 | 结果用途 |
|---|---|---|---|
| K61质量损失来自哪一支或组合？ | dense五TRAIN F1 47.088608，同时半宽fluctuation21.754386 | 只减视觉、只减语言两完整物理模型；继承相同父包参数、原mask/补偿 | 补齐固定2×2，决定这两个无恢复条件是否还值得未来独立DEV |
| 参数是否确实可以替代？ | 同时半宽包少31.701%但质量不通过 | 逐参数来源核对、fresh完整重载、原Answer F1逐题及包/RAM测量 | 无功能损失证据前，不称某支冗余或无用 |

不重跑原prepare/校准/三条件，不调比例，不换原5题，不用DEV/TEST选择。五题teacher已见；医学、金融参考原本为0，因此不能用这五题证明五域质量保留。只有五题逐题F1不低于dense且实际完整包减少，才把固定条件保留为未来FIT teacher/独立DEV候选；没有自动21454路径。

子agent研究评审暂缓PAS固定混合局部拟合，理由是局部误差不是质量、折叠通常只追平OSFT稠密基座成本；转而接受补齐两支隔离。代码复审在READY前补齐独立报告封链、旧源检查、参考整模释放断言与成本口径。[ADMISSION_REVIEW](ADMISSION_REVIEW.json)保存源与数学函数AST一致证明。小实际HF两条件移植、fresh状态及文本/视觉输出精确只属工程准入。

READY SHA7101cae0e5f23251a63c2097d85c2a4080c20eac38d7c6fd0111c47311beeaff。新科学源/协议、旧父包和预测、原helper全部绑定；故障保留并停止，不热改、不降门槛、不盲重试。prepare1次原医学NLL精确控制、两个case各1预热+5强制8token+5自由最多64token，顺序CPU、2数值线程/interop1/nice10、0训练/0GPU，预算prepare2h+每case2h。原/新时序不同，单轮CPU速度只是观察，不称受控加速、手机/GPU或正式端侧优势。

[问题、协议、行动出口](PROTOCOL.md) · [冻结证据](READY.json) · [真实启动](LAUNCH.json) · [实际进程与基线保全快照](START_SNAPSHOT.json) · [旧完整负结果](../../structured_width_20261020/v1/FINDINGS.md)。借鉴[FLAP AAAI2024](https://ojs.aaai.org/index.php/AAAI/article/view/28960)的功能波动/补偿视角，本轮是已有成本点的隔离干预，不是作者全局自适应方法复现或新算法。

原GPU1四条件队列、Qwen基线7965/181717和发布器51266实际存活，完全保留；旧K61完成worker不恢复。目标尚未实现，Ours及正式部署列空白。新结果只在两case COMPLETE与独立原评分报告完成后解释。
