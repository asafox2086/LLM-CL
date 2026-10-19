# 连续LoRA的更新几何：借鉴LoRA-RITE

目标是在少量训练机会下保留已有适配，改变更新方式，直接验证最终五域Answer F1。沿用同一个rank8 q/v LoRA的A+B，不在任务边界清零、合并、增加专家或路由。借鉴[LoRA-RITE，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Abstract-Conference.html)，这是已有优化器的持续学习适配，不宣称原创或复现作者完整训练配方。

作者PyTorch参考代码固定在`code/LoRA_RITE/source_manifest.json`的revision，源不改动；作者README说明其配方与原JAX论文实验不同。学习率固定使用共同1e-4，betas(.9,.999)、eps1e-6、作者默认skip1/cap0/escape关闭/balance关闭。按作者要求只在优化器内对去尺度梯度clip1，不再额外对因子梯度clip。真实跳过数记录；若整步无有效更新则停止留证，不隐瞒为正常训练。与AdamW比较是整套优化规则的对照，不能将任何收益唯一归因某一个细节。

与原native SeqFT共享1313冻结TRAIN、19原生任务、93主更新、seed42、1epoch、batch16/micro1、floor-warmup/cosine、BF16基座与FP32 LoRA、dropout.1和原生AnyRes。每阶段保留当前A+B，优化器状态按原顺序基线在阶段边界重置。图片／问题／原答案／唯一原始Answer F1不改。没有TRAIN重放、检索、语言转换、选项映射或额外答案提取。

单候选必须完成完整19阶段训练与完整21454最终确认，不依155诊断决定是否继续。不按TEST选择学习率、阈值、初始化、停止点、容量或方法。初始500NLL／155生成必须精确原native `base_initial`；完整确认前新进程精确重载最终A+B并重测同500／155，再从头全测21454，500已看／其余20954分开。缺合法阶段格时完整FWT/BWT为null。native结果不填正式V2 Ours。

保存20个阶段A+B、每步原始TRAIN梯度及前参数、loss／真实LR／实际deltaW／作者跳过数、优化器张量字节和逐题输出。每阶段记录同PID CPU RSS/PSS/USS、GPUallocated/reserved/NVML前后与纯TRAIN峰值，进程总窗口另报。540672训练参数和固定一个专家不等于总内存更少；作者低秩基底、动量、协方差、临时候选和额外分解计算全部计入。

CPU真实TRAIN梯度反事实显示A/B等功能缩放会改变AdamW更新，而作者RITE在测试的二进制缩放下相同。这是已知优化机制的本权重局部验证，不是新模型前向或F1因果证明。首个GPU任务仍须原六真实图文门槛（包含已有FinMME:5082哨兵），不能称全进程从未读取TEST输入。

仅在现有GPU自然结束、原12格TRAIN-only更新对照之后，由唯一研究CPU接续；所有旧完整确认与负条件保留。实际空闲新GPU0/1择一、一个研究流、新机我们最多3卡，两机基线不停。准入后源码与协议冻结；故障保存证据、独立版本修复，不热改或盲重试。准备／CPU检查不当作已启动GPU或性能收益。
