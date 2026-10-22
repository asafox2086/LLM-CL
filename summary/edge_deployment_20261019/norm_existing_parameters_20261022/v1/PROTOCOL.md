# 已有归一化增益能否承接适应功能：单条件配方可行性先导

## 问题、依据与唯一干预

新增分支不是适应的唯一可能载体。ICLR2024《Tuning LayerNorm in Attention》的LayerNorm-simple仅更新已有归一化参数，提供功能替代的先例。[原论文](https://proceedings.iclr.cc/paper_files/paper/2024/file/7cc16e8635e6f27c295355bd214ef8d8-Paper-Conference.pdf)研究Vicuna/LLaMA2、更多数据与学习率选择，不能直接外推本Qwen五域持续学习的1213条/87步。这里只借鉴假说，不称原论文复现或新算法。

归一化增益改变进入冻结线性映射的通道缩放，因此可改变功能；不是任意矩阵更新，也不预设预训练能力已经足够。原LoRA_all完整确认负结果、同坐标分配与去保护包对照尚未解释强基线收益。本次不重开这些已结束配置，不据官方TEST选择归一化位置或学习率。

仅新增RMSNorm_only：24语言块各input_layernorm与post_attention_layernorm，共48处/43008个已有增益。冻结最终语言norm、视觉、projector、embedding、head及其他全部权重；不新增LoRA/专家。既有参数是否足够是待检验假说。固定LR1e-4沿原Seq配方，不称论文最优值或与LoRA相同功能步长；不搜LR/位置/比例。不是纯参数空间因果或匹配有效维数的实验。

## 数据、数值与证据

原冻结1213 FIT/100已探索DEV、每域20、19原生阶段/87正LR更新；原TRAIN AST、seed42、epoch1/batch16/micro1、floor warmup/cosine/clip1、各阶段新AdamW/.9/.999/eps1e-8/zero decay保持。T7/8/9/12无DEV，不能称19任务独立泛化。新鲜同基座，无teacher/蒸馏/回放，不开官方TEST，不改图像/问题/提示/答案抽取/全部原参考评分。

FP16基座/原外层autocast不变。仅增益的优化主副本用FP32，forward显式转回原FP16权重dtype，保持原RMSNorm归一化运算与输入输出dtype；主副本/实际forward权重/优化器状态分别公开，不是精度研究。小HF准入核验初态整体logits/RNG/权重值精确、其他参数冻结、有限FP32反向、完整小图Cloudpickle重载；不启动实际CPU模型。原Transformers4.46.3 RMSNorm类源码SHA冻结，实际worker复核。

逐步记录FP32主副本和实际forward FP16增益的变化数/L2，以及阶段首尾差。主副本变化可能被FP16舍入；不能据主副本的19validity声称19阶段均改变实际函数，实际增益变化也不等于整体输出/F1贡献。私有保存初始/最终两种增益完整张量及SHA，独立CPU复核最终差。TRAIN前后全量冻结参数digest核验不变，digest与保存开销计入总预算。

原19FIT图文准入另计38NLL+38最多4token短生成；之后原FIT训练，再唯一100DEV。独立CPU verifier核验全部ID/全参考原Answer F1/预测和checkpointSHA、19validity/87有限正LR、87原preclip梯度统计/19首步梯度、实际增益变化记录/原源与数据SHA。封存后才解释结果，不读partial。

## 判据与出口

固定两个已有FIT参照LoRA_qv宏39.870438970139254、OSFT_all宏40.78380056394307。primary为新宏严格高于两者较高者；五域与两个逐题配对均全报告。整已知组探索区间10000次/seed20261022；不重算旧对比，不称显著/等效/新留出/跨seed稳定，披露金融及任何域退步，不选路由。

未越过两个参照即结束这一个配方，不搜LR/层，不据阴性宣称必须新增参数。越过只保留可行性候选；不能宣布替代最强PASs。只有另行有意义的独立冻结版本、fresh FIT-only PASs参照完成后，才讨论与最强方法的功能替代或完整确认。正式PASs见过DEV，不能拿其DEV成绩筛选。本次不自动新增PASs、1313重训或21454评测，Ours/正式表不填。

## 投入与边界

只一组1213FIT/87更新+100DEV，STARTED至COMPLETE总2h含加载/原图文门槛/TRAIN/19次保存及SHA/参数digest/DEV；小HF准入与独立核验另计。逐步loss/LR/原梯度/参数变化/同PID资源/墙钟与逻辑保存字节完整记账，重叠计时不相加；原参照和此前探索成本保留。43008只是训练坐标数，不是推理压缩、有效维数或方法成功。

仅NEW物理GPU1单流、最多三卡，尊重其他用户，有效正式基线保全。五源/协议/原端口/基座/SPLIT/参照与预测SHA经独立review与CPU准入后READY冻结；STARTED后失败留证停止，修复独立版本，不热改/放宽/盲重试。端侧只研究背景，不迁移设备/导出部署或扩展工程交付。
