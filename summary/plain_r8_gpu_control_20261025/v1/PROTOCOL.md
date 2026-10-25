# 朴素 q/v-r8 的 GPU 匹配控制

本点只建立后续同秩持续学习干预需要的朴素锚，不是新方法，不复活已关闭的宽 QV 正式确认。用户要求方法简单、有适配创新，不能依靠增加容量。已完成的宽 QV GPU 控制（100 个已探索 DEV 宏 F1 45.085962854）仅作为容量背景；原历史 r8 使用 HostAdam/CPU fold，与后续 GPU 学习不匹配，无法把差异归给持续学习机制。因此现在补齐同原 r8 容量、同 GPU 数学的必要基线，之后可直接复用，不重复该基线。

## 固定学习内容

Backbone 为完整 LLaVA-OneVision-Qwen2-0.5B-OV；OneVision 是 LLaVA 系列的该具体变体，不能把语言模型标称 0.5B 当作完整视觉语言权重数。仅 24 层 q_proj/v_proj 共 48 个位置，各 r8、alpha32（scale4）、dropout0.1，A/B 均学习。共 96 个 FP32 参数张量、540672 个训练坐标，实际梯度 2162688 字节；每阶段新建 GPU AdamW，两个 moments 与 96 个 FP32 step 合计 4325760 GPU 字节。没有新增专家、rank、selector、响应标量、teacher、回放或额外训练数据。

采用已运行的 BudgetQV 构造器，将其固定 rank_plan 的每个值设为 8。该构造器先调用原 LoRA_qv r8 初始化，然后在 fork_rng 内将原 A/B 的完整前 8 维复制到替换层；rank8 时即完整 r8。保留构造器原初始化与 RNG 行为，reset42 后实例化，再 reset42 开始训练。运行中记录原 r8 全段复制断言和 48 组 A/B 摘要。不能与宽 QV 的全 A/B 摘要作相等断言，因为形状和 rank 不同；也不把零 B 的数学关系称为本次新生成的初态回答等价证据。

固定原 1213 FIT、19 阶段、87 个正 LR 更新，micro1/累积16；原训练 AST、样本顺序、loss、answer512+EOS、GradScaler、clip1、stage reset schedule/warmup、lr1e-4、betas(.9,.999)、eps1e-8、weight_decay0 全不变。GPU AdamW foreach/fused false、capturable true；原 FP16 基座和 autocast 下 SeqLinear 参数存储 FP32，不能误称所有适配器计算均 FP32。GPU 禁 TF32；合并在 GPU FP32 且禁 autocast，再转原 FP16。模型前向、反向、optimizer 和合并全在 GPU；CPU 仅继承的初始化、数据/摘要和 checkpoint I/O。没有 CPU 神经网络门或 CPU 算子探究。

逐 QA 的 1213 个真实 prepared/transport/raw/image/context 记录、阶段样本顺序及 AST 与已核宽 QV 所继承的原 dynamic FIT 记录逐值相同。87 个梯度 receipt、19 个 adapter 和阶段正 LR/参数实际改变证明保留。冻结原权重与全部普通 buffers 在训练前后不变，A/B 必须实际更新。无 DEV 训练、未来阶段 FIT 回放或额外恢复步。

## 原态、完整部署和参数量

先从同一原权重加载，完整 digest/buffer/别名/config/生成配置绑定原 calibration_COMPLETE、pair 已核 original_base 及已封 QV 来源链；导出完整初始包后，通过自包含 meta 图 strict reload 再训练。训练末将 48 个 q/v 合并回原 FP16 Linear，完整末态必须实测 893675552 个权重系数、1787351104 权重字节、9032 普通 buffer 字节，最终新增部署权重为 0。保存实际完整 checkpoint 文件大小、SHA、别名、所有普通 buffers 和严格重载证明；不能把训练的 540672 个坐标称为最终总参数。

完整包不暗含 adapter、optimizer、外部教师或外部 base 依赖。保留 19 份 adapter/87 梯度历史只供审计与后续同秩控制的起点核验，不作为部署图。模型占用、权重字节、文件大小、训练参数量分别报告；不把采样训练内存称为受控推理峰。

## 唯一末态评价

固定 OLD 物理 GPU0 训练，随后独立进程在 OLD 物理 GPU1 strict reload 评价，两个 phase 串行。与旧 pair、宽 QV 和强 PAS 资源窗同一物理评价卡/相同输入约束。无初态/阶段中 DEV、NLL、TEST 或新刚习得评价。

先按既定每域 hash 最小 ID 做五个固定资源窗，各强制 32 新 token，保留 baseline/peak allocated/reserved 和实际输入 digest；再生成原 100 DEV 的最终完整回答。总 105 次新生成，其中仅 100 个新回答按原 token Answer F1 评分。五域各 20 QA，主量是五域等权宏 F1；同时完整报告五域及逐 ID，不要求每域不降低，不根据结果改汇总方式。

唯一 verifier 只能在所有 phase COMPLETE、EXIT0、ALL_EXIT 且原身份退出后运行一次，先核完源/资产/输入/训练/完整权重/重载/状态不变/资源窗口链，再写 VERIFY_STARTED 并评分 100 个新回答。旧宽 QV、dynamic、static 的质量仅复制其 VERIFIED 标量，不重评分、重生成、bootstrap；旧 prediction 只用于输入证明与 ID 对齐。宽 QV 与本 r8 的差是容量背景，绝不能写成新方法收益；本次不证明创新或胜最强基线。

后续 I-LoRA 适配或其他同秩机制须另有问题、协议和独立审阅，不能自动启动，也不能按本次 DEV 结果搜 rank/LR/QA/插值参数。本基线用于固定同秩机制的整体净作用比较；即使后续获益，也须区分适配工具与可支持的论文创新，并在正式协议验证总体 F1 与部署成本。

## 来源、预算与故障

继承已封 QV RUN `/tmp/llmcl-response-qv-control-1084-nkmx9g5k` 的运行环境及 ASSET `/tmp/llmcl-response-pair-1084-geic2vef/asset_view`；READY 逐字节绑定本七源、协议、原 runtime/environment、manifest/split、calibration、pair 及 sealed QV 证明。新增字段 prior_qv_run 指向该 QV RUN；需要的复制名称见 AUTHOR_SOURCE_REVIEW.json。源码和协议完成不等于 READY、启动或效果。

首次 train_qv STARTED 起 3600 秒，随后 eval_qv STARTED 起 600 秒，总 4200 模型秒，包含加载、摘要、初末保存/重载、1213 反向/87 更新与全部 105 生成。所有预算不因观察、换会话或文档重置。只在实际空卡启动，隔离 PGID，保全他人/基线工作；Restart=no，不 enable。本代理只交付源码/协议，未运行模型、CPU 模型门、服务器命令或部署。

首次模型故障/中断/超时保留证据与已发生费用并关闭固定点，不续 CP、加时、自动重试或搜索配置。无质量则记缺测，不当作机制阴性。历史 pair/宽 QV/校准的训练与评价成本分列，不能抹去，也不能冒充本次新计算。
