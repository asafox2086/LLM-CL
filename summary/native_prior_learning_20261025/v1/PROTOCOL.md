# 完整原生紧凑先验在1%连续学习中的资格

本项只回答：保留原生小模型的视觉、连接器、语言整体，以普通有限预算 q/v 学习，能否得到达到合法父参照各域水平、同时完整权重与GPU内存更低的终态。它是载体／学习基线资格，不是新算法或原生联合预训练的独立因果证明。

## 研究理由与边界修订

已完成的581M拼接图只取Smol视觉塔、换仿射接入OneVision语言；即使起点改为先小后学、等4399104坐标的七位置更新，末态39.262042仍低于完整wide父43.791987，且四域末读数未支持将全部差距归为后续遗忘。该结果没有测原生视觉／连接器／语言耦合保留的完整图。

旧原生Smol初态检查在首题processor.resize的2048配置与实现默认1820上限冲突时停止，尚未generate，0回答／评分；实际加载和15.292007秒失败区间保留。旧E/Q/R更早在CPU小图E梯度一致性失败，Q真实载体训练未发生。不能把这些技术缺测当原生图质量阴性，也不能要求先有F1才能准许测F1。

**透明窄修订：** 当前人类要求继续有价值的端侧研究并用子agent评判；两位评判认可重开旧未执行Q的控制角色。旧BF16/HostAdamW-Q及所有旧源／失败仍保留，不复活E/R，不声称这是全新假说或只改processor的逐字节复现。当前固定FP16／GPUAdamW条件与父相同存储位宽；这是结构资格的单一数值条件，无BF16对比、精度／后端搜索或精度收益贡献。规则029保持。

作者[SmolVLM论文](https://arxiv.org/abs/2504.05299)已研究紧凑视觉与语言容量配置、原生视觉token压缩等。借用该现成模型可获得相关资源空间；其设计与已有预训练能力归原作者，不能记成我们创新。本项正结果只提供可用的小图地基，再以同载体基线评判后续机制是否必要。

## 固定条件与原训练预算

- 唯一原生资产 HuggingFaceTB/SmolVLM-500M-Instruct，revision a7da5b986cb59b408707209984f360a5f4ad7e47；复用已封12项资产，无型号搜索。声明507482304个完整系数／1014964608个16位权重bytes；相对893675552父预计少43.214%，并非已测实际部署包或GPU峰。
- 固定FP16基座／外层CUDA autocast、FP32 SeqLinear q/v-r8 alpha32 dropout0.1，32层64模块／128 A/B／819200坐标；原native命名seed初始化，B为零。普通视觉、连接器、词表与语言基座冻结，无新专家、路由或外部教师。FP32参数存储不称全FP32前向。
- 固定native vision=eager、text=SDPA；安装4.46.3的Idefics3VisionTransformer不支持SDPA，不改库造算子。父vision/text=SDPA。跨骨干实现例外公开，资源是完整实际实现的比较，不归为纯架构单因果。
- 同原1213 FIT、19阶段、seed42、原确定顺序与原训练AST，micro1／batch16累计、87次更新、lr1e-4、每阶段新optimizer与原cosine／warmup／clip1。仅将HostAdamW换GPU torch AdamW（betas.9/.999、eps1e-8、weight_decay0、capturable=True）；无CPU激活／优化状态托管，不称与旧Host算术逐值一致。
- 原100 DEV仅初态／末态生成；不作训练、早停或phase选择。不读取TEST、不额外诊断NLL／逐阶段DEV／真实QA门。每次FIT QA恰一次forward/backward，所有1213与87计费；不同tokenizer、层数和视觉序列意味着计算量不相等。
- 原问题／原图SHA／instruction／上下文BM25策略／图像面积401408相同，各自native模板和processor。native官方2048最长边、512 tile、scale4和64图像token／tile均固定。只对该processor实例恢复2048上限调用与image_seq_len64的资产语义，不修改安装库、其它作业或分辨率。目标原生答案body前512 token后加唯一assistant终止，公开相对旧未执行Q的严格拒长策略变化，不删除长QA。上下文8192，原greedy生成cap不缩短。

## GPU执行、完整图与评价

只OLD真正空GPU单流串行；NEW四卡当前有他人／正式工作，不抢占、不发信号。模型前向／反向、GPU AdamW、合并和必要检查均在GPU；CPU为常规数据预处理、文件／张量I/O、哈希与评分。构图使用标准GPU from_pretrained完整权重I/O，禁止初始化随后会丢弃的完整随机CUDA模型；相同loader来源的所有持久／非持久buffers及FP32 RoPE实际核验。

顺序固定initial→train→native_eval→parent_resource，初分不作为训练准入门；只有全部退出后唯一评分200条。initial100只描述作者先验，train0DEV，native_eval独立fresh完整重载先5个固定32-token窗再100原DEV，parent_resource同卡fresh父图仅同5个原ID／公共raw policy／32-token窗，0新父F1。合计210新生成、200新F1；旧父F1只由原完整VERIFIED按ID／原始数据绑定复用。跨骨干prepared tensor、视觉token及prompt长度不要求相同，全部实际记录；相同raw输入预算不称同计算量。

训练后GPU FP32合并64个适配器，禁autocast／TF32后转FP16，物理移除A/B，完整包不含优化器、教师、cache或额外完整基座。精确计权重／普通buffers／实际文件与别名，严格独立重载后评价。推理窗包含准备输入和native视觉前向，报告allocated/reserved峰与输入形状；训练峰／全流程RSS另列，不称CPU优势或推理速度。

各阶段在同一进程CUDA上下文中使用fresh重载模型；弱引用证明旧模型、参数、buffers与优化器载荷已释放，记录剩余allocated，不以任意空载显存门槛判故障。库工作空间可能常驻，因此不称独立冷进程测量。本次父图资源参照仅核验完整持久权重及同加载器buffers一致；43.79为历史封存质量目标，未通过本次新父生成重新证明。

固定STARTED到COMPLETE预算：initial600秒、train3600秒、native_eval600秒、parent_resource240秒，总5040模型秒，包含加载、数据输入、所有模型计算、统计、保存／重载；首次STARTED后不重置。一次必要GPU接口检查若采用另列0真实QA成本，不重复旧CPU门或把检查当科研结果。真实故障／中断／超时停止本固定点；无rank、LR、专家、memory、额外步数、型号、dtype或backend救跑。若纯实施修复确有必要，必须先透明评审、保留失败／全部成本、不延原截止，不能把自设条款当永久科学阴性，也不能滚动修补填时间。

## 结果如何改变后续行动

只有末态五域点F1均不低合法完整wide父、完整权重／实际包更小且同卡5个GPU峰均更低，才保留为同载体方法设计的紧凑地基；这仍不是新框架／正式最强基线胜利／显著或非劣证明。宏均值过线但某域缺口不算通过。

未通过则关闭这一个固定native-Q实现，不自动加专家／rank、换型号或改数值重跑。若初态好而末态下降，只能说明该适应配方未保能力；若初态已足够且学习没有增量，已有能力归作者先验，不把全部终态能力归87步。若实施／数值失败则质量缺测，不判原生模型能力或纯容量不足。原DEV已探索、单seed每域20，只点比较；跨骨干预训练、tokenizer、视觉接口和实际实现差异未分离。
