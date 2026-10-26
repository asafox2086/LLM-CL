# 三臂固定预算适应：独立实质源码审阅

结论：接受当前两源，未发现必须修正的算法或producer接口阻断。仅允许据此推进既定实现；不是GPU执行、完整重载、质量或资源指标已通过。

实际完整阅读并AST检查的最终源码：

- `exp/conditional_shared_budget_20261026/v1/adaptation.py`：`251a2d490f9ce29069de141686a16da549c28de82f7c1ce58fab2df8da31252a`，159行。
- `exp/conditional_shared_budget_20261026/v1/worker.py`：`912d5e08edb2642268a483f98f01da7440f7fed730e7b2e91d32609127e190c8`，493行。

已对照本目录PROTOCOL／IMPLEMENTATION_MAP及原SeqLinear、plain-r8初始化／Batcher／readouts／完整导出接口、原run训练段和study.training_ast来源。

尺度与容量：shared保留原SeqLinear；uniform／soft只绑定forward，没有新Parameter或初始化。均保留48个q/v模块、96个r8 A/B对象和540672坐标，scale4、dropout0.1及原A／B0来源不变。一次dropout后只计算一次完整A响应；uniform对后两组r2固定各乘0.5；soft使用相同两组响应的FP32平方均值和温度1 softmax，梯度通过权重回传。共享前4维恒开，两私有组都计算，不是扩rank或硬稀疏加速。均匀与soft之间共享分支缩放合同一致。

训练与读数：原训练prefix的AST摘要被核对，GPU AdamW／逐stage原scheduler／clip／精度路径复用；before读数在prefix前、acquired在prefix后，无DEV backward。阶段读数保存恢复Python、NumPy、CPU Torch及当前工作GPU的RNG，恢复参数requires_grad和注册子模块mode；下一个原训练prefix的model.train()仍按原位置执行。单独readout Batcher不污染1213次训练输入账本。保留19份adapter、小阶段／gradient证据及原200条前后读数，不新增TEST或额外训练曝光。这里的1213FIT／100DEV／87更新是明确的探索口径，不冒充10%主实验。

最终图：shared／uniform以GPU FP32合并后转FP16，uniform正确包含私有0.5系数；soft保留完整显式r8。包保存完整core state、全部named buffers、base／generation配置和method mode，processor另存实际文件。eval入口在独立进程通过meta构造、模式安装、strict assign加载、buffer／别名／RoPE／tie恢复与完整receipt比较，不以外部基座权重补缺。soft没有被套进旧pure-FP16 receipt；FP32 A/B和FP32 inv_freq被分别保留。最终质量和资源都来自各自实际部署图。

GPU窗：eval先完整重载并核训练包／来源，再调用原五域hash-min固定32-token窗口，然后生成原100 DEV。窗口包含GPU视觉prepare和decoder，不先做DEV暖缓存。训练过程监控、独立推理窗口、参数／权重bytes／文件bytes分开，不能把设计计数当实测GPU优势。全源与输入绑定、frozen base／buffer及终态完整性检查均有输出，但其通过仍须实际运行证明。

范围：只读源码、协议及stdlib AST；没有导入Torch执行、运行模型、读取QA／新结果、评分、连接服务器或改变活动队列。没有新增controller、服务、审批层或实验配置。后续整体F1与真实成本决定是否保留条件机制，不以静态接受代替论文结果。
