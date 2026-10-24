# 四状态原图文配对支持诊断

## 要改变的研究判断

本实验检验：在原合法 FIT 轨迹中，OSFT 相对 q/v 的回答支持变化是否更依赖原图文配对关系。若原图支持改善、配对支持增加且相对 q/v 的差中差为正，可将该条件响应列为紧凑载体需要承接的功能；否则撤回这项具体依据。它不直接识别参数空间、知识类别、纯视觉语义或哪块权重可删，不选择压缩结构，也不是论文新框架。现有性能分数不能代替该输入干预。

这是冻结首阶段 PROTOCOL 的条件模型方案细化，不修改已完成配对。原 100 DEV 中固定 61 QA：自然 6、医学 13、遥感 20、金融 4、交通 18；全部配对、排除以原 PAIRING.json 为准，按其中 rows 的既定 ID 顺序执行纳入题。单次固定置换、几何条件子集、已探索 DEV、单 seed；不新建留出或 bootstrap，不称稳定、显著、等效或泛化保证。测量对象仅这两条 1213FIT 末态轨迹相对当前重建初态，不直接解释正式 1313/93 运行的刚习得优势，也不分离即时习得与后续遗忘。

## 四个独立状态与来源

按 qv_before、qv_after、osft_before、osft_after 串行，各独立进程，只保留一个模型。after 为原 osft_increment_screen 的合法 1213 FIT、19 阶段、87 更新的完整 stage_19 方法图，绑定 SCREEN_COMPLETE、对应 COMPLETE、checkpoint metadata 与载荷 SHA；不使用正式 1313 权重。before 从相同原基座以 seed42 和原 make_method 构造，不训练、不调用后续阶段再 SVD。两方法各有自身 before，不假设初始函数相同，当前重建态不是历史 S0 逐值复现。

四状态用同一当前 processor、原公共输入/视觉 bridge、方法实现、安装包文件与数值路径。旧 READY 缺预处理小文件历史 SHA，当前统一重建不能补称旧输入逐值链；同时记录本次包、配置与实际输入摘要。after 使用当前统一 processor，不复用 checkpoint 内可能不同的 processor。任何加载键、来源、stage、FIT/DEV 或必要类型不符即停止，不救跑。

## 干预与度量

每条保留自己的 question、context、task instruction 和原 `row.answer`；只用事前固定 donor 的 images 替换图片。不得用其它参考择优、人工改对象/标签或假设错图使 gold 为假。所有原 references 只做输入绑定，不参与挑选。

每个状态每条原图与错图各一次 teacher-forced NLL，61×2×4=488 次；0生成、训练、反向、优化、额外校准、原评分和新 F1。目标 tokenizer 不截断，附原 EOS；原 TRAIN helper 使用 max_length512 后附 EOS，本诊断与其目标上限不同，费用及 token 数公开。原 loss 算术和 shifted logits 对齐不改；只去掉目标截断。空目标、EOS 歧义、总输入超过 config.max_position_embeddings、非有限值均停止，不裁目标/输入。

原图和错图必须逐 tensor 验证 input_ids（包含视觉占位）、attention_mask、image_grid_thw、原始 position 及 packed 特征形状相同；只特征内容允许不同。每个状态记录两边非视觉输入、target、原问题/context/references 与图源 SHA，以及 packed 特征内容摘要。缓存键须来自实际供图路径/文件状态与布局，不以原 QA ID 命中原图；供图 SHA 和实际图源一一核实，记录两边视觉缓存计数/内容摘要，不能只核 token 形状。跨四状态实际输入和 target 摘要必须一致；视觉基座没有更新，所以同一 view 的 packed 特征摘要也必须一致，否则完整诊断不进入结论。实际形状不符时整固定诊断关闭，不跳题、换 donor、改插值或重新配对。

使用 eval、no_grad、原 FP16 基座/外层 autocast，保留原 q/v FP32 适配器与 OSFT FP32 因子/重构路径例外，不称适配器计算严格一致。不修改参数 dtype、路由、rank、LR 或已训练权重。清理旧视觉缓存，以当前图片/布局计算；128 MiB 特征缓存只作已存在的有界加速，全部费用/资源单列，不称整机内存上限。

每 QA 先对回答 token（含 EOS）求平均：C=NLL_wrong−NLL_original，DeltaC=C_after−C_before，D=DeltaC_OSFT−DeltaC_qv。并列 I_original=NLL_original_before−NLL_original_after 及错图变化、两方法 I_original 相对差。逐 QA、原任务、领域摘要全部保留；任务和领域内部各按纳入 QA 等权，总体摘要按五个领域等权，固定这两层权重，不看值后改权或删除自然/金融小子集。领域宏只在本子集内计算，不冒充原完整 100DEV 宏/AP19。候选依据要求 OSFT 的 I_original>0、DeltaC>0、D>0，并列报告相对 q/v 的原图改善；三个方向只检验本候选叙述，不设人为幅度门或显著门，不自动准入框架。D>0 若只来自错图恶化，不能称原图功能更好；q/v 的增加更大时不能据 OSFT 自身增加解释相对优势。不会以 NLL 直接认证生成 F1 或知识增加。

## 成本、完整性与停止

每状态从 STARTED 至 COMPLETE 最多 1800 秒，含加载、必要权重 SHA、构造/重载、122 NLL、参数及权重 buffer 前后摘要、全部观察和保存；四状态累计模型最多 7200 秒，等待资源不计。初始化 SVD 与所有视觉请求、token 数及运行历史计费，不称等训练计算或无额外曝光。此为资源投入上限，不是用户时限或 1% 定义。独立 controller 绑定自身 PID/startticks/UID/PGID 与单调钟，超时只终止自身隔离进程并回收；不操作他人/正式进程。

只 NEW GPU1 一个研究流，我们在 NEW 最多占三张、至少一张留给他人，尊重已有占用；不要求别人正在使用的卡也处于空闲。当前繁忙可先完成必要源码审阅，不能以空卡绕过准入。一次 metadata freeze 绑定源码、协议、配对、原 proof/源/小配置、实际解释器/包位置；不重新 tiny 训练或旧门。full-answer loss 的静态 AST 检查须确认相对原 helper 仅目标 tokenizer 截断参数及明确的输入/有限性守卫有变，原模型算术不变；模型本身的有限性和参数不变在实际推理中验证。

STARTED 前资源变化返回75可原条件等待，首次模型 STARTED 后任一状态失败、中断、非有限、布局不一致或超预算即关闭整个固定诊断；保留 partial 和费用，不补跑/拼接/续预测/换数值后端。全部四状态 COMPLETE 且退出后，冻结唯一 verifier 只做新标量/输入/参数/预算/文件完整性核验和上述差分摘要，不重算 NLL/F1/bootstrap。不得先读 partial NLL 挑方案。结果不自动触发新模型、压缩、训练或完整 21454TEST，下一设计仍须独立功能替代与成本依据。
