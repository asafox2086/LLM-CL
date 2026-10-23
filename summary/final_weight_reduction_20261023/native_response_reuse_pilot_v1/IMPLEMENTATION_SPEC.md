# 冻结前实施细化：不更改三个科学条件

原PROTOCOL及NECESSITY_REVIEW保留原字节。六个实现文件core/runtime/common/run/freeze/dispatch需独立源码审阅；当前没有SOURCE_REVIEW、资产下载、CPU门、READY或模型启动。旧WIP讨论不是本文件的准入证据。

实施分三道门，不把必要性接受当激活：

1. 完整源码审阅接受后，只允许固定revision官方资产获取及一次必要tiny CPU门。下载仅原始BF16 safetensors与明确列出的processor/tokenizer/config文件，不拉作者Python、ONNX、量化或其它模型。每文件核官方LFS SHA或Git blob ID，配置与既有元数据SHA逐个匹配；模型头计数不实例化真实权重。
2. 唯一CPU门在Torch导入前隐藏CUDA，使用微型Idefics3图，Q/R/E各2阶段4更新；E另有直接作者forward与learner回调的4步参考，检查loss/梯度/权重/RNG与历史精确。门只证明FP32小图、持久化与最终图重载；不能证明真实0.5B BF16/数据质量。解析实际原生模板/生成终止符只用固定合成文本，不按DEV调接口。门失败保留marker/日志，不自动重门。
3. READY绑定全部科学源、原TRAIN AST、manifest/SPLIT/作者及运行依赖、实际解释器/安装源、资产SHA和门。之后还须独立ADMISSION_REVIEW接受才可激活controller；此文件和SOURCE_REVIEW都不授权启动service。

当前阶段首FIT门只额外19次回答NLL与19次4-token短生成；主TRAIN1213次、最终DEV100次，两者分列。没有额外真实父模型前后NLL试验。编译前后差异由完整编译重载后的唯一DEV衡量，不默认为无损。CPU参考的额外优化不计成真实数据更新，但时间全部另计。

原TRAIN AST保持不变，通过命名空间接入原生input/loss/观察，实际每批QA IDs写入持久optimizer证据，输入日志区分TRAIN/NATIVE_GATE/DEV。验证精确1213FIT与100DEV的集合、阶段内固定顺序、每批覆盖、原图SHA；19首步原梯度、87标量与87实际optimizer witness逐文件必需。训练CP保存全state_dict/config/graph spec及learner状态，方法不是可pickle的局部类；没有恢复入口，不能称部署或续训证明。

最终编译后清除训练模型引用，再构建原生/专家图strict加载并核state digest；公共q/v公平fold，R恢复BF16 Down，E保留实际推理专家和必要buffers。unknown buffers保守计入权重；只将明确的inv_freq/position_ids/expert_mask派生或索引buffers另外报告。所有shared storage按device与实际区间去重，非连续view拒绝而不猜共享。

内部alarm与独立controller均从STARTED单调钟起算。controller只核UID/startticks/PGID后终止自己的隔离worker组并wait，不触碰其它GPU。controller记录退出观察时的保守elapsed bound（含最多一次轮询滞后），同时保留worker RUN，不假称该上界为精确活跃计算时间。若上界不能证明≤原6h/2h预算则关闭，不给予补算时间；独立核验另计。controller不自动复活，STARTED失败/中断/超时不重做；75只在无STARTED/private目录时健康等候。

每完整case只有一次原score AST计算，验证marker防重入；E/Q门只消费CASE_VERIFIED并绑定READY/COMPLETE/退出证据的SHA。汇总不重新打开TRAIN QA、不重算E/Q分，只在确有R时读封SHA的SPLIT组元数据并新增两对固定seed配对。E≤Q则R未测并关闭本固定点。与新颖性、稳定性、非劣及完整确认的边界均遵循原协议。
