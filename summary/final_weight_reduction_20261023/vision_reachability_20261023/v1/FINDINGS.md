# 回答路径可达性：否决视觉末层大幅删除，仅有微小公共冗余

## 问题与用途

在不猜测梯度/谱重要性的情况下，当前模型是否保存了根本不进入回答输出的视觉权重？若有大块不可达参数，可先形成真实可删依据；若没有，停止这条微小工程分支，不假删视觉塔或为了省几个KB启动大模型验证。

主agent和review_next_probe独立核对了当前配置、OneVision→SigLIP→所选特征→projector→语言生成的调用链。只读源码/config及94568字节safetensors头，头SHA与原声明874dbcdff7fd946c2f1d74b0d46059e598b03bdc2b93853f65603360a984c5a5一致。没有加载张量/模型/QA、调用GPU或改正式源码。

## 决定所依据的反证

| 最初候选 | 实际证据 | 决定 |
| --- | --- | --- |
| SigLIP末个encoder block可能在中间特征选择后不可达 | 实际vision_feature_layer=-1，26层encoder结束后追加最终hidden state；选的是末个block输出 | **不能删**。最后block实际15239504权重系数参与所选特征，不能照搬常见-2假设 |
| 池化head可能被保留但回答不用 | 实际vision_use_head=false，构造函数不创建head；权重头中0条相关权重 | 没有这份存量权重可删，不能把不存在的head算成节省 |
| post_layernorm可能不进入所选特征 | 该层另绑定last_hidden_state；hidden_states返回原encoder tuple；LayerNorm无原地写，get_image_features只取后者 | 当前eval回答路径支持其为**功能不可达候选**，范围仅限该调用链 |

PackedVision和Bridge.generate经core.get_image_features取同一字段；原生forward/generate与NativeGate也没有发现last_hidden_state或pooler_output旁路。缓存命中只复用所选特征，不改变必要性。此结论不覆盖整个SigLIP公开API，也不宣称TRAIN轨迹、RNG或真实浮点执行已实测完全相同。

## 参数账与行动

post_layernorm只有[1152]的weight和bias，共2304个F16系数/4608字节，约原893675552权重锚的0.000258%。这是选定三类候选中的一项真实公共冗余，不代表已经审计完全部模型，更不是显著的完整模型压缩。

**不为此启动模型删除、GPU门槛、checkpoint重载、导出、F1或延迟实验。** 没实际删除/运行，不能报最终模型已缩小或实际评分保持。只记工程附录/必要性账本的候选；若未来顺便简化，所有同骨干基线须公平应用，不能用这几个KB制造我们独占的轻量优势。原模型、权重、READY、正式证明和已发表参数量均不回写。

本分支关闭。它排除了“大段未使用视觉尾部可直接砍掉”的错误假设，但没有解决真正小模型的功能承接问题，不称创新算法或科研目标完成。

证据见EVIDENCE.json；独立RESULT_REVIEW SHA88c0cb30c0f0442af546988e8e4c2355ed19e33eb365670fb46a10998f6b87dd。源哈希是当前读取的快照，不能追认所有历史运行包已封链。没有新统计质量结果、旧F1/配对重算或旧worker恢复。
