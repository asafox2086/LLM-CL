# 独立 FOREVER 多模态接入

从 V2 复制共同实现，原目录不改。只准入非正式 GPU 接入门槛，不能直接当作正式可发布基线。协议与来源锁见 [接入记录](../../summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v1/README.md)。运行目录、原始梯度和模型checkpoint禁止入Git。

主训练沿用共同一轮1313/19/93预算，阶段末额外回放使用作者两轮请求及warmup5/linear；实际曝光、正LR/零warmup步、正则和CPU/GPU成本单列。两骨干都须真实前向/反向/回放/重载通过后另做正式发布准入，当前不填FOREVER分数。
