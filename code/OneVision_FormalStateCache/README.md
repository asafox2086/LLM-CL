# 公共V2协议的连续习得与功能缓存

最新用户要求先沿用研究设置、方法简化；此公共候选的完整GPU确认暂后移，保留代码、来源与所有条件。当前优先[初始化来源诊断](../OneVision_InitializationSource/README.md)，不是继续增加路由模块。

这不是旧原生研究成绩的改名。主轨迹精确复用正式SeqFT已核验的19阶段小适配状态；公共FP16输入、格式、目标截断和生成直接调用正式框架。原训练成本、额外构建与推理资源均计入。

- [冻结协议](../../summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1/PROTOCOL.md)
- [真实GPU驱动](../../analysis/onevision_formal_state_cache.py)
- [来源CPU提取](../../analysis/onevision_formal_state_export.py)
- [CPU合成集成检查](../../analysis/onevision_formal_cache_cpu_pipeline.py)
- [唯一队列扩展](../../analysis/onevision_formal_cache_queue.py)
- [公共Answer F1表](../../summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1/VERIFIED_RESULTS.md)

模块A保持连续习得并有界保留功能；模块B以输入相似度估计TRAIN回答质量。没有TEST答案路由或择优删除候选，CPU通过与阶段来源不是模型收益。FWT/BWT尚无该方法完整稀疏格时留空。
