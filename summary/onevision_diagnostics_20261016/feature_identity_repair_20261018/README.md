# 金融图像特征临时行缺少ID：故障、修复与接续

## 探究的问题

完整评测能否覆盖原生图像token需要修复的样本？旧质量硬调用完整确认在`FinMME:5082`停止：相似度特征提取创建的临时行只有question/images/_task，没有id；修复器成功计算2052→2106的token扩展后，写修复日志时读取`row['id']`抛出KeyError。原来的官方前向门禁使用有ID的原QA，因此没有覆盖这个辅助特征调用路径。

这是实现故障，不是数值不稳定、算法退化或新方法性能结论。原GPU358091与CPU389122均已终止；原日志、预测、异常文件和所有参数保留，不把缺失完整证明解释成成功。见[故障清单](FAILURE_EVIDENCE.json)和[原始堆栈](FAILURE_TRACEBACK.txt)。

## 做的实验与结果

1. 新源`analysis/onevision_feature_identity_repair.py`从固定原feature函数编译，仅给aux字典增加`id=row['id']`。AST剥离该项后与原函数完全一致；不改问题、图像、缓存键、向量空间、路由、A/B或优化器。
2. CPU控制复现原缺ID错误，修复后通过；兼容路径的实际CPU张量特征与缓存键完全一致，aux没有答案。见[CPU证据](CPU_CHECK.json)。这是实现检查，不是模型性能。
3. **实际GPU门禁通过**：原生6例官方/手工前向NLL误差均为0，包含FinMME:5082；辅助feature原缺ID错误可复现，修复后得到1792维有限特征；兼容样本特征/缓存键精确相同，全部参数版本和CPU/CUDA随机状态不变。额外4次feature调用计费，之后清空feature缓存。见[实际门禁](GPU_GATE.json)。

## 证明了什么

已经定位并修复一个元数据日志异常，证明兼容路径精确不变、原失败路径可执行。没有证明完整21454已经通过，也没有证明新的Answer F1提升。后续完整确认仍须原500 NLL/155预测精确重载和21454重新生成，不能拼接旧失败run的部分预测。

## 改进的方法与执行效果

这是实现修复，不作为方法第三模块。原模块A仍为完整旧A/B继承，B仍为TRAIN功能质量硬调用，冻结1313 TRAIN/19阶段/21454 TEST/seed42/Answer F1不变。

唯一研究队列现在为`queue_v2/queue_state.json`，冻结入口`analysis/onevision_identity_repair_queue_v2.py`，实际激活CPU405559；统一GPU入口`analysis/onevision_identity_repair_worker.py`。**原温启动19阶段实验已实际启动，GPU405584；不是仅完成CPU排队。** 完成证明尚待，不预填最终收益。

为避免再次用长完整评测阻塞首个初始化结果，接续顺序为：可用原依赖→原温启动19阶段→原温池四视图→全部18数据量对照→质量硬调用21454从头重测→全部原冷池/温池/组合/软调用完整确认与其余既定任务。全部旧条件保留，没有读取测试分数来排优先级。仅新GPU1；新GPU0保留，其他基线及其他用户进程不动。

修复队列v1在启动GPU之前被祖先命令精确检查拒绝，证据保留在[queue/needs_attention.json](queue/needs_attention.json)。v2保留原命令检查，再验证新入口逐项包装同一原命令，37策略＋66祖先策略＋36集成检查通过；没有删除原断言或放宽模型等价门槛。实际接续见[ACTIVATION](ACTIVATION.json)。

只读报告器现在为`llmcl-onevision-identity-repair-report.service`，激活CPU406495；旧389384计划退役。新报告器验证修复执行收据后维护原[四格结果](../warm_quality_20261018/FOUR_CELL_RESULTS.md)，不读取失败部分预测为结果。其他只读观察器保留。见[报告器交接](REPORTER_ACTIVATION.json)。

## 尚未完成与资源边界

完整四格仍待，不构成“已超过简单基线”。新增实现门禁的额外前向及预热历史记录在每个run，CPU/GPU前后/纯TRAIN峰值与整run峰值分开；不能据不同门禁历史的before差异宣称内存优势。修复后的冷/温完整确认都走同一新入口。发现另一个错误时保留证据、开新版本，不热改活动源或盲重试。

实际训练补证：第一阶段347 TRAIN/22次正LR更新后的全部A/B与原生顺序LoRA逐元素完全相同；第二阶段e1已完整继承e0的A/B，并完成11 TRAIN的一次正LR/有限梯度更新。说明模块A已真实激活、修复未改变共同首阶段，仍不等于完整性能提升。见[PAIRED_START_VERIFIED](PAIRED_START_VERIFIED.json)。
