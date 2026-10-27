# Type auxiliary 实现只读复核

结论：当前六源未见阻断；可作为后续接入评判的源码依据，不代表GPU准入、实际完整重载通过或有效性。

- S15使用原`checkpoints.save/load`完整保存未fold图、A/B、processor及Python/NumPy/CPU/全部CUDA RNG；保存恢复随机流，分支进入训练前再次恢复。阶段边界按原合同重建optimizer/scheduler/scaler。旧adapter快照没有被冒充完整状态。完整cloudpickle GPU加载仍未执行验证。
- 原答案前向只扩展返回实际packed prompt最后位置的最终hidden；没有全层hidden或额外decoder。辅助头896→3、2691坐标，在T16取训练参数前注册，实际进入optimizer和共同clip；T16后移除head及loss闭包。plain不启用辅助梯度。最终完整core部署/重载不含head，单独保存的training_head仅诊断用途。
- 新verifier的`SUPERVISOR.phases`逐项覆盖实际EXIT字段，再核worker STARTED与COMPLETE摘要；不把控制器事后追加的三个proof字段要求为原EXIT已有字段。控制器及七worker都检查原身份已消失。
- 三个`HEAD_INITIAL`核相同digest、2691参数和RNG恢复；69个真实/打乱标签从原冻结FIT ID顺序及metadata重建。Random(42)同原helper、40/69改变；监督条目完整映射且plain为空，没有读取DEV/TEST选择标签。
- 660原评分对象与675生成（含15资源窗）口径正确。题型F1与严格原始字母标记正确；最终五域各20条，macro按五域均值等权，未把prefix/suffix局部域均值当全域宏。
- 新TypeAuxSlot在原PairedSlot.wrap之后包同一pipeline.idle；原resource/paired/position链及锁继续使用，先确认原paired released。slot独立，claim一次、不重试；在controller、全部phase及failure最后尝试worker原身份消失并经NVML确认后释放。其sources为queue/controller/worker/helpers/training五项，删除queue后恰等controller的四源，终态集合匹配。phase子进程仍沿原entry，不另起分配器；drain只在原main正常返回后复用原lock/idle。

范围：本轮只读六源及实际继承producer/合同，另做AST语法检查；无服务器、Torch导入、模型、QA读取、生成、评分、队列接管或GPU验证。未改六源。审阅者曾编写被复用的旧paired/conditional部分源码，本轮六个type辅助文件均非审阅者编写；因此属于跨接口复核，不能声称所有依赖均由全新独立作者审查。

源码SHA：

- `controller.py`: `f7b7c20a8e8cd8a154fbb6a4c799fc915c23214aefce4122d6bf99b542375622`
- `helpers.py`: `b141ef72c7adf9e1c5adc93eab174fe4577d82c8bca2a1759308533f2c7e97f5`
- `queue_adapter.py`: `9dc9e3baef0eee4d81f9a7bdc6a4a2618cf8326c2bfc6a949619f1620b81dae7`
- `training.py`: `e9f1ab7fb4df06c17aa0ba1a2a3db7ba211d200dbe6490c18d01761a22f6e297`
- `verify.py`: `1caa7e611ac1b5e970dafebd622cf13046501830ce10d3a9662d7f7b9b57c750`
- `worker.py`: `82fc897d957027471ef303556957826afb9e5ac2f6a16119d8b67ade7c8d6b57`
