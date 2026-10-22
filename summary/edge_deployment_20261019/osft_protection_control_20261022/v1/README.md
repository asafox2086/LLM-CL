# OSFT逐步保护包的净效果

端侧只作科研背景。固定原OSFT七类位置、25%尾部USV初始化、113766144训练坐标、LR1e-5与HostAdamW/offload/逐阶段重SVD，仅同时关闭每微样本梯度投影与每更新参数投影；复用原OSFT_all参照，只新增1213FIT/19阶段87更新→100DEV一组。

关闭后允许集合扩大，后续重SVD高基也可分化；它检验保护包整体，不能分开归因于两投影/Adam几何，不能仅凭末态DEV解释习得与遗忘。无论正负，本机制控制完成后正常退出，不列完整候选、不自动1313/21454、不追加投影组合或LR搜索。DEV已探索，不称新未见确认。

五源/协议独立review接受，小HF必要准入2.517秒通过，READY6caf387f4f62c534b50928731911d8d8e7ef53de7c531fc4fe8f451baee9f885已冻结。真实CPU48920/start2451943、GPU1 worker48939/start2452452已STARTED；启动观察时仍加载/SVD、尚无配置/有效更新，不能把此记录当持续live或质量成果。后续以实际身份与VERIFIED为准，已完工作不恢复，正式基线保全。

[问题、正负用途与成本边界](PROTOCOL.md) · [源码审阅](REVIEW.json) · [必要准入](CPU_GATE.json) · [唯一干预](INTERVENTION.json) · [问题来源](QUESTION_PROVENANCE.json) · [冻结](READY.json) · [启动观察](LAUNCH_OBSERVATION.json)。无部署工程或新方法成功结论，Ours/未测部署空白。
