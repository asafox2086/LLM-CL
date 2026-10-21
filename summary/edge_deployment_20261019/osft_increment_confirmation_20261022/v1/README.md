**2026-10-22训练阶段完成：** 同一GPU1 worker19535完成原1313/19/93，最终权重SHA416ed177803b9bf986dacd73a6d1bf18c6c0c59a1eb65f42785d81a628d31519、1941599549bytes；TRAIN_COMPLETE后已进入原最终21454一次生成。未有最终VERIFIED，不读partial分数，不是质量收益。下方启动观察均为历史；[阶段元数据](TRAIN_PHASE_OBSERVED.json)不代替完整最终核验。

**2026-10-22已启动：** READY a8c4b99aef2e7b5e803d027e45d31aebd5ad60df49fea7c2b459d719249da628；独立审阅accept，CPU19516/GPU1 19535。首次实查19 TRAIN图文门槛完成、stage1有3有限正LR更新，TRAIN未完成、TEST未开，不是质量收益。原正式基线和原screen终态保全。此为启动观察，未来以queue及完整VERIFIED为准；旧“审阅阶段”段落是历史。

# 唯一候选LoRA_all：完整1%最终质量确认

本独立版本承接[原四组完整结果](../../osft_increment_screen/v1/RESULTS_20261022.md)及[已接受的确认决定](../../osft_increment_screen/v1/FULL_CONFIRMATION_DECISION.md)。只LoRA_all新鲜训练原1313/19/93，再最终21454一次；小测另外三组保持结束，原screen入口不恢复。

主要量为五域QA域均值再等权宏均分，预定阈值46.619976395158304；AP19与每域对PASs/OSFT差同时报告。单seed不判稳定/显著，不以某域下降宣称五域全面改善。仅最终19格，FWT/BWT空，非原74格正式基线。普通LoRA不是新算法，训练参数少不是推理压缩，端侧仅科研背景。

当前是源/协议审阅阶段，尚无READY或启动，不是性能成果。权威后续为本目录READY、queue/ACTIVATION、STARTED、COMPLETE及独立VERIFIED；启动记录不当存活，partial分数不读。实际状态更新只覆盖本段，不改已冻结协议/科学源。

[协议、用途与成本出口](PROTOCOL.md)。真正失败留证停止，不盲重试；模型/QA/checkpoint私有，不入Git。研究只NEW GPU1，两机正式基线保全。
