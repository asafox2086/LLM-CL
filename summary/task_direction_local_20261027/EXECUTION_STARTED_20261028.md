# 同起点方向对照：真实启动与研究优先让卡

2026-10-28，四阶段实验已在OLD GPU1实际启动；首次外层STARTED为unix1793116903.9510057，aligned/train worker为2094139/start781532432/UID1084，controller2094023/start781531848。原5400秒总预算从实际启动计，不重置或扩展。此记录没有方法F1结果；完整四阶段及唯一80评分后才判断方向效用。

此前Qwen PaRSP卡在初始完整评估，尚无训练更新或完整checkpoint。按人类已明确的“停运行时间最短基线、研究优先”授权，独立审阅后停止其精确匹配的worker/wrapper，原RUN全部文件保留，并以逐文件SHA/大小核验完整tar归档。它重新进入原队列的ready_training；后续按原数据、种子和配置建立新RUN、从S0重新运行，不称checkpoint续训。其他三组OLD基线未被发信号，LLaVA PaRSP的S3恢复许可保持。旧主表相关记录不删除、不覆盖。

原分配器和idle优先链保持，新CPU2093656/start781530315/UID1084，unit `llmcl-pct10-direction-local-priority-old-20261027.service`。完整停止/归档证据在OLD耐久 `.deployment/direction_local_zero_update_preempt_20261027/`；tar SHA c842157eb0895ec79202e15bbecdd31f5f0d51daa07461eb1d218b1f662b4dd5。仅外层协调交接，不修改冻结的科学源、binding或协议。

研究问题仍是：同一个shared-r8 S15完整模型、同一T16的69 FIT/5更新，当前任务梯度给出的初始方向是否比等范数随机方向带来更好的20题金融DEV回答？540672只指主优化器坐标，校准临时dense坐标和SVD峰另账。最终两臂严格重载后的完整模型与五资源窗均取本次证据，不借用旧小图显存；局部结果不填正式五域主表。

[原准备与历史等待](EXECUTION_PREPARED.md) · [固定协议](PROTOCOL.md) · [实际启动/停止来源](STARTED_20261028.json)
