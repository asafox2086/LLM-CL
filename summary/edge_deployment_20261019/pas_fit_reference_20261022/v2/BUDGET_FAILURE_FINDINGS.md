# Fresh PAS参照在冻结预算内未完成：质量缺测，按协议关闭

问题：原ShortFixed先导越过两个锚的固定点门槛后，fresh FIT-only PAS参照用于决定是否值得完整确认。正式1313 PAS见过DEV不能替代。当前参照v2的4h STARTED→COMPLETE预算在模型启动前冻结，失败即停止；不据已见TEST改变配方。

结果：同一boot、CPU10505/worker10568在冻结alarm触发后自然退出，unit failed/MainPID0/exit1。记录77/87更新，最后S17 step3/46QA；17完整checkpoint与17阶段validity文件，未完成TRAIN/19阶段/100DEV，无COMPLETE/VERIFIED，质量没有测得。已记录FIT曝光1074，最后记录后的inflight前后向未知，不能按零计。原19FIT门槛38NLL+38short已发生，原v1历史费用另计。77个即时持久optimizer witness存在，但不是87步完整核验；不调用原完整verifier/配对，不读partial DEV。

最后完整CP3051081597bytes，实际一次流式SHA45caf48ac72dae644cd52d7eba24c70aa43b5baad5cde9d497b6fd88b28f3d33与原metadata相等；仅作为未完成轨迹封存，不加载或续训，不称最终模型。135个JSON/log当前原字节snapshot、2677个有效资源行（无坏行）已封SHA；另34个PT原位保留，仅列尺寸，没有补称内容全部已核验。源/READY0f0faa9d...不改。

费用：STARTED文件mtime08:42:10.848860UTC至FAILURE12:42:15.516663UTC为14404.667804秒，非完整RUN；FAILURE栈位于backward；该文件mtime间隔比名义4小时多约4.668秒，不能定位为C++或信号处理延迟。冻结alarm未修改，未追加训练预算。77条更新主体累计12754.420571秒，17次保存元数据累计1075.757412秒，逻辑保存51925459759bytes非物理IO，各区间与墙钟重叠不相加。故障封存采用seal中sealing_seconds_before_write=9.153464秒口径（只到写seal之前，不含后续写出及queue闭合），单列，CPU准入8.026794秒和v1既有成本另列。原seal错误使用不存在的save_seconds键将保存统计记成0，原错误seal保留，FAILURE_COST_CORRECTION以17个原metadata的seconds更正；没有重跑模型/大CP hash/评分。

结论：这只是该原配方在本设备、实现与预算下未完成的成本事实，不是PAS能力/质量阴性，也不证明ShortFixed胜出或失败。ShortFixed对原OSFT/qv的已有探索结果仍保留，但fresh PAS比较条件缺测，当前不能放行完整确认。

行动与效果：关闭本参照及当前完整确认分支，不续训/延时/开v3/自动新实验，不拼旧权重/预测、不放宽87步门槛。保留最后CP、原源/READY/FAILURE/needs_attention/资源与费用；canonical queue closed_budget_timeout。原两机正式基线继续，实际2497/8260/8265/10495/10498在终态观察时匹配，没有GPU信号/科学源改动。Ours/正式表不填，无新F1/方法收益，科研目标尚未实现。
