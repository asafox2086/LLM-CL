**2026-10-22已complete_verified：** 宽q/v宏43.791987409，对all+1.839438pp、区间[−3.134644,6.855608]跨0，15题高/15低/70平；金融3题贡献宏+3pp，其他四域抵消−1.160562pp。仅固定DEV点估计，不是全面/稳定/等效或新方法成功。[完整结果与结论](FINDINGS.md) · [封存核验](VERIFIED.json) · [终态观察](TERMINAL_OBSERVATION.json)。按原机制协议不列完整候选、不自动1313/21454/第二scale；44689/44708正常结束、service inactive/exit0，不恢复。下方启动记录是历史，原源/READY不改。
+
+# 一个同坐标预算的分配机制对照

唯一新增BudgetMatchedQV：4399104训练坐标集中于48个q/v，与原七类r8的168模块/4399104比较；原qv/all的FIT/DEV结果复用，不重训。rank65及6个按模块名哈希固定的v模块rank66由预算算术得到，scale4/LR1e-4固定，不是rank搜索。只匹配坐标与乘数，不匹配有效自由度/Jacobian/权重步长/计算量。

本问题在候选TEST前已有记录；[来源](QUESTION_PROVENANCE.json)绑定此前Git blob。本独立协议确实在完整确认后形成，不能称预注册；不解释或挽救已停止的完整LoRA_all，不据TEST选rank/位置。原同一1213 FIT/100已探索DEV，19阶段/87更新。唯一新增配置只作机制对照，不自动完整1313/21454或列候选。

源/协议经独立审阅，新增小HF准入5.112秒通过，READY61c2605f4c6726ee1dc43e022289b293a0f92af90c1ef0d29f68c9962bf6d137已冻结。独立CPU44689/start2220228、实际GPU1 worker44708/start2220751已STARTED，实际配置48模块/4399104/scale4核实；以[启动观察](LAUNCH_OBSERVATION.json)、实际PID与后续VERIFIED为准，不把启动记录当持续存活或F1。基线保护身份未变，原完成作业不恢复。

[协议及正负出口](PROTOCOL.md) · [源审阅](REVIEW.json) · [必要CPU准入](CPU_GATE.json) · [固定rank与条件维数边界](RANK_PLAN.json)。当前尚无完整新DEV成绩，不读partial；普通LoRA加宽不是新算法/推理压缩，端侧只是科研背景。
