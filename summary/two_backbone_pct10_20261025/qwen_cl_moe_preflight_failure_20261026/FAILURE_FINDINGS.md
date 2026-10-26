# Qwen 10% CL-MoE：首次预检联合有限性检查失败

真实队列状态为`failed_preflight`，不应继续显示排队或正式训练。worker512937／start770646734及wrapper512924／start770646715均已退出；原phase EXIT在unix1793008105.158171记returncode=1。本次只取回失败小证据，未改远端、设置或模型，没有重跑/救跑或评分。

冻结预检`run.py:198`触发`assert torch.isfinite(loss) and torch.isfinite(grad)`。日志没有记录本次loss或grad的数值；**只能确定两者的联合有限性条件失败，不能判定是哪一项或两项，也不能直接归因数据、算法、dtype或硬件。** 当前trace是AssertionError，没有独立根因证据。

源码顺序与原日志共同支持：这是第一个T01预检QA，真实answer-loss前向、backward、unscale及梯度裁剪已经到达；断言在`scaler.step(optimizer)`之前。日志已完成preflight case数为0，因此本预检0个成功optimizer step。不能写成0模型/0反向，也不能把它当完整训练的质量阴性。

`run_status.json`的`initial_evaluation/S0`在源码中位于预检分支之前统一写入，**不证明已做正式初始TEST评价**。本次`gpu_preflight.json`、reload proof、latest checkpoint及metrics均不存在；没有通过预检、重载或完整19阶段/TEST结果，最终参数与正式F1仍缺测。源/config明确为pct10、10% TRAIN/完整TEST配置、FP16基座/外层autocast、RTX2080Ti、Torch2.6.0+cu118；这只是实际条件，不是非有限的原因诊断。

phase status记录到exit约58.185866秒，只是该监督记录区间，不冒充完整RUN或纯GPU时间。原config、supervisor.log、phase status/exit、run_status均按原字节保留；冻结run片段与config源SHA闭合，见[FAILURE_EVIDENCE.json](FAILURE_EVIDENCE.json)。0新QA/CP载荷读取、0模型、0新评分。

## 正确协调器交接入口

已从OLD取回实际`CONTROLLER_HANDOFF.json`原字节并更新本地同名文件；旧本地版本另以`CONTROLLER_HANDOFF_LOCAL_STALE_<SHA前缀>.json`保留，没有修改远端。原observer据旧交接记录查询已停止unit，不代表当前协调器停止。

正确user-manager一次补查unix1793013692.0331175：`llmcl-pct10-baselines-conditional-old-20261026.service`为active/running，MainPID609661／start771089493／UID1084／PGID609661真实匹配，Restart=no、KillMode=process；conditional slot尚不存在，未读partial质量。见[COORDINATOR_METADATA.json](COORDINATOR_METADATA.json)。我的首次补查漏`--user`命中system scope的inactive记录，已保留在私有原fetch bundle并明确纠正，不作为user unit状态。

这里仅保存当前失败和正确交接快照；不恢复基线、不修改数据/精度/预算或旧科学源。OLD relay已释放，不继续轮询。
