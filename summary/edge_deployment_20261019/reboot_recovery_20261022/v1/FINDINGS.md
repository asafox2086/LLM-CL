# 新机重启后的作业恢复，尚无新增质量结果

2026-10-22。新机SSH短暂不可达后恢复；`/proc/uptime`约180秒、boot ID `8052b133-a708-4278-97f0-c153682956d7`证明发生了重启。旧PID1154006/7965/181717/51266失效，不再作为存活证据。这是实际故障处理，不是科研性能进展。

## 已恢复和保全

- 原冻结四组调度由systemd自动接续为CPU2400/start10012；首组LoRA_qv的GPU worker3609/start11422已写STARTED、通过原19FIT图文门槛并实际训练。实查日志已到stage9，原LR、梯度和源不变；未读取DEV部分分数，不声称该组完成或有效。
- 新Qwen PASs-MoE5774/start52206、PaRSP5779/start52212已由原正式队列完成原preflight/reload并开始正式运行，分别用GPU2/3。处理前后身份相同，未中断这两条或GPU1小测；GPU0空闲不用研究，当前我方三卡。
- 原Qwen CL-MoE与SMoLoRA中断。旧run配置`checkpoint_all_stages=false`、checkpoints目录为空，且原冻结代码仅允许Seq阶段恢复；不能强加`--resume`或拼旧预测。独立只读评审确认这一点。旧run、日志、部分评测保留，旧源与当前源逐项一致；新鲜完整attempt按原条件重新排队，额外历史成本保留，不称断点续训。
- 只停止CPU正式调度器2521，核验其`KillMode=process`后事务更新上述两条状态，再启动同一原入口为6262/start58840。PASs/PaRSP在操作间身份未变，已被新CPU调度器接管；没有改训练源码、策略或评分。
- 原只读七方法发布器恢复为CPU6264/start58844；FOREVER Qwen原CPU入口恢复为6266/start58877，仍按原GPU2优先规则健康等待。FOREVER原state明确worker=null且从未启动训练，旧OWNER已失效；原state/OWNER逐字节归档到private后才启动原one-shot入口，没有删失败或重试模型。

## 阻止过期研究服务随开机恢复

重启还自动拉起了早已撤销的旧basis-budget/capacity/deterministic-native/knowledge-fit/research五服务。按现有人类方向与AGENTS“不恢复旧入口”，逐条核对ExecStart并停用开机启动。处理时basis CPU仍在，先按PID/startticks冻结再停；其他旧入口已自行退出。五服务最终均`inactive/disabled/MainPID=0`，原科学源/旧证据不改；不恢复旧容量全测。

最初恢复辅助脚本因旧preflight PID自然结束，在写审计和任何变更之前停在身份前置断言；重新读取原队列得到实际training5774/5779后才完成恢复事务。这不是模型失败、重试或门槛放宽。

证据见[RECOVERY.json](RECOVERY.json)、[OBSOLETE_FREEZE.json](OBSOLETE_FREEZE.json)、[OBSOLETE_DISABLE.json](OBSOLETE_DISABLE.json)。其中身份是带时间的恢复证据，不是永久live状态。原ACTIVATION留作历史，新boot身份以当前`/proc`、原队列和本恢复审计交叉确认。

正式完整结果仍8/16，最强OneVision点估计仍PASs46.406169；本轮没有新增F1成绩。最新人类“端侧只是科研背景”继续优先，不恢复电脑迁移或部署包工程。OSFT完整保护块的坐标重参数化仅保留为待原四组证据后评估的机制问题，未实现/准入/启动新研究实验。
