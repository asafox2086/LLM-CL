# 新会话交接

## 状态

人类已要求本聊天停止继续探究；自动跟进已暂停，科研目标未实现。现有实验、数据和代码保留。**没有做出保质减重的新框架，也没有解释完OSFT优势。**

- [全部成果与负结果](README.md)
- [补充说明、原证据和运行交接](SUPPLEMENTARY_NOTES.md)
- [最新停止／交接记录](summary/research_handoff_20261024/STOP_HANDOFF.json)

## 当前已有实验

OSFT阶段刷新对照：从同一个已学S1起点分叉，低因子均继续训练，仅后续是否重分解不同；比较刚习得与末态，不是新压缩方法。

2026-10-24T15:50:24.235649+00:00仅交接实查：共同前缀完成，Fixed正在运行，Refresh未启动，尚无完整VERIFIED。**原训练服务没有被终止**，保留给接手者；本聊天不再自动分析或新增实验。实际存活状态以后重新核查，不照抄旧PID。

- OLD目录：`/data2/liyapeng_grp/program/LLMCL/.deployment/osft_stage_refresh_mechanism_20261024_view`
- unit：`llmcl-onevision-osft-stage-refresh-mechanism-20261024.service`
- 观测时controller3728445/start755358934、Fixed worker3731901/start755415692，UID1084／PGID同PID／GPU0。
- [冻结协议](summary/final_weight_reduction_20261024/osft_stage_refresh_mechanism_v1/PROTOCOL.json)，READY `d276508a82e51855b3c7e4ac94f8fa3329167c7bec6cf190368c09f00271d4a5`。原预算与退出规则保持，不续跑、延时或版本救跑。
- 全三段完成退出后原dispatch会唯一核验。结果消费者[analysis/osft_stage_refresh_outcome_20261024.py](analysis/osft_stage_refresh_outcome_20261024.py)已准备但**未执行**；只能在完整VERIFIED后一次消费，不读部分质量，不重评分／bootstrap。

## 不能被交接掩盖的结论

1. 正式8组结果及完整权重列已整理；OSFT更高刚习得不等于已经识别参数空间原因。
2. ShortFixed只省训练状态，最终权重没少；小DEV点高，未证稳定改善。
3. 多项真实压缩确实减重，但完整回答质量严重下降；不是成功方法。
4. 冷启动证据仅局部新专家，不是所有基线每域从零或不更新。
5. 现有先例可借用，有意义的机制对照可继续；不能因不确定而无限更正，也不能轮流换压缩工具。
6. 旧关闭点、稀疏统计修复待人类例外保持原状态；不复活旧worker、重旧分数或依据TEST追参。

权威为NEW Git main `/data1/liyapeng/program/LLMCL`；本地为非Git镜像。成果整理提交`2e315388`，结果消费者准备提交`28bfa3c0`。当前交接也会推送并精确同步；数据正文、模型、checkpoint、图像、密钥不入Git。
