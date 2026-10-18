# 2026-10-18磁盘写满事故与保全续跑

本轮现场核验发现，新服务器多个作业于凌晨03:55前后开始遭遇`OSError: [Errno28] No space left on device`，CPU研究队列和结果观察器也退出。失败JSON本身写不出，仅留下0字节临时文件；因此“没有needs_attention.json”并不等于健康。未将写盘失败当算法数值失败，也未继续宣称六个基线都在运行。

当前/data1恢复约1.1TB可用；本轮对项目根/exp/summary/新Qwen run目录做1MiB写入、fsync、逐字节回读验证，通过后才恢复。没有通过删除模型/数据/旧checkpoint来释放空间，也不推断何人或何种操作释放了空间。

- Qwen SeqFT原第9阶段完整checkpoint字节数/SHA及框架源码/划分精确核验，通过原`--resume`从该边界接续。失败stage10更新事件先完整归档，后续无权重的预测由原恢复逻辑隔离，不拼接失败轨迹。
- Qwen ER没有中间完整参数，**没有声称续接第11阶段**；复核原预检/重载证明后，另建训练run从头重跑。同TRAIN/TEST/代码/超参，保留旧未完成run。
- 原新机CPU基线pipeline先核验KillMode=process、确认GPU2/3无作业、保存状态，才重新接纳上述作业和失败的预检。其他用户GPU进程startticks保全；旧机四个活动OneVision基线不动。
- 研究CPU队列与学习复用观察器恢复的是相同冻结源码；case18故障完整日志保留。原worker对未完成case重新建立独立run，不拼接旧参数/预测。GPU1被其他用户占用时继续等待。

恢复脚本：[recover_storage_incident_20261018.py](../../analysis/recover_storage_incident_20261018.py)。`AUDIT.json`/`APPLIED.json`/`RESEARCH_REACTIVATION.json`为具体证据；大日志只留服务器。恢复后SeqFT实际进入stage10评测，ER新run初始全评测，GPU2/3进程真实存在。复用同目录的SeqFT跨PID总内存端点不能直接相减，需使用原始同PID窗口；不据错误汇总宣称内存优势。

正式OneVision1%此时2/10完成（SeqFT、ER，19阶段/21454最终QA与最终参数俱在）；4运行（CL-MoE14、PASs-MoE15、PaRSP18、OSFT17阶段评测），4等待。完整主表按原Answer F1，不混用研究AnyRes155题分数。
