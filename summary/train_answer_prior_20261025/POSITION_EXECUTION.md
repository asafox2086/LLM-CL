# 既定360生成的单次空卡执行

本次没有新训练配方。问题、45原TRAIN ID、唯一循环、四个原完整图、512-token原生成设置、每态首次1200秒及完整后一次判读均沿[既定协议](POSITION_DIAGNOSTIC_PROTOCOL.md)。目标仍是区分“规范化集中”与“同内容换位置后选择改变”，不将绑定稳定性当正确率或PIF效果。

[执行准入](POSITION_RUN_ADMISSION.json) SHA `8e46d2972f7646606a5d365d949dc29594d7d808e72cb27e6e89bb7dc37723d1`只绑定四源、已通过真实输入合同的binding、协议与变换哈希。它不表示GPU已启动或诊断有效。协议末段的“当前无READY”是准备时记录，实际状态以后看本准入、队列ACTIVATION、POSITION_SLOT及模型STARTED／终态，不改科学协议或重置预算。

- `queue_adapter.py`导入既有FOREVER／native调度，只在主CPU入口包装原pipeline.idle。第一次真实ownerless且原free／util条件满足时，先持久化单次slot再启动controller；四态间即使没有CUDA worker也保留同卡，其他GPU仍原分配。它不改baseline job状态、科学源或正在运行的worker，子phase继承admission环境但不装idle hook。
- `controller.py`按1%S15、1%S16、10%S15、10%S16顺序启动四个原worker，各1200秒绝对上限（含Python启动，比worker自身预算更严格），总4800模型秒。同一实际GPU，不训练或改图；任何模型故障／中断／超时不再推进后续状态、不续CP、不自动新版本，只能清理本次身份匹配的私有child进程组。
- 只有完整四态后才启动一次`verify.py`：核全部完整证明与文件摘要后解析360原响应，保留完整45分母、无效覆盖和共同有效ID。原样／循环pair的实际图像特征必须相同；跨保存态原缓存数值SHA是否相同另记证据，差异限制跨轨迹解释，不清缓存或修改原模型去迎合硬门。原F1不重算。
- 完整或故障终态都不可重启。只有supervisor及全部已记录模型child（含异常时last_attempted_worker）实际身份消失、NVML不含对应PID后，才能释放卡；失联或无可靠终态继续保留slot等待核实，不推断训练已停。

CPU接管必须先核旧coordinator身份／单一所有权／KillMode=process／无ExecStop，只停止它本身，保全全部原GPU及各wrapper身份；不与旧分配器并跑。当前文档形成时仅源码及AST／小metadata审阅，没有CPU模型门、模型执行或实际360消费者。

私有模型输出仍位于OLD SHM，非耐久存储。成功消费者仅将公开OUTCOME／VERIFIED原字节复制到本report目录；**当前collector只自动镜像10%基线report，本train_answer_prior目录不在其白名单**。冻结controller注释中“已镜像此report”的说法不准确，不据此宣称耐久。公开结果、原始360回答和完整运行证据均须结果后显式转存NEW并逐摘要核对，不另起一个归档器或据目录存在称已保存。两主表、16组基线及最终完整模型留存继续，当前自研方法总目标未实现。

## 实际接入快照

事前发布79b5260e后已逐字节运输并验证实际metadata合同（0Torch／模型）；unix1792991197.0531332仅OLD旧CPU318427退出，新CPU342620/start768961721/UID1084、unit `llmcl-pct10-baselines-position-old-20261026.service` active/running、Restart=no／KillMode=process。原Qwen Seq4046298、双PAS4049854／4048289、LLaVA CL236681及各wrapper身份前后全同，0GPU信号，原三个入口SHA不变。slot=null，尚未占卡／STARTED／产生新响应，4800模型秒预算未开始。以后看真实proc、slot、各phase STARTED与终态，不把快照永久当live。[接入收据](POSITION_QUEUE_ACTIVATION.json) · [运输与metadata核验](POSITION_QUEUE_TRANSPORT.json)。
