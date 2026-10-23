# 固定规模回答引导归并：当前状态

## 问题

同样更小的完整模型，阶段末按原回答损失保留方向，是否比普通谱归并保留更多独立DEV质量，并值得新增的历史信息和计算费用？这是框架可行性检验，不是OSFT优化或原创性已经成立。

## 固定设计

两个完整流程：Spectral和Answer。共同未训练初态；72个语言FFN矩阵物理因子替换，r200；阶段内只学临时b16，结束后物理归并回相同每层200维，不保留原dense残差。设计完整规模893675552→662833184（约25.83%减少）；临时训练6635520个参数。实际0.9B构造、质量和成本尚待完整运行验证，不把设计数说成实测成果。

原1213FIT/19阶段/87更新/100已探索DEV，每流单独完成；不读官方TEST、不用正式teacher。Answer只对最多2个当前QA和最多8组旧QA提出门梯度mask，再与谱mask做实际物理切片后的整体回答NLL二选一。当前16MiB、旧库32MiB，新增额外曝光最多570次，全部计费；最终清理bank/current/LRU。各流模型总时限7200秒、两流14400秒，失败整个先导停止，不临时追rank/LR或补恢复。

## 已完成与未完成

必要性和五源静态审阅通过；唯一tiny-HF CPU准入3.056610914秒（整个freeze入口23.365613秒），两流各两阶段四更新、物理归并与完整图重载通过，CUDA未初始化。这是实现准入，不是新F1或实际大模型压缩成果。READY SHA bb00ee0d5c50b9854f01fdb6e5a6f8efd50f3b547409e04487ae25d4b25bec06。

调度服务已激活，最近实查UTC 2026-10-23T11:06:33.171919+00:00，CPU 830049/start9629402，phase=wait。STARTED={'Spectral': False, 'Answer': False}，COMPLETE={'Spectral': False, 'Answer': False}。这些是带时间的真实/proc与unit快照，不永久代表live；当前无质量读取。LAUNCH_OBSERVATION保留最早启动时queue尚未出现的快照，PHASE_OBSERVED是后续状态，缺state的早期观察不是故障。

## 判据与行动

实际最终完整权重须≤原锚75%；Answer五域宏严格高于Spectral且高于既有合法FIT-only最高锚43.79198740853657，所有完整性/预算通过才保留后续必要性评审。未过关闭这一固定配方，不追加恢复/rank/LR/旧备选。逐域/逐题和mask差异、fallback必须全报告；低anchor NLL不算科学成功。DEV已探索，单seed、20QA/域，不称显著/稳定/等效，不自动1313/21454，不填Ours。

## 证据与保全

PROTOCOL及五源由READY冻结，NECESSITY_REVIEW/SOURCE_REVIEW/CPU_GATE/ADMISSION_REVIEW是准入链；只有后续COMPLETE与唯一VERIFIED才是完整结果链。所有旧研究仍关闭，正式基线和他人GPU进程保全，研究仅NEW物理GPU1一流/最多占NEW三卡。

首次激活包装器把审核decision误猜为简单accept，在调用服务前断言停止；保留原包装器，按实际更具体的accept_CPU_admission_and_guarded_single_stream_scheduling修正读取。未改冻结五源/READY，没有重复CPU准入或实际模型尝试。
