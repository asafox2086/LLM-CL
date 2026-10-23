# 固定规模回答引导归并：完整负结果，已关闭

两流Spectral/Answer均已完整完成1213FIT、19阶段、87正LR更新及各100已探索DEV，并通过原唯一验证器。实际完整权重从893675552降至662833184（−25.83%），但五域宏F1分别只有0.9103938463/0.8801465055，远低于旧合法FIT-only锚43.7919874085。这是减重但未保留回答质量的固定配方负结果，不能称轻量高性能方法。

Answer在全部19阶段提出不同mask，整体回答NLL二选一全部选回谱候选；没有空集合或相同mask fallback。额外138门反向QA、276候选前向QA计费，不能声称回答mask发挥了保留质量的作用。两流实际结果未建立等同，根因尚未由本实验隔离。

当前无研究worker，原unit已自然inactive/exit0、canonical queue complete_verified。按预定出口关闭本r200/b16/原LR配方，不全1313/21454、不追rank/LR/库、不自动补恢复或v2，不恢复旧worker。所有旧负条件仍关闭、正式基线保全、科研目标未实现。

结果详情见[FINDINGS](FINDINGS.md)、[OUTCOME](OUTCOME.json)、[RESULT_REVIEW](RESULT_REVIEW.json)和[RESULT_PUBLICATION](RESULT_PUBLICATION.json)。原PROTOCOL、五科学源及READY完全不改；原PUBLICATION/LAUNCH_OBSERVATION/PHASE_OBSERVED是冻结与启动历史。VERIFIED SHA90942989b8518e318f82c66ea3abfeb5b06249cbbe8d256aa88f367d4b329596，OUTCOME SHAae84b2b852c9c75e68852ad83606b0425b6aed26fa3fe647d0cae6fd7cec07ff。

DEV已探索、单seed、每域20QA，不称新留出、显著、稳定、等效或完整TEST结果。细分成本与进程内存观测均在FINDINGS公开；32MiB库上限不是整机RAM上限。原verifier/评分/bootstrap不重跑，Ours/FWT/BWT不填。

# 以下为原冻结与启动历史，已由上面的完整结果覆盖

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
