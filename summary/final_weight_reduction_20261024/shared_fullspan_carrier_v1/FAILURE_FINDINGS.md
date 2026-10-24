# 固定共享载体检查：捕获缓存超过事前上限，关闭本点

本次首次实际模型尝试在教师激活捕获阶段触发预定的8 GiB tensor payload上限，按冻结协议关闭整个固定构造，不扩大缓存、删减或替换样本、截取token、续跑或自动建v2。没有到达共享因子初始化、恢复更新、物理压缩打包或压缩诊断，因此没有新的压缩质量结果，也不是FiPS功能承接假说的质量阴性。故障关闭和证据封存不作为论文方法进展。

STARTED绑定OLD UID1084、GPU0、worker3587377/startticks752538740及READY a4d918ab…。ACTUAL_EXIT记录returncode=1、身份消失；receipt的2026-10-24T07:34:40.590961+00:00快照记录service failed/MainPID0、controller和worker均已消失、queue为closed_failure。这是终态时间快照，不是持续存活检查。

冻结worker先保存原父包，再完成65条父诊断，之后才进入32条恢复样本捕获。receipt仅封存父诊断文件65条非空记录的计数和SHA，未读取分数或预测；结合固定源码可推定父65次NLL和65次生成已发生，不将其说成逐调用独立counter证明或已完成评分。原父包只作stat记录，未重复大包hash/加载。具体已完成多少捕获QA或层没有持久记录，32次是计划数而非完成数，正在执行的工作不按零计。updates、cache_index、compact包及完整结果均未产出，恢复更新为0。

hook先把一层输入/输出clone为独立CPU FP16 tensor，再累加storage字节并检查上限；本次在该检查处失败。它不是CUDA OOM、SVD失败或恢复优化失败的证据。8 GiB不是整机RAM上限，也不代表分配从未瞬时越过上限或恰好留下8 GiB有效完整缓存。此次缓存获取方案及资源估计未满足事前限制，不据此判共享/稀疏权重是否能保留回答功能。

STARTED到FAILURE文件mtime差132.307458878秒是文件时间区间，不是完整RUN；controller的退出观察上界为133.798528592秒，两者不相加。receipt记录26条有效资源行及失败轨迹观测最大值：RSS/HWM 11,164,413,952 bytes、GPU allocated 1,910,975,488 bytes、reserved 2,713,714,688 bytes、NVML进程观测2,917,138,432 bytes。它们不是已完成载体成本、压缩终包推理峰或受控性能比较。0恢复更新不等于0加载、保存、诊断、视觉计算或捕获费用；tiny门及运输费用仍按原记录另列。

运输记录有两个明确版本：NEW权威TRANSPORT_RECEIPT含NEW监督秒数和archive字节，SHA为3ba91942607a91f8dbfbdece4a8ae70146c457d2b1b8f531647c1b0f21b4acdf；故障receipt列的是OLD接收版，SHA为9ac5db4d1fdb326fd4d8256dcaed985a3711a86f17c80cbce1db4c0a13986ede。去掉上述两个NEW补充字段可重建后者，不能称作逐字节同一文件，也不构成权威准入文件被篡改或覆盖的证据。原TRANSPORT、READY、源码和准入记录均保留，不回写本次故障。

本点结束，不读取父partial质量，不运行原评分/模型/门，不另做缓存修跑或自动下一模型。既有几何/梯度标量不能补出优化坐标与可达功能的缺失反事实，本次故障也没有补上它，不追加旧审计或另换压缩工具。
