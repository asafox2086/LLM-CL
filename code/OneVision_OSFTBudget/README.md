# 原SVD机制对照与数值修复

当前冻结入口：analysis/onevision_osft_budget_entry_v3.py，核心core_v3、trial_v3、report_v3和queue_v3位于analysis/，唯一nohup队列CPU86675。v1 autocast故障及v2 CUDA默认SVD误差失败独立留证；v3作者CPU FP32分解后传回原设备、严格FP32重建，原门槛未变。

同540672训练坐标、原1313TRAIN／19阶段／两个完整21454条件保留；数值／CPU传输成本计费，没有性能或内存优势结论。两条件是机制对照，不是预定两个模块。[协议](../../summary/onevision_diagnostics_20261016/osft_budget_20261018/v3/PROTOCOL.md)。
