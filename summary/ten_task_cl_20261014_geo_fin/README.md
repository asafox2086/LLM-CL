# 本轮独立实验报告

[已完成结果报告：每方法 10 条任务曲线×11 阶段节点及最终指标](results_report.md)

实验 `ten_task_cl_20261014_geo_fin`；共同 manifest SHA `dab25e3940758cfac3e1bb16da5305ce6393dfa4eb22e1c39d4f99a28fdc4208`。样本划分见 [记录](../../data/TenTaskCL/ten_task_cl_20261014_geo_fin/README.md)。

[实时运行状态](report.md)、[JSON状态](run_status.json)、task_metrics.csv 和 dataset_metrics.csv 均只汇总同一实验 ID／同一划分的真实已完成记录。每任务 EM正确数／总数、TokenF1、ROUGE-L分别保存；只有完整11×10矩阵才能发布AP、ForgetRate、FWT、BWT，不补齐尚未完成的阶段。旧日期和失败重跑的结果不串接。

每题预测及训练loss原日志在 exp/ten_task_cl_20261014_geo_fin/runs/。GPU0保留，最多GPU1–3三方法并发。状态会持续变化；adapter_pending不是已经运行。
