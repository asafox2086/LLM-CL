# ALER 完整十任务结果

运行：`ten_task_cl_20261014_geo_fin_20261014_205644`；划分 SHA：`dab25e3940758cfac3e1bb16da5305ce6393dfa4eb22e1c39d4f99a28fdc4208`。

三个 CSV 都是11行×10个任务：第0行是未训练基座，第1–10行分别是完成T1–T10训练后对全部十任务的结果。EM、Token F1、ROUGE-L统一采用0–100尺度。每个单元来自50条测试QA，共5500条预测；已逐项核对固定测试ID、数量和记录完整性。

原逐题文件及日志保留在服务器 `exp/ten_task_cl_20261014_geo_fin/runs/ALER/seed_42/ten_task_cl_20261014_geo_fin_20261014_205644/`；原始问答引用保持不变。原指标不是各数据集官方leaderboard指标。FWT/BWT等定义沿用本轮固定协议。
