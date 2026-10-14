# ten_task_cl_20261014_geo_fin：独立基线并发实验

当前基座 Qwen2-VL-2B-Instruct；10任务按T1→T10；每任务100训练／50测试，无验证集，一轮，seed42。共同划分详见 [数据划分记录](../../data/TenTaskCL/ten_task_cl_20261014_geo_fin/README.md)，SHA `dab25e3940758cfac3e1bb16da5305ce6393dfa4eb22e1c39d4f99a28fdc4208`。

只使用 GPU1、GPU2、GPU3；**GPU0 按用户要求保留，队列不会调度它**。每个方法占一张卡独立完成十阶段，阶段前及每阶段后都评测全部十任务。队列闲卡时才运行该方法的真实任务GPU前向／反向预检；预检权重丢弃，正式运行从统一基座重新加载。

当前可执行方法：SMoLoRA、PASs-MoE、Hyper-LLaVA、OSFT、SpaRTA、ALER。另两种 PaRSP 和 FV-guided Training 尚未完成适配，注册为 adapter_pending，不能冒称已运行。具体差异见 [方法适配记录](METHOD_ADAPTATIONS.md)，实时状态见 [本轮汇总](../../summary/ten_task_cl_20261014_geo_fin/report.md)。

调度入口：`.vision-env/bin/python exp/ten_task_cl_20261014_geo_fin/parallel_queue.py`。现有后台调度器已经启动，不要再启动第二份；`.queue.lock` 防重复。allowed_gpus／reserved_gpus 写在 execution_policy.json。

日志及逐题结果：`exp/ten_task_cl_20261014_geo_fin/runs/<method>/seed_42/<run_id>/`；汇总、阶段任务／数据集分数和图：`summary/ten_task_cl_20261014_geo_fin/`。每次重跑分配独立 run_id，不能将失败进程已训练的阶段续接为同一次完整实验。

不保存模型、优化器或恢复 checkpoint，方法必需参考与统计只保存在运行RAM；冻结视觉输入缓存是固定基座的输入特征缓存，不是本轮训练权重。进程中断后必须从基座重新训练；保留失败／暂停运行的日志，不混入其他运行。

GPU0 上的 SMoLoRA 在训练前初始评测阶段按用户要求暂停退回队列，没有丢弃已训练参数。PASs-MoE 初始评测曾因作者导入环境引起默认ASCII写文件失败；本轮明确使用UTF-8文件及进程环境，失败日志留存，后续独立重跑。旧 ten_task_cl_20261012 的队列和结果均不参与本轮。
