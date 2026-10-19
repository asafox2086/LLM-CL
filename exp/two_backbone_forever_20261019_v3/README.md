# FOREVER正式图文运行框架

沿用正式V2共同数据、模型、输入、训练和评价，作者图文回放内核与双骨干v2预检完全相同；本v3仅增加阶段事务与资源分段。来源及限制见[正式协议](../../summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/PROTOCOL.md)。

由`analysis/forever_formal_queue_20261019.py --backbone onevision|qwen`的既有nohup实例管理。先核验实际进程、该主机state与完成证明；已有作业不重复启动。源由FORMAL_ADMISSION固定，后续故障需新版本。

`committed_latest.json`是唯一可恢复提交点；每阶段独立事件、持久提交和旧权重删除按顺序执行。恢复使用原SHA／图结构／RNG加载逻辑，未提交分支整体归档。资源按PID分段，窗口损坏不能补造或跨PID推导内存增长。仅保留最新／最终权重，历史SHA保留。原始QA／图像／权重／私钥禁止提交Git。
