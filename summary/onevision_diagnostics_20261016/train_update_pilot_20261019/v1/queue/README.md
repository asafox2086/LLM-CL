**历史队列：** 已由../priority_v2/queue_state.json接管；旧384025只计划退役，不恢复。[当前操作修订](../priority_v2/SCHEDULE_AMENDMENT.md)。科学worker和全部原确认义务不变。

# TRAIN-only更新对照排队

当前两系数完整确认优先，随后12格训练对照，再回到所有原工作；不根据TEST分数调序。唯一服务仍llmcl-onevision-causal-priority.service，实际入口、进程及READY以ACTIVATION.json和queue_state.json为准。GPU当前运行者自然完成，基线不停止。
