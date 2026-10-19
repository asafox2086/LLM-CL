# GPU1-only端侧研究调度

2026-10-19最新用户目标为手机／普通CPU部署。唯一研究nohup服务仍为llmcl-onevision-causal-priority.service，新CPU为739098；实时以queue_state.json及实际PID/startticks为准。

旧CPU502584只计划退役，KillMode=process保全所有GPU、正式基线与只读发布器。新入口analysis/onevision_edge_gpu1_queue_20261019.py仅将设备选择限制GPU1，不热改原科学源码。GPU1被他人占用时健康wait，不转用GPU0；6CPU设备控制通过。

原12格TRAIN-only→连续RITE19TRAIN／21454→random完整及全部旧任务保留，原数据、预算、核验门槛和顺序未改。不存在新F1或部署收益；主表未测列保持空白。未来研究从一般性问题出发，并以实际手机／CPU部署成本及完整质量共同验证。
