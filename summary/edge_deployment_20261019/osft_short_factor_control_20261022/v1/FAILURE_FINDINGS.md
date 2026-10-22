# v1已真实失败，保留原源与全部成本

STARTED文件mtime为2026-10-22 03:07:21.420307UTC，FAILURE为03:08:17.357054UTC，差55.936747秒；这只是文件时间区间，非完成的ResourceMonitor RUN。原READY61cbee3f...与科学源均未修改。

19原FIT图文门槛（38NLL+38短生成）完成。首个16QA minibatch前后向后、clip与optimizer之前，保存首步梯度时因private/gradients尚不存在而停止。删除重复JSON写入时，我漏掉了其创建父目录的作用；此前门槛与静态评审没覆盖该路径。这是实现故障，不是短侧固定的质量负结果。

训练事件0、optimizer更新0、checkpoint0、DEV生成0；模型加载、门槛、首批前后向仍有成本，不能称零开销。原STARTED/FAILURE/needs_attention/私有日志和资源记录保留，当前artifact SHA见独立v2的V1_FAILURE_SEAL。没有恢复v1、拼接梯度/状态/预测或放宽门槛。

机器约03:28UTC又发生重启，boot由8052...变为f801...；目录错误发生在它之前，两事件分开解释。后续只在独立v2修复保存路径及必要回归，原科学条件/原基座重新执行，不计新科学条件、不按未发生的DEV结果追分。v1不再启动。
