# 真实个人电脑部署的最小前置检查

本轮只验证已封存 K61 dense OSFT 完整包在个人电脑 CPU 的兼容性，不是新的压缩方法或质量/速度结果。相较反复巡检，这一步补的是目标设备实际能否运行、后续公平测量能否开展的缺口。

设备清单见 DEVICE.json：i7-13650HX、约39.7GiB可见内存；RTX4060 Laptop的实际显存8188MiB，采集时已有其他活动，本次不使用GPU。清单的空闲量只是瞬时快照，不是部署峰值。

复用1,803,323,773字节的原完整包，保持原CPU FP16/SDPA；服务器的Torch CUDA build换为相同2.6.0的Windows CPU build，平台差异公开。隔离环境与完整依赖另计磁盘字节。只处理原SLAKE:5162医学TRAIN校准请求一次，原teacher见过该题，不生成答案、不更新参数、不访问DEV/TEST。具体SHA链、停止边界与预算见PROTOCOL.md及REVIEW.json。

ENGINEERING_LAUNCH记录启动，只有READY后才允许模型进程；COMPLETE/EVIDENCE或FAILURE记录真实结果。安装和传输尚未完成时不称模型已运行。私有模型/原图/QA/环境/日志在.deployment/pc_cpu_admission_20261022/v1，不进入Git。

后续完整比较仍需PASs的公平动态部署包及fresh功能核验，并保留质量接近的可合并OSFT参照，同机、同线程、同输入及生成预算测量。单个OSFT请求成功不能填正式部署列、不能宣布端侧优势。原GPU1四组质量小测与服务器CPU两支结构对照继续按冻结条件执行，完成后再统一分析。
