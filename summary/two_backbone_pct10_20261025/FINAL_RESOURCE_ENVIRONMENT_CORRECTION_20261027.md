# 10%最终参数资源测量：环境故障与独立补测

原两骨干测量已failed_closed并释放，不是仍等待。LLaVA worker启动2.735秒后，导入runtime_utils所需PIL失败；已初始化CUDA上下文，但未到checkpoints.load，未加载模型或生成答案。Qwen没有启动。原失败和耗时完整保留，两行GPU/延迟/速度仍为缺测，不能填0。

与Qwen OSFT的初始化SVD显存溢出不同，本次是测量环境缺依赖，不能当作算法内存不足的论文证据。

只读实查确认：原`.conda-env`无Pillow；既有`.vision-env`含Pillow12.3.0。两者torch2.6.0+cu118、transformers4.46.3、accelerate1.2.1、cloudpickle3.1.2一致。`.vision-env`实际transformers解析到原`.conda-env`路径，15个原绑定library文件逐个SHA完全一致。两个原binding的validate_metadata及原checkpoint/methods/runtime_utils/bridge/frozen_vision导入链均通过，未初始化CUDA、加载模型或生成。

据此准备独立补测：保留原worker、两个最终CP、原bindings、未fold图、同五raw输入、每题32tokens、原资源及计时定义，使用既有`.vision-env`与新输出目录。原failed_closed及no_retry记录不重开，不改任何训练、生成、评分参数，不重复F1评分。Pillow作为原缺失依赖补齐必须披露；实际输入准备及GPU结果仍待执行核验，包版本相同不能替代实测。

补测只能接在当前type辅助研究真实释放后，不能抢其GPU。此文件不是启动收据；以新slot/STARTED/COMPLETE为准。目前没有新GPU指标。

证据：[原失败](FINAL_RESOURCE_FAILURE_EVIDENCE_20261027.json)、[子agent审查](FINAL_RESOURCE_FAILURE_ENVIRONMENT_REVIEW_20261027.md)。环境实查记录与导入证明见`FINAL_RESOURCE_ENVIRONMENT_CHECK_20261027.json`。
