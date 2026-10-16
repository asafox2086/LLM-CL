# 五数据集原生任务持续学习：2026-10-16 用户确认

此规则覆盖旧六领域、全阶段全测及不存checkpoint的要求。

- 实验ID：`five_domain_native_cl_20261016_sparse`。旧实验用户主动停止，禁止重启或拼接其预测。
- 移除AQUA，保留TextVQA、SLAKE、RSVQA-LR、FinMME、CODA-LM。移出Hyper-LLaVA，加入ER-LoRA与CL-MoE，共十方法；SMoLoRA保留。
- Qwen2-VL-2B-Instruct，seed42，一轮，无验证集；两组为全量训练、固定1%训练，均使用完整且完全相同的测试集。
- 固定19阶段：TextVQA；SLAKE的Modality/Position/Organ/Size/Abnormality/Quantity/Plane/Shape/Color/KG；RSVQA-LR的rural_urban/presence/count/comp；FinMME；CODA-LM的general/regional/suggestion。
- 类别标签来自原标注；上述**顺序为研究预先固定**，不是声称作者给出了课程顺序。金融答案格式不扩展为训练阶段，医学OPEN/CLOSED和语言维度仅作分组统计。
- 全量131019训练QA，1%1313训练QA，共同测试21454 QA。每条记录恰好属于一个训练/测试阶段；不重复训练、不改问答，不因原生混合题型筛除样本。
- 沿用父六领域实验的原始来源划分及冻结1%样本ID，仅按原生任务划分成互斥阶段。父采样规则为seed42、按原题型比例最大余数分配ceil(1%×各数据集训练量)、哈希选ID。不得为单个方法重新抽样或把1%解释为每阶段至少100条。
- 正式运行前冻结full/pct01的manifest、frozen_ids和split_lock.json，设只读；启动严格校验manifest/QA/顺序哈希。两服务器必须一致；所有新方法独立从基座开始，不使用诊断preflight更新的权重。
- 初始测所有19任务；阶段1–18只测刚学完的任务与下一个任务；阶段19测所有任务。共74组测试，80316条逐题预测。未评测单元格为null，不补0、不伪造完整20×19矩阵。
- 报告最终各原生任务成绩、各领域按测试QA数加权的最终成绩、领域宏平均、原生任务等权AP/FWT/BWT；明确两种平均的权重不同。不报告历史峰值ForgetRate，也不能据稀疏结果画完整遗忘曲线。统一EM/F1/ROUGE-L是辅助文本匹配指标，不冒充各来源的官方指标。
- SeqFT-LoRA在两个条件均保存全部19个训练阶段checkpoint；其最终checkpoint就是第19份。其他九方法在每个条件仅保留最终第19阶段checkpoint。无需再重复导出同一份最终权重。
- checkpoint保存完整模型/架构与方法图，包括合并权重、冻结专家、路由及方法运行状态、processor、RNG、配置及文件SHA256；不能只保存当前requires_grad参数。每个新方法/条件正式启动前，必须通过19任务真实GPU前向/反向，以及独立进程的2B图重载、损失等价和有限梯度检查。
- 阶段边界会重建optimizer/scheduler/scaler，故阶段checkpoint不额外保存旧optimizer。SeqFT可从完整阶段边界恢复；其他方法因仅存最终权重，中断后不能恢复未完成阶段，必须从基座新建run，不能拼接旧预测。推理加载仅适用于自己可信的pickle文件，且须使用相同代码/运行库。
- 持续记录CPU RSS/PSS/USS、GPU allocated/reserved/NVML，训练前/后具体大小、变化与峰值；loss、真实梯度范数、学习率、token数、逐题答案和参考。内存结束快照先于checkpoint导出。
- 两台服务器共同运行。新3090机GPU0保留、GPU1留给自研，GPU3全量，GPU2抽样；旧2080Ti机GPU0–3均可抽样。每卡一进程，原方法先GPU预检，显存不足回到新机，不减少测试集或替换算法。
- 两台机器运行Torch2.6.0+cu118 / Transformers4.46.3 / Tokenizers0.20.3 / Cloudpickle3.1.2。不同硬件的墙钟时间分别报告，不把卡型差异解释为算法速度差异。
- 日志/参数：`exp/<ID>/runs/<condition>/<method>/<run_id>/`；报告：`summary/<ID>/`；冻结划分：`data/FiveDomainCL/<ID>/`。服务名`llmcl-native-five.service`；旧`llmcl-six-domain.service`停用并禁用自启动。
- 旧机任务与新机分工互斥。旧机完成后通过LAN收集结果和最终checkpoint到权威新机，并复核SHA256。私钥目录`.deployment/`、QA大文件、模型和运行权重禁止提交Git。
