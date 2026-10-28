# FullFT_Seq 准备说明（未冻结、未启动）

仅新增隔离候选 `exp/full_parameter_seq_20261028`，面向 LLaVA-OneVision-Qwen2-0.5B-OV 和 Qwen2-VL-2B-Instruct，各自原 nested 1% / 10% TRAIN 与完整 TEST。它是全参数顺序微调朴素参照，不是新方法。原六基线、TaskGrad 正式研究、源与队列均未改变；NEW GPU0 保留。

## 当前源

- `core.py`：`905d2e929e9ed0d6c368bcd5b2a57c40513d740ca7a60bcec658f94721d87464`（245行）。
- `run.py`：`f7164de3cce95f93666e6d6f4982f5309feae3382cf2176f2f864dca73a83f79`（173行）。

两源已在 NEW 上仅执行 stdlib AST/compile 源码检查，未 import 候选/Torch 模块、未调用 main、未加载模型/权重/QA、未评分或训练。baseline_review 对上述实际源独立只读审阅，未发现明确 fatal；这不等于 GPU 准入或执行通过。此前纯 BF16 参数更新草案已被 FP32 母参数方案替换，未运行两种方案比较。

## 训练与部署口径

全体唯一原生 Parameter 参与同一个优化器，无 LoRA。LLaVA 包括 language/head/vision tower/projector/image_newline；Qwen 包括 language/head/visual tower/merger。桥接图与原 core 的 Parameter 对象集合必须完全一致。最终总参数及权重/普通 buffer 字节仍须实际测量，不以可训练坐标、容器文件大小或 FP32 训练重量替代部署重量。

单一待验证数值方案：FP32 全参数、BF16 autocast、原 `HostAdamW`，LR=1e-4、betas=(.9,.999)、eps=1e-8、weight_decay=0、无 GradScaler。FP32 moments 驻留 CPU；模型前后向与原 Adam 数学运算在 GPU，逐参数搬运 moments。使用原激活 saved-tensor CPU offload；过程资源监控记录 CPU/GPU 与主更新耗时。源字节账分别记录 moment H2D/D2H，不冒充实测 PCIe 带宽/耗时。

S0–S18 保持同一 FP32 母图、BF16 AMP 评价；S19 主更新/优化器清理结束后仅一次转换为 BF16，再对完整最终 TEST 实际生成/评分并保存该同一 BF16 图。之后禁止再训练，不恢复 FP32、不建第二模型。中间刚习得/FWT/BWT须保留上述精度边界，不能把中间母图分数称为纯 BF16 图分数。最终 BF16 与既有 FP16 基线同为每系数2字节，但数值不等价；最终完整评分、独立重载和部署测量之前不填其质量/资源列。

原视觉 `frozen_vision.install/prepare` 不用于新方法。Qwen 保留原 raw 视觉输入，在 answer_loss 中参与反传；LLaVA 保留原图像预处理、native feature selection/packing/token长度修复，将一次 QA 的视觉 feature 图直接传入原答案损失，无 detach、无跨 update cache。显式修正 PackedVision 内层和 evaluate 外层 autocast 为 BF16，避免 FP32 母参数把 AMP 关闭。答案截断、EOS、生成 cap、prompt/retrieval/processor 路径继续来自原 runtime。

## 协议与保存恢复

CLI 草案：`LLMCL_ROOT=/data1/liyapeng/program/LLMCL LLMCL_HOST=<new3090 或 new3090_onevision> CUDA_VISIBLE_DEVICES=<1/2/3> python exp/full_parameter_seq_20261028/run.py --condition <pct01/pct10> --backbone <qwen/onevision> --manifest <原manifest绝对路径> --manifest-sha256 <原SHA> --run-dir <本候选runs下新目录>`。此处是接口说明，尚未执行。`--preflight-only` 仅诊断，不能恢复为正式训练；`--resume` 仅接受已完成的阶段暂停，不是自动失败重试。

原 seed42、一轮、batch16/micro1、TRAIN SHA 排序、cosine/warmup、19任务和完整21454 TEST保持。核实原 manifest/split lock/frozen IDs 与逐文件 SHA；TRAIN分别1313/13105。复用原 sparse_results 的74必需格，不将缺格补零。每个完成阶段保存原 full-graph CP（模型、方法、processor、RNG、配置与源摘要），原生 optimizer 每阶段重建，不保存阶段中 optimizer。

从原 native run.py 仅 AST 提取 `pause_at_boundary`：S0评价完成可保存暂停，S1–S18在评价与 CP 提交后协作暂停，S19正常完成。恢复须匹配 paused 状态、pause receipt/latest/config/source/SHA 与一次性 RESUME_REQUEST；S0可恢复。原科学源文件不修改，暂停/恢复不依靠强杀。阶段边界之后未提交的评价文件移入独立 interrupted_evaluations，不能混作当前已封前缀。

## 继承依赖与尚未验证

原依赖根为 `exp/two_backbone_native_cl_20261017_v2`。core 硬绑定：runtime_utils.py `4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240`；onevision_bridge.py `f3b1dd38af3b58dbfa69073da698ea2bf39a3a11850fb1069a80e38d0e3669a9`；checkpoints.py `06c53eab2c6810901f5206edbfb1df9defabdc747b1856ace0afa31e74b37a78`；host_adam.py `5e733e6b87c12eb9331ff23e8919a142f1861a2beec82ec8b0361d878b9aec8c`。还复用原 run.py 的暂停函数、training_validity、sparse_results、resource_monitor，以及 runtime 的 methods/retrieval 等导入依赖；runner 会记录整个原 native 与本候选 py 文件源摘要和副本，正式准入前仍需外部冻结绑定，不能把当前四个 pin 称作完整正式 source READY。

尚无任何 GPU 验证：需要 NEW Ampere/BF16 支持；两骨干真实 QA token/feature 一致性；vision/projector/language 等模块非零梯度与真实权重更新；优化器覆盖所有唯一 Parameter，finite 梯度/矩/权重；HostAdam 矩 dtype/CPU驻留/传输与全量梯度/激活峰；阶段暂停和 RNG 恢复；完整图 CP 重载；S19 转换后实际评分图与保存图一致；完整74格/最终总参数/权重/独立同卡 GPU 资源。不能承诺24GiB必 fit，也不能因潜在容量而将训练范围改成 LLM-only。

预检沿原每任务首条 TRAIN 的19案例，不新增 QA 搜索；此候选保存的 diagnostic probe schema 为 `diagnostic_only`，不能直接声称兼容原 checkpoint_probe.py 所需 task/row/loss 协议。真实重载验证仍待明确接入，当前没有调用原 probe 或冒称其通过。

没有 launch、queue、调度服务、模型操作、真实评分或对运行中 GPU 的任何信号。
