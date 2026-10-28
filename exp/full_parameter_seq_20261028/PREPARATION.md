# FullFT_Seq 准备说明（未冻结、未启动）

仅新增隔离候选 `exp/full_parameter_seq_20261028`，面向 LLaVA-OneVision-Qwen2-0.5B-OV 和 Qwen2-VL-2B-Instruct，各自原 nested 1% / 10% TRAIN 与完整 TEST。它是全参数顺序微调朴素参照，不是新方法。原六基线、TaskGrad 正式研究、源与队列均未改变；NEW GPU0 保留。

## 当前源

- `core.py`：`67ec0cc0ba660201b24f842556f8291afa4153d1fee17ee1ccb86f7255a7a2d4`（299行）。
- `run.py`：`c603aecee855d6e269048e4b2c6670ef78b364b6a92a7abe9b63e438f850a2f1`（174行）。
- `reload_probe.py`：`779aa9d0c01237774c41d69487f84cb9a7e7b2b41fc64d5f12af67edaab8b64c`（104行）。

三源已在 NEW 上仅执行 stdlib AST/compile 源码检查，未 import 候选/Torch 模块、未调用 main、未加载模型/权重/QA、未评分或训练。baseline_review 对上述实际源独立只读审阅，未发现明确 fatal；这不等于 GPU 准入或执行通过。此前纯 BF16 参数更新草案已被 FP32 母参数方案替换，未运行两种方案比较。

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

预检沿原每任务首条 TRAIN 的19案例，不新增 QA 搜索。独立重载 schema 已实现而未执行；不直接调用原 checkpoint_probe.py。19次正常更新和 optimizer 清理后，用最后一条已用 TRAIN 图像 QA 在 eval/BF16 AMP 下保存 loss、完整 prepared 输入 tensor SHA/细节、原参数 SHA/layout、原 row SHA 与 producer identity，写入原完整 CP 的 fullft-checkpoint-probe-v1 probe。不是 TEST、新 split 或 F1。

独立 CLI：`python exp/full_parameter_seq_20261028/reload_probe.py --run-dir <同预检RUN> --config-sha256 <config字节SHA> --checkpoint-sha256 <原CP SHA> --checkpoint-metadata-sha256 <latest.json字节SHA>`。保留同 NEW host/backbone/GPU0规则。先核 config/source/meta/preflight receipt，再由原完整 CP loader 核全文件 SHA 并加载，核同 QA 的 loss（原 probe 容差 max(.005,abs(loss)*.002)）、prepared 输入、参数/layout 摘要精确一致。随后 TRAIN 模式重新打包同 QA，做一次反传，核 language/vision/projector 非零 finite 梯度、0 optimizer、参数 SHA 不变；不能复用 eval 的无图 features。原 CP 不删除。GPU_RELOAD_STARTED、失败文件和 checkpoint_reload_proof 独占，拒绝重复执行或覆盖，不自动失败重试。

仅增加已有 TRAIN QA 的参考 loss 前向、重载同题前向和一次反传，0新题/TEST/F1。检查全部原 Parameter 的掩码/数量/布局及原 core 对象集合，无 LoRA/冻结视觉缓存。stage1 预检重载不能替代未来 S19 BF16 部署重载/资源。

下一次实际 GPU 预检限定 LLaVA 1%，尚未准入或执行；未来经原 NEW allocator 真正 idle GPU1/3 插入 preflight→reload后释放，不自动训练或四组批跑，不抢正式研究/原基线/NEW0。本轮只发布准备，尚无 allocator hook。

没有 launch、queue、调度服务、模型操作、真实评分或对运行中 GPU 的任何信号。
