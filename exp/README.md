# 实验执行与技术细节

面向阅读结果的说明与进度放在 [summary](../summary/README.md)；本目录记录实现、参数、适配和复现步骤。数据/评价总协议为 [rules/001_experiment_protocol.md](../rules/001_experiment_protocol.md)。

## 当前试跑：T5-Large（2026-10-07）

当前四个方法统一改用原版 `google-t5/t5-large`（通常称 770M），固定 revision `150ebc2c4b72291e770f58e6057481c8d2ed331a`。旧 Llama-2-7B 实验已按用户要求停止，结果、预测和当时的代码快照均保留；Phi-2 仅下载并准备配置，未启动正式实验。

```bash
bash exp/run_olora.sh --config exp/configs/olora_t5large.json --model model/t5-large --gpus 0
bash exp/run_migu_lora.sh --config exp/configs/migu_lora_t5large.json --model model/t5-large --gpus 1
bash exp/run_sapt_lora.sh --config exp/configs/sapt_lora_t5large.json --model model/t5-large --gpus 2
bash exp/run_seq_lora.sh --config exp/configs/seq_lora_t5large.json --model model/t5-large --gpus 3
```

已经运行时不要重复执行。仍使用 nohup + setsid；每个方法独立目录 `exp/result/superni_generation7_v2/<method>_t5large_order1_seed42_epoch1/<UTC时间戳>/`，保留同样的日志、回答、逐条评分和恢复 checkpoint。

本轮是统一协议适配试跑，数据成员、七任务顺序、1000/200/500 配额、seed 42、训练 batch 16/micro batch 1、1 epoch、lr=1e-4、rank=8、alpha=32、dropout=0.1 和 greedy 最大 256 新 token 均保持不变。模型和 tokenizer 已改变，不能把旧 7B 结果拼进新矩阵。具体差别如下：

- T5 encoder 只接收 prompt（最多 768 token，末尾 EOS）；decoder 独立接收右移答案，目标最多 256 token，PAD=0、EOS=1，没有 BOS。T5 使用相对位置，不把模型的 `n_positions=512` 当作硬性位置表长度。
- 使用 `AutoModelForSeq2SeqLM`，目标 loss 不再按 causal LM 方式额外移位；按实例平均保持原协议。生成输出只去除 decoder start token，不能按输入长度裁切。
- 采用 GPU FP32，不量化、不启用 AMP，以避开 T5 的 FP16 数值稳定性风险；评价 batch=8。同一轮四方法设置完全一致。
- LoRA 接入 encoder self-attention、decoder self-attention 和 cross-attention 的 `q/v`。固定 LoRA 共 2,359,296 参数，O-LoRA/SAPT 每阶段新增一组。MIGU 按每层输入来源使用 encoder 或 decoder 的有效 token mask；cross-attention 的 V 使用 encoder mask、Q 使用 decoder mask。
- SAPT 保留现有 prompt embedding max-pool 路由与 ARM 适配；query/key 维度由 T5 的 1024 hidden size 确定。生成器输入 `[Gen] + EOS`，独立 decoder 重建 Train 原始 input，仍是 1000 样本×1 epoch 和 128 伪文本。不是 SAPT 官方 T5 实现的逐项数值复现。

验证入口：`exp/tests/test_t5.py`（label/生成对齐、MIGU mask、第二任务中断恢复）、`exp/tests/smoke_t5.py`（四方法七阶段与 SAPT 六次 reflection）、`exp/tests/preflight_t5.py`（真实 T5-Large 长输入、阶段 6/7 更新、显存、SAPT 全预算生成器）。测试输出不进入正式成绩，验证记录归档于 `exp/result/validation/t5large_20261007/`。四方法均通过七阶段小模型及断点恢复；真实模型四方法均通过长输入更新/生成，SAPT 另通过 1000 样本、63 更新及 128 条非空伪文本的完整辅助预算检查。

论文任务对照和本轮差异见 [T5 试跑协议](../rules/002_t5_large_trial.md)。下文 Llama/NF4 内容记录旧轮次。

## 1. 历史配置：Llama-2-7B 四个并行实验

| GPU | 方法 | 方法文件 | 启动脚本 | 配置 |
|---|---|---|---|---|
| 0 | O-LoRA | `code/olora.py` | `exp/run_olora.sh` | `exp/configs/olora_first.json` |
| 1 | MIGU-LoRA | `code/migu_lora.py` | `exp/run_migu_lora.sh` | `exp/configs/migu_lora_first.json` |
| 2 | SAPT-LoRA | `code/sapt_lora.py` | `exp/run_sapt_lora.sh` | `exp/configs/sapt_lora_first.json` |
| 3 | SeqLoRA 基础对照 | `code/seq_lora.py` | `exp/run_seq_lora.sh` | `exp/configs/seq_lora_first.json` |

每个脚本只创建一条持续学习主线。四个方法独立使用同一份冻结基座权重，各自拥有方法状态，互不继承已训练参数。三个已启动的论文方法继续运行，不因新增 SeqLoRA 而重新启动。

```bash
bash exp/run_olora.sh --config exp/configs/olora_first.json --model model/Llama-2-7b-hf --gpus 0
bash exp/run_migu_lora.sh --config exp/configs/migu_lora_first.json --model model/Llama-2-7b-hf --gpus 1
bash exp/run_sapt_lora.sh --config exp/configs/sapt_lora_first.json --model model/Llama-2-7b-hf --gpus 2
bash exp/run_seq_lora.sh --config exp/configs/seq_lora_first.json --model model/Llama-2-7b-hf --gpus 3
```

这些命令默认使用 `nohup + setsid` 后台启动，标准输入断开，输出写入日志，立即返回 PID 和日志路径。已经运行的实验不要重复启动。脚本统一使用 `.conda-env/`；新实验产生新 run_id，恢复时使用 `--resume`。调试时可用 `LLMCL_FOREGROUND=1` 前台运行。

`launch_olora.py` / `run_olora.py` 现为四种方法共用的调度和训练引擎，保留原文件名以兼容已有入口。它们按配置的 `method` 加载对应单文件方法。模型指纹写入采用唯一临时文件和原子替换；汇总程序使用文件锁，支持多个方法并行结束。

## 2. 固定比较条件

- 基座：`meta-llama/Llama-2-7b-hf`，当前本地权重来自 revision `01c7f73d771dfac7d292323805ebc428287df4f9`；每次按文件 SHA-256 核验。同一 tokenizer，禁止换成 Chat 模型。
- 精度：基座 NF4、double quantization、冻结；FP16 计算，适配器参数 FP32。梯度检查点开启，AMP 初始 scale 1024。
- 数据：`superni_generation7_v2`，同一 manifest、相同实例、相同顺序，每任务 1000/200/500。没有重采样或方法专属测试集。
- 顺序：XSUM → Quoref → EVALution → PersonaChat → GLUCOSE → Reddit TIFU → SciQ；seed 42。
- 主任务预算：每任务 1 epoch，lr=1e-4，AdamW，effective batch=16，micro batch=1，尾批保留；每阶段 63 更新，总计 441 更新。warmup 为阶段更新数的 3%，向上取整；线性衰减。
- LoRA：q_proj/v_proj，rank=8、alpha=32、dropout=0.1。方法容量差异见下文，不能宣称总参数量相同。
- prompt≤768、目标≤256 tokens，目标部分 CE；每实例先做 token 平均，再做实例平均。模板、截断、计分保持一致。
- 当前 Dev 选模；Test 不调参。基座初测与七阶段各测全部任务，生成同一规格 8×7 矩阵。统一 greedy、eval batch=2、max_new_tokens=256。
- 评价协议 `cl_standard_fwt_v1`：AP、F.Rate、BWT、训练前 FWT，仅需主线，不运行用于 FWT 的独立单任务对照。

这是统一预算下的首轮方法适配比较，不是各论文原始表格的复现。辅助训练/回放的计算、额外参数和存储必须单列。

## 3. O-LoRA

对应 *Orthogonal Subspace Learning for Language Model Continual Learning*。源代码为 `source_code/O-LoRA-language-src`；原 `O-LoRA-src` 是另一篇视觉论文。

每阶段新增一组 LoRA，冻结旧组；推理累加已有组，不使用任务标签选择模块。正交项为各目标层历史 A 与新 A 乘积的元素绝对值之和，系数 0.5；L2 系数 0。七阶段最终有七组适配器。原运行的代码指纹已经保存，新增方法不会改变它已加载的训练代码。

## 4. MIGU-LoRA

对应 *Unlocking Continual Learning Abilities in Language Models* §3.2–3.3。依据 `source_code/UNLOCK-MIGU-src/src/uie_trainer_lora.py` 的激活收集、`accelerate_local.py` 的梯度遮罩及论文中的 LoRA 接入规则。

使用固定的一组 LoRA，任务切换时继续更新同一参数，不增加适配器。第一阶段不遮罩；后续阶段在每个有效训练 batch 上：

1. 缓存 A 输出 `xA` 和完整线性层输出 `xW + (alpha/rank)·xAB` 的绝对幅度。B 的遮罩使用完整输出，不使用单独的 xAB。
2. 按非 padding token 累积每个输出维度的幅度，覆盖全部 micro batches。按 token 求均值与求和对当前层的分位数排序等价；与归档实现一致，不额外按每个 token 的通道总和归一化。
3. 用 `torch.quantile(magnitude, 0.7)` 作门限，保留 `>=` 门限的维度；并列时保留比例可能大于 30%。0.7 为论文消融推荐值，未用本实验 Test 搜索。
4. 在 unscale 后、梯度裁剪和 AdamW step 前，将对应 A/B 输出行的梯度置零。

梯度检查点重计算不重复统计激活；评价阶段关闭统计。只遮罩梯度，不清除 Adam 动量，和归档实现的操作一致，因此不能说被遮罩参数在存在历史动量时绝对不移动。任务间仍按统一协议重置优化器。没有记忆库和回放。

LoRA q/v、rank/精度和训练预算按公共协议适配；原论文的 T5 分类任务分数不作为本实验对照结果。

## 5. SAPT-LoRA

对应 *SAPT: A Shared Attention Framework for Parameter-Efficient Continual Learning of Large Language Models* §4.2–4.3，包含 SALS 和 ARM，不是只加一个路由器的消融版。依据 `source_code/SAPT-src/src/llama_prompt.py`、`cl_trainer.py` 和 `gen_script_superni_llama.py`。

### 5.1 共享注意力 SALS

每阶段新增 LoRA 和一个 task key，冻结历史 LoRA/key；共享 query 投影持续学习。对完整的公共 prompt（不含答案）做 frozen embedding、非 padding token max-pool，再投影得到 query。按归档 Llama 实现使用 `Linear(d,100) → Linear(100,d) → SiLU → LayerNorm`，温度 `sqrt(d)`（7B 上为 64）。该顺序采用源码布局，论文式 (1) 的文字布局把非线性写在两层之间，需区分。

对当前所有 key 的点积 softmax 得到实例级权重，所有目标层共享同一组权重，对各 LoRA 输出作加权和。生成时权重由原 prompt 计算一次，全程保持；不读 reference，不让后续生成 token 改变路由。未来任务只能使用已经学到的模块；stage 0 没有模块，直接测基座。

### 5.2 注意力反思 ARM

阶段 1–6 完成并加载当前 Dev 所选状态后：

1. 在**当前 Train 的 1000 个 prompt** 上计算平均路由分布，保存为该任务的注意力目标。原文 §4.3 使用 test inputs 的平均权重；本项目明确改为 Train，禁止 Test/Dev 流入后续反思训练。
2. 在相同冻结基座上临时创建独立的生成器 LoRA，只训练重建当前 Train 的原始 input。生成器与解决任务的适配器不混用；条件为文本 `[Gen]`，不扩充 tokenizer/vocab。重建目标右截断至 255 tokens 加 EOS，1 epoch，其他优化设置相同；固定最后 epoch 保存，不用测试集或额外 Dev 选模。
3. 生成 128 条非空伪 input：top_p=0.9、temperature=1、max_new_tokens=256，生成器随机种子 `42+10000+阶段号`；在独立 RNG 上下文运行，结束后恢复主线 Torch RNG。允许重复，并记录唯一条数；最多尝试 512 条仍不够则报错，不静默用真实样本补齐。
4. 将伪 input 放回该任务原有 instruction 模板，缓存 prompt token IDs 和平均注意力目标。目标顺序按任务加入顺序排列；后续新增任务在目标末尾补零。
5. 后续每个主训练更新，对每个旧任务取 16 条伪 prompt，按 `(update×16+offset) mod pool_size` 循环取样。计算 `KL(保存的目标 || 当前路由)`（PyTorch `kl_div(log_current, target)`，与归档实现一致），对旧任务求和、系数 1。该 KL 只更新共享投影和当前 key，不用旧任务答案，也不做语言模型 replay CE。

KL 和当前任务 CE 一起按 micro-batch 权重累积，避免 16 次累积把正则项放大 16 倍。每个主任务仍然只有 63 个更新。第七阶段没有后续任务，不再训练生成器。

生成器是 SAPT 方法自带的辅助训练，不是 FWT 的单任务对照，不产生单任务评价成绩。它额外增加六次 1000 样本×1 epoch 的 input 重建（378 个更新）及 768 条伪样本生成。各阶段生成器检查点、伪文本、数量、耗时单独保存在 `continual/reflection/stage_XX/`。推理时不加载生成器。

### 5.3 必须披露的适配

本项目统一 rank=8、dropout=0.1、lr=1e-4、每任务 1 epoch；归档 SuperNI/Llama 配置为 rank=4、dropout=0、lr=5e-5、50 epochs。公共数据划分与原文也不同。生成器采用固定 1 epoch/128 伪样本的小预算，按当前已学 Train 现生成；不导入上游生成文件，避免未知训练来源与本地测试集重叠。路由目标由 Train 计算是主动避免测试泄漏的适配。结果应标为“SAPT-LoRA，统一协议适配”，不直接等同原论文数值。

## 6. 检查点、资源与结果

### SeqLoRA 基础对照

第四张 GPU 使用 SeqLoRA：标准 LoRA（Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, https://arxiv.org/abs/2106.09685）在七个任务上连续训练，是 SAPT/MIGU 等论文中的顺序学习基础对照，不宣称是另一篇新持续学习算法。

只在第一任务创建一组 q/v LoRA，A 用 Kaiming 初始化、B 为零；后续任务继续更新同一组参数，保持 4,194,304 个适配器参数。任务切换按公共协议重置优化器/调度器，但不重置 LoRA 权重。没有梯度遮罩、正交损失、回放或任务路由。基座始终冻结。

rank、alpha、dropout、学习率、batch、epoch、数据、解码和评价口径均与另外三个方法保持一致。它特别适合与相同固定容量的 MIGU-LoRA 比较，判断梯度遮罩是否带来提升；也可作为 O-LoRA/SAPT 增加容量或辅助机制后的收益参照。所有方法仍各自只有一条主线；新增对照不是逐任务独立训练。

继续优先 LoRA 方法，因此第四个选择 SeqLoRA；Progressive Prompts 属于软提示方法，尚未启动。

### 保存与汇总

每次运行保存 config、model/code identity、split manifest、环境、状态及全部预测。主任务检查点在 `continual/checkpoints/`。MIGU 保存固定 LoRA；SAPT 保存 LoRA、router 和此前生成的 memory，反思完成后的 `stage_state.pt` 包含刚加入的 memory。每个完整优化器更新后原子保存 `recovery.pt`，含方法权重、AdamW、scheduler、AMP scaler、随机状态、epoch 和样本游标。阶段完成后保存 `completed.pt`，恢复时跳过完成阶段。

`continual/resources.json` 记录主训练更新、适配器/路由参数、辅助训练资源和峰值显存；反思阶段单独的 `resources.json` 在任务完成前就会落盘。阶段 7 时 O-LoRA/SAPT 的 LoRA 总参数约 2936 万，MIGU 约 419 万；SAPT 另有约 85.6 万路由参数，当前阶段可训练量约 502.6 万。参数量不同是方法差异，不应以相同 rank 推断相同容量。

真实评价必须等完整七阶段完成。功能检查、少量真实 7B 更新和临时显存检查均不写入正式成绩；CPU smoke 使用每个 split 前两条样本，只验证代码流程。

## 7. 本次验证

- 两种新方法均通过 CPU 七阶段完整流程（基座初测、训练、Dev 选模、全任务测试、汇总）；SAPT 包含六次生成器训练、伪输入生成和 ARM。
- SAPT 另通过 NF4/FP16 的七阶段小模型检查（warmup=0，实际执行参数更新），覆盖生成器和采样分支。扩展后的公共引擎运行 O-LoRA 小模型，8×7 分数矩阵与修改前完全一致。
- 检查 MIGU 不增长适配器、遮罩后梯度确实置零、基座无梯度；检查 SAPT 历史模块冻结、路由只依赖 prompt、KL 有效且不更新 LoRA、checkpoint 恢复包含 memory。
- 真实 Llama-2-7B/NF4 在七阶段容量下各完成两个训练更新，使用真实 XSUM Train 中长度 846/834 的样本。MIGU 峰值 allocated 4.39 GiB / reserved 5.24 GiB；SAPT 4.69 / 5.50 GiB。该检查验证此形状的训练可行，不代表覆盖所有生成长度，也不构成论文结果。

全部正式比较条件在启动前固定。后续若因错误必须改算法、数据或预算，应新建 run_id 并记录原因，不混合不同实现的矩阵行。

## 后台运行与断点恢复

四种 `run_*.sh` 均默认后台启动。例如：

```bash
bash exp/run_olora.sh --config exp/configs/olora_first.json --model model/Llama-2-7b-hf --gpus 0
# RUN_DIRECTORY 为包含 config.json、status.json、continual/ 的原运行目录
bash exp/run_olora.sh --config exp/configs/olora_first.json --model model/Llama-2-7b-hf --gpus 0 --resume RUN_DIRECTORY
```

也可使用 `bash exp/start_detached.sh METHOD GPU --model model/Llama-2-7b-hf`。
启动返回的日志记录准备过程和 run directory；训练详细输出在 `RUN_DIRECTORY/logs/continual.log`，当前进度在 `continual/status.json`。后台启动成功不代表模型准备完成，应检查日志和状态文件。关闭终端/SSH 不会停止实验；断电或系统重启后需要手动执行恢复命令。不要删除 checkpoint；只恢复可信的本地文件。

| 保存内容 | 运行目录下的位置 |
|---|---|
| 参数、数据划分、模型/代码指纹、环境 | `config.json`、`split_manifest.json`、`model_identity.json`、`code_identity.json`、`continual/provenance.json`、`continual/environment.txt` |
| 每阶段各任务 ROUGE-L、Exact Match、Token F1、样本数和资源 | `continual/stages/stage_XX.json`、`continual/scores/stage_XX/` |
| 每条输入、实际 prompt token、回答原文、生成 token、参考答案及各指标 | `continual/predictions/stage_XX/*.jsonl`；未结束为 `.jsonl.tmp` |
| 每步 loss、学习率、梯度范数、AMP scale、样本 ID | `continual/checkpoints/stage_XX/training.jsonl` |
| 训练恢复、Dev 选中权重、完整阶段状态 | `continual/checkpoints/stage_XX/{recovery,best,completed}.pt` |
| Dev 逐条回答和指标 | `continual/checkpoints/stage_XX/dev_epoch_*.jsonl`、同名 `.json` |
| SAPT 辅助训练、伪输入、memory | `continual/reflection/stage_XX/` |
| 分数矩阵和最终持续学习指标 | `continual/score_matrix.json`、`metrics.json` |

这些任务主要是文本生成，主指标为 ROUGE-L；Exact Match 是归一化完全匹配准确率，三种指标均按 0–100 记录，不混称为分类准确率。

训练在完整更新边界恢复，尚未保存的更新重新执行；评价逐 batch 刷盘，校验缓存样本 ID/参考答案/阶段后复用，残缺尾 batch 重新生成。SAPT 生成器也保存优化器状态；中断的伪文本采样从生成器恢复后重新执行。历史版本未保存的 token IDs 等信息无法补回，不能声称旧记录包含这些字段。恢复事件保留在 `resume_events/`；开始训练后代码指纹改变会拒绝复用，避免混合实现。

验证命令：`.conda-env/bin/python exp/tests/test_recovery.py`。覆盖四种方法的真实微型 Llama 更新、中断、恢复和逐参数/RNG 一致性；不构成正式实验成绩。

## 医学生成任务续训

在完成 SuperNI 七任务后，使用 `bash exp/start_medical_detached.sh METHOD GPU` 启动 MTS-Dialog → IU X-ray 两任务扩展。配置是 `exp/configs/<method>_t5large_medical2.json`，各方法加载自己原实验的第七阶段 checkpoint，新结果独立存放，旧运行不修改。

恢复使用 `bash exp/start_medical_detached.sh METHOD GPU --resume RUN_DIRECTORY`。

运行 `python summary/collect_medical.py` 更新医学汇总，`--watch` 每60秒更新直到四运行结束。医学指标和原七任务 AP/FWT/BWT 分开报告，详见 `rules/003_medical_continuation.md`。
