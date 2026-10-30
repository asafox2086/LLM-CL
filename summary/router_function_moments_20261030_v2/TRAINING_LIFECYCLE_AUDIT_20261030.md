# 训练生命周期核查：尚不能解释终态条件变化差异

仅审冻结源码及既有S19 pickle-layout元数据；不加载模型/权重、不执行pickle、不重消费者，不改当前四臂或PAS结果。

**结论：** 否决“router没进optimizer／被固定冻结／每stage重建相同专家”作为这两份实现的解释。已有终态gate观测不等于router学习轨迹；没有router专属梯度或阶段前后参数值，不能声称PAS被正则压平、没有学动或某个生命周期步骤造成当前差异。

## 可核事实

- **optimizer与保存状态。** 冻结`runtime_utils.py:213–224`收全部`requires_grad`参数；`run.py:235–245`在`begin_stage`后为每阶段新建optimizer。新建optimizer不等于重置模型权重。既有S19 layout中，CL的168个router、1344个A及1344个B（按每expert的`.mlp.weight`计，避免`.weight`别名重复）全部标True；PAS的336个A及336个B全部标True，独立router参数为0。这证明保存的训练资格与源码选入规则相符，不证明每个参数实际梯度非零或更新有效。
- **PAS是固定两专家连续继承。** `pass_adapter.py:60–78`只在模型构造时创建两套r128；作者`coinmoelora.py:388–413`初始化A、B。每task的`pass_adapter.py:83–86`及作者`moelora.py:329–339`只重置当前统计/计数和tau，不重建或冻结A/B。作者`moelora.py:361–390`更新历史`_A_ref/_B_ref`，并不把参考均值写回模型，不是合并当前两个专家。实际a路由使用每个A的全部128行投影平方均值（`coinmoelora.py:674–682,719–722`）；`gate_rank=16`不用于此a分支。B不直接定义router，不可用整bank参数量代表选择器大小。
- **CL是固定八专家与独立router连续继承。** `clmoe_adapter.py:17–28,50–62`构造8×r8及独立8路linear-router；`72–77`只保存阶段前A/B副本、清global prior，不重新初始化router/专家。`123–132`在stage末将A/B与阶段前副本按0.3/0.7旧权重作momentum，router不参与该混合。
- **CL的top2不做train/eval专家裁剪。** `clmoe_adapter.py:125–132`的top2只决定stage末哪些首次入选专家采用0.3而非0.7旧权重，并更新seen集合。原`forward:33–46`在train/eval均计算全部8个专家；训练g为softmax(router(x))，不存在top2梯度/前向mask。eval且`global_weight`存在时，g=0.5×softmax(router(x))+0.5×global_weight；否则仍为原softmax。`prepare_row:100–104`通过最近question-anchor选一份历史task prior，不是选top2专家。
- **实际eval混合是归一化概率。** `end_stage:112–126`先清global_weight，再以原softmax的逐样本token均值收集task_weights；它是概率向量的均值。故eval的两项各sum=1，0.5/0.5混合理想算术下仍sum=1，浮点只有舍入偏差；代码没有额外重归一化，也不能把两项写成相加后sum=2。stage末收集时虽为eval，global_weight为None，所以不混旧prior。既有原CP layout保存19个anchor及3192个layer-task-weight张量；原`auxiliary/clmoe_stage_19.json`保存49TRAIN/top2记录。这里只核口径，不再消费gate矩或新算归一化表。

## 证据与完整SHA

以下冻结native文件位于`exp/residual_exit_native_sources_20261030/`，与原config映射逐字节一致（本次实际映射25文件，不用旧交接29这个数量替代）：

| 源文件 | SHA256 |
| --- | --- |
| runtime_utils.py | 4f5c4a34be046c29cfbcdaa4b2957faa907684872ba69083bdef53136a851240 |
| run.py | 0e1f794ff1daf0694bdf960c3f89ccd4078764577db75271d8f78684282a39a9 |
| pass_adapter.py | c8e23c85a93e97130d464b4b6d87110470035c5af37f9575fce705923c80c7c4 |
| clmoe_adapter.py | 33946eddfc6424638ded0675ded68860273859c2f71993a6a60bd22556341be9 |

作者源`code/PASs_MoE/source/CoIN/peft/tuners/coinmoelora.py`：75d46bcd7850ec3c03d13968f1e4180098d62a1da7ab905df6f9fa1d054a4b6a；`code/PASs_MoE/source/cl_learner/moelora.py`：95ff81926866e3a94b041a83fd67dfa0ec27087f58dba590ba3c8d6a50e27728。

既有`summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/EVIDENCE.json`：9bcf7199541b4e4be8d5af8c621bf1c1b524fa9542484dbc8446fb9ede739797；其绑定的既有Windows小metadata记录如下（不是本次新解析checkpoint）：

| 方法 | `.deployment/checkpoint_weight_storage_20261024/`记录 | 记录SHA256 | 原S19 checkpoint SHA256 |
| --- | --- | --- | --- |
| CL | onevision_CL_MoE.json | 123888acc5bac875c6c119ce7f8e65e305fd14eb702d8efd86256c49259271b7 | 8550fdeb886a0e942ee9038ab0aeae7e5933b943094ee3af40c0816df592ffa4 |
| PAS | onevision_PASs_MoE.json | d463c7c141394813700ae7a8408be47025b90e72f794dbdf81eb07477b3d358c | 583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a |

元数据明确`weights_not_loaded=true`、`pickle_globals_not_executed=true`。目前可陈述的是两种选择器结构和阶段接口不同；没有证据把终态PAS较小方差归因某个训练过程。不据此新增方法、机制实验或调参分支。
