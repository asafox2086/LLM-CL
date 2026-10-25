# original 与朴素 r8：共同学习包的可识别边界

本记录只用已完成 ARC 结果、封存训练计数及实际 `worker.py`／`loss_core.py` 数学式，不读取正在运行的 acquired 诊断。它不授权新模型、不改冻结条件，也不是新的科研结果。目标仍是让完整 **893,675,552** 系数共享图保有足够总体问答质量；答案门的微小差异不能代替这个目标。

## 两者实际比较了什么

两者具有相同 q/v-r8 的 540,672 个训练坐标、原 current FIT 顺序、87 次 optimizer 更新和原阶段学习率；最终都 GPU FP32 fold 为同一个尺寸的 FP16 稠密图。original 的训练 GPU 为2，naive 为0，推理读数统一 GPU1。因此这不是只有一个学习项不同的严格对照，也不是相同训练计算预算。

S1 无历史，两者都是 `CE_current`。S2–19 的 actual loss 为：

- naive：`CE_current`。
- original：`(CE_current + CE_replay)/2 + MSE_whole_hidden`。

各 CE 是各自 QA 的答案 token 平均值，组合 loss 再按原 current minibatch 大小累积；不是把 current 与 replay token 拼接后作 token 等权。故 original 同时将后886次 current QA 的直接 CE 系数减半、加入886次有梯度历史 CE，并加入整包隐藏约束。**当前项系数减半不等于真实参数步长减半**：历史梯度、全局 clip、Adam 矩及其阶段 reset 共同决定实际更新。不能用87个更新相同，声称有效监督或计算机会相同。

每臂 original 有1,213 current＋886 replay working decoder 请求，另886次 no-grad slow decoder，共2,985次训练 decoder；naive 是1,213次。original 还多886次实际 max-target 的答案 head 读出，保留25份 working／slow hidden 用于整包 MSE。teacher 不执行废弃答案 head。共1,213次组合 backward 调用，不能说只有1,213次有梯度 QA；0新增 gate 参数也不等于0训练费用。固定87个 optimizer 更新约束了更新次数，没有消除额外历史监督和计算的收益来源。

## 记忆与约束也不是独立因素

original 的 reservoir500 在 task 结束后插入，每次 current QA 从过去 task 有放回取1题；同一 QA 可反复出现。当前与回放域曝光如下，均为原封存记录的描述量：

| 域 | current FIT QA | current 更新数 | 历史 replay 曝光 | replay 独立 QA |
| --- | ---: | ---: | ---: | ---: |
| natural | 327 | 21 | 469 | 237 |
| medical | 79 | 11 | 113 | 58 |
| geography | 553 | 37 | 284 | 177 |
| finance | 69 | 5 | 12 | 11 |
| traffic | 185 | 13 | 8 | 6 |

这不是域均衡 replay，也不是所有原 QA 等次数学习；后到域可被后来回放的机会更少。曝光计数本身不能解释遥感收益、医学损失或从中推导最佳分配。

slow 只有540,672个冻结 FP32 A/B，额外2,162,688 bytes。每个 stage 开始 `slow=working`；每次实际更新后才 `slow=0.25*slow+0.75*working`。它不是保持旧 task 端点不变的跨域教师。单更新 stage 的更新后 EMA 不再用于该 stage 的下一次约束，下一 stage 又先同步 working；不能把这些 stage 的结果归为长程 EMA 记忆。

MSE 的目标是每层 `max(H_work.detach(), H_slow.detach())`，包含 embedding 和24层的全部有效 packed 位置，25项等权。它不是 slow 的答案分布，也不是只约束答案 token。working 与 slow 分别处于训练／dropout 模式；即使 stage 起点参数相同，独立 dropout 也可能造成隐藏差别。因此 MSE 同时涉及历史监督、隐藏特征/随机扰动约束与当前学习轨迹；“保持旧答案”不是源码自动保证的功能。记录到的原 MSE 数值约为 `4.52e-4`，不能据其小于 CE 就推断梯度影响小。

## 已有结果允许与不允许的解释

original 宏43.703262 对 naive38.775370 的约+4.927893pp，主要伴随遥感+35pp、医学−10pp，其余自然/金融0、交通−0.360537pp。这是共同 ER／EMA／MSE／current 权重与执行机会整包的结果，不能单归 MSE、EMA、回放或“更多有效习得”。新 gate 比 original 仅+0.016628pp，四域逐题相同，不能承接这约+4.93pp作为新增门贡献。

正在进行的读数能定位这两个域的差异在各 QA 所属 task 刚学完时是否已经出现；它不消除上述混杂。acquired 按不同 task 端点聚合，final−acquired 允许有正迁移或负迁移，不是纯遗忘量；也不识别具体哪次后续更新或哪个 loss 项造成变化。

## 结果之后才可能需要的最小控制

以下是互斥的条件性优先建议，每种情形最多一项；不是两个待运行试验，不要求诊断给出预设方向，也不授权追 LR、rank、memory、步数或门尺度。

**若主要差异在刚习得时已经形成：优先考虑一个“同当前系数、同调用预算，但历史目标不给梯度”的控制。** 保留 original 的 current/replay IDs、所有 working／slow／target-head 调用、EMA/dropout 调用顺序和原更新预算；S1 仍用原 current CE，后续 loss 只保留 `0.5*CE_current`，历史 CE／MSE 以零权重走原图，不新增或重分配训练样本。与 original 比较时，current 直接系数及计算调用已对齐，因而可检验历史损失的功能梯度是否有额外价值，而非仅凭多 forward 或 current 系数减半解释。应实际记计算/梯度执行费用，不能仅称“同算力”。它仍将历史 CE 与 MSE／EMA整包视为干预，不能进一步分离两者；与历史 naive 的差也不能自动叫纯系数因果。若不能优于此控制，不支持将共同包收益解释为有用的历史目标梯度；小点差不证明等效。

**若主要差异在后续更新才形成：优先考虑一个同 replay 的 CE-only 控制。** 保留同 `0.5*CE_current+0.5*CE_replay`、相同历史曝光、更新数及 slow／head 诊断计算，但将整包 MSE 权重固定为0，不更换回放策略。它与 original 的差检验“在已有等量历史标签回放之外，隐藏巩固是否必要”，比直接对 naive 更能区分普通监督 replay 与额外巩固。应同时读自身 acquired 和 final，才能避免把更好的起点误说成更强保持；结果仍只识别给定 MSE/EMA目标包，不单独认证 EMA、max 操作或纯遗忘机制。若 CE-only 已承接后续优势，则没有证据必须保留隐藏巩固模块。

若诊断是混合结果，应按域和实际变化描述，先确定哪一种功能缺口值得检验，不将两项控制串成自动队列。任何后续方法还须证明总体质量及完整共享图的实际部署竞争力，不能把一次小样本门正向或已有图的内存优势写成新算法已成功。

来源：[ARC正式协议](../../answer_reliability_consolidation_20261025/v1/PROTOCOL.md)、[完整结果](../../answer_reliability_consolidation_20261025/v1/OUTCOME.json)、[结果审查](../../answer_reliability_consolidation_20261025/v1/RESULT_REVIEW.json)；数学式核对 `exp/answer_reliability_consolidation_20261025/v1/{worker,loss_core,adaptation}.py`，未改这些源。
