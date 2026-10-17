# TRAIN梯度冲突发生在首步，还是同阶段后续更新？

只读已完成旧8条noop/projection轨迹；128个后续阶段更新日志与32个保存首批原始向量，SHA核验。不是新GPU实验，也不是确定性双锚实验的替代。

共31次负点积，其中首批5次、阶段内部后续批次26次。进入stage2的8个首批均非负。多数负点积不在保存完整梯度的阶段首批，不能拿首批性质概括整阶段。

| 序列 | 原机制 | 首批负点积/4 | 后续负点积/12 | 真正投影次数 | 锚Answer F1 S1→S5 |
| --- | --- | ---: | ---: | ---: | ---: |
| repeat_anchor | anchor_gradient_noop | 0/4 | 0/12 | 0 | 16.232→36.957 |
| repeat_anchor | anchor_gradient_projection | 0/4 | 0/12 | 0 | 16.232→36.957 |
| within_medical | anchor_gradient_noop | 0/4 | 2/12 | 0 | 16.232→44.203 |
| within_medical | anchor_gradient_projection | 0/4 | 2/12 | 2 | 18.716→42.029 |
| cross_short | anchor_gradient_noop | 1/4 | 5/12 | 0 | 18.716→19.130 |
| cross_short | anchor_gradient_projection | 2/4 | 6/12 | 8 | 16.232→21.739 |
| cross_five_domains | anchor_gradient_noop | 1/4 | 5/12 | 0 | 16.232→17.681 |
| cross_five_domains | anchor_gradient_projection | 1/4 | 6/12 | 7 | 16.232→13.712 |

## 证明与边界

重复原锚的两轨迹没有记录到负点积，同医学新任务有少量，跨域轨迹更多；这是实际固定序列的描述性线索。它支持进一步检验阶段内部的局部方向冲突，不能证明跨域数量本身是原因。

32个首批的原始向量在CPU重算点积全部通过；其余96更新仅有原运行当时记录的点积/范数，不声称保留了全部原始梯度。旧EWC最新中心首步梯度0的性质继续成立，但不能把它直接解释为所有后续冲突或性能缺陷的原因。

质量列统一是原始Answer F1；原首阶段权重差异尚在。不能把投影与noop的终点差异称为投影收益，也不能将负点积解释成有限Adam更新后必然掉分。核心双锚匹配对照仍需完成。

- Original default-backend stage1 parameters differ; no method superiority or strict domain-count causality.
- Only first minibatch full gradients retained; later signs are actual recorded FP64 dot logs, not recomputed complete vectors.
- Reference is mean gradient of8 TRAIN anchor QA at each current weight, eval-mode; current TRAIN gradient uses train-mode/dropout.
- Raw-gradient opposition is a local direction property, not measured old TEST loss/quality change and not an Adam update guarantee.
- Updates/conditions are dependent; conflict frequencies are descriptive, not independent Bernoulli confidence tests.
- Different conflict counts on projection/noop are different own trajectories; initial numerical differences forbid causal attribution.
- Matched two-anchor48-case GPU study still pending; no new method or changed GPU training.
