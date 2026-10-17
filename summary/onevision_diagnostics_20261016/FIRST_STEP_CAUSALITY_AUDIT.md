# 首步保护项为零，不等于首步正在遗忘

这是K01的必要反例边界检查。既有旧TRAIN梯度、优化器范数和实际阶段位移分析的生产脚本／源文件SHA重新核验，无GPU或新训练。

| 序列 | 控制 | 新／旧梯度cos | 理想fresh Adam首步旧损失一阶项 | 实际4步位移旧损失一阶项 |
| --- | --- | ---: | ---: | ---: |
| repeat_anchor | anchor_gradient_noop | +0.7992 | -0.9300 | -3.2167 |
| repeat_anchor | anchor_gradient_projection | +0.7550 | -0.9074 | -3.1981 |
| within_medical | anchor_gradient_noop | +0.4986 | -0.5779 | -2.1747 |
| within_medical | anchor_gradient_projection | +0.4945 | -0.5429 | -2.0036 |
| cross_short | anchor_gradient_noop | +0.1898 | -0.1776 | -0.7986 |
| cross_short | anchor_gradient_projection | +0.1594 | -0.1372 | -0.7846 |
| cross_five_domains | anchor_gradient_noop | +0.1642 | -0.1318 | -0.6431 |
| cross_five_domains | anchor_gradient_projection | +0.1855 | -0.1645 | -0.8429 |

8/8新旧TRAIN梯度cos为正；8/8理想Adam首步和实际四步位移的一阶项为负，局部近似均指向旧TRAIN损失下降。因此这些初始转移不能作为“EWC首步项为零导致遗忘”的正证据，反而要求保留可能的正迁移。

这不是实测有限步旧NLL下降，也不是生成准确率证明；不能把noop／projection的梯度代入其他EWC条件。尚未测得的因果不被补全。K01的数学启用条件继续成立，但“为何最终掉分”必须另由匹配的权重／损失干预验证。

原研究中的较后阶段可出现梯度冲突，但原默认后端、内容／token预算和单锚限制仍在，不能直接据此建立领域数的唯一因果解释。完整记录见[JSON](FIRST_STEP_CAUSALITY_AUDIT.json)。
