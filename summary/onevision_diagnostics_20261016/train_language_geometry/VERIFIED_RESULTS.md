# 平均旧功能改善会否掩盖子组受损：TRAIN-only实测

来源为既有确定性19阶段轨迹，4条冻结医学TRAIN探针（英文2/中文2），无新GPU运行、不读TEST、不改变评分。与严格48组、正式V2权重不同。

| 阶段/当前域 | 四题全部已学 | 平均实际ΔNLL | 英文实际ΔNLL | 中文实际ΔNLL | 平均改善但有组变差 |
|---|---|---:|---:|---:|---|
| T02/medical | False | -0.371926 | -0.251658 | -0.492195 | False |
| T03/medical | False | -0.360787 | -0.283379 | -0.438195 | False |
| T04/medical | False | -0.184633 | -0.225959 | -0.143307 | False |
| T05/medical | False | -0.063086 | -0.131638 | +0.005467 | True |
| T06/medical | False | +0.025692 | -0.044343 | +0.095728 | False |
| T07/medical | False | -0.168006 | +0.001606 | -0.337618 | True |
| T08/medical | False | -0.075899 | -0.169265 | +0.017466 | True |
| T09/medical | True | +0.003167 | -0.036950 | +0.043285 | False |
| T10/medical | True | -0.016767 | +0.001562 | -0.035097 | True |
| T11/medical | True | -0.027887 | -0.023486 | -0.032287 | False |
| T12/geography | True | -0.013535 | +0.023509 | -0.050580 | True |
| T13/geography | True | +0.120431 | +0.338867 | -0.098005 | False |
| T14/geography | True | -0.067930 | -0.175355 | +0.039496 | True |
| T15/geography | True | -0.002970 | +0.194803 | -0.200743 | True |
| T16/finance | True | +0.054087 | +0.176711 | -0.068537 | False |
| T17/traffic | True | -0.005374 | -0.028031 | +0.017282 | True |
| T18/traffic | True | -0.018749 | -0.125334 | +0.087835 | True |
| T19/traffic | True | -0.024389 | -0.027489 | -0.021289 | False |

ΔNLL<0为这些固定TRAIN探针改善，>0为变差，不是Answer F1。原始梯度、实际参数位移Taylor项和有限NLL逐项分列于JSON。

## 六栏判断

- **探究的问题：** 旧任务平均loss或平均梯度方向是否代表所有旧子功能？
- **做的实验：** 校验原完整证明、19个状态/梯度、4个原TRAIN ID与语言，全部18段都报告。只有四题所属任务都已学习后，才计旧功能受损；不把尚未学过的探针称遗忘。
- **结果：** 四题均已学的11段中，平均NLL改善但至少一语言组变差的阶段为[10, 12, 14, 15, 17, 18]。有英文受损，也有中文受损，不是中文专属现象。
- **反方向证据：** 整个T11→T19后续跨域段，这2条英文TRAIN NLL总变+0.377682，2条中文-0.294541；由首尾NLL与各段相加精确核验。该轨迹中文TRAIN总体改善，不能据此前K18另一轨迹的中文测试低分宣称所有跨域都伤中文。
- **证明了什么：** 在本固定探针上，平均旧loss改善不保证每个子组保留；这是已知平均化性质的本例验证。语言与任务/答案内容混杂，2题一组不足证明语言是原因，不能解释所有持续学习方法或最终F1。Taylor与有限NLL有反向情形，局部线性符号不能替代真实有限变化。
- **改进的方法/效果：** 原因判定需同时保留逐题/子组与整体有限变化；没有新增保护方法或模型收益。GPU1既有受控推理和严格48实验继续等待，不用这个CPU审计代替因果验证。

[完整18段、同权重梯度/参数/原数据SHA](EVIDENCE.json)

- Two fixed TRAIN QA per language, different native tasks; no population language effect, confidence interval or held-out accuracy claim.
- All18 transitions reported; only stages after every probe task is acquired enter old-function masking count.
- Same exact saved parameter points only, never reused as strict48/V2/EWC gradients.
- Taylor is a local prediction; measured NLL before/after is separately stored and may have a different sign.
- Opposite subgroup movements do not identify language as cause; question/task/answer content remains confounded.
- These are eval-mode TRAIN probe gradients, not historical optimizer gradients or proof projection would fix accuracy.
- No randomized method/weight intervention; not a new mathematical discovery, algorithm improvement, or domain-count causal law.
- Cross-segment Chinese TRAIN NLL improves overall here; cannot generalize a Chinese-only cross-domain degradation story.
