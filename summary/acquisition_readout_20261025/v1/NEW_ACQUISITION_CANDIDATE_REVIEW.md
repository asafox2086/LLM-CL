# 新习得候选审阅：等函数LoRA坐标依赖不构成新命题

本轮只读历史小证据与一次LoRA-RITE原论文；0模型、0评分、0服务器、0张量重算。结论：**不立新实验，不重跑尺度控制，不为补旧质量而恢复RITE队列。**

## 已经回答的部分

- [optimizer_geometry原结论](../../onevision_diagnostics_20261016/optimizer_geometry_20261019/v1/RESEARCH_FINDINGS.md)与[原EVIDENCE](../../onevision_diagnostics_20261016/optimizer_geometry_20261019/v1/EVIDENCE.json)已用三个任务的真实TRAIN梯度，保持同一BA，将A乘4、B除4及反向缩放。AdamW实际deltaW范数约0.103变为0.411或0.033；作者RITE在这些缩放下deltaW逐值相同。AGENTS的K52亦已记录。这是已知坐标依赖在本权重上的局部验证，不能再次称新发现。
- [旧连续RITE协议](../../onevision_diagnostics_20261016/lora_rite_continuous_20261019/v1/PROTOCOL.md)已提出同r8、不扩专家、跨阶段保留A/B、替换优化规则的完整试验。现有本地入口没有对应完整GPU质量证明；既不能写成RITE质量阴性，也不能据缺测自动恢复旧分支。其原1313TRAIN／93更新／BF16等条件亦不同于近期1213FIT／87更新／FP16探索。
- [LoRA-RITE原论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/bcbc0f660d2dde42f9d1d0ecb14a6f9a-Paper-Conference.pdf)§2已直接研究等价因子缩放／旋转下的更新差异；§3用低秩侧矩阵预条件并处理动量的基变化。借用它可以是不扩推理图的已知工具，但不能将这个一般目标或预条件本身称为本项目创新。

## 为什么还不能解释新习得不足

上述K52是固定梯度的一步CPU反事实，没有新前向、有限学习轨迹或新习得F1。不同坐标产生不同更新，不等于默认坐标产生较差回答；不变优化器也可能稳定地产生无益更新。当前低分或短阶段没有把这一坐标依赖与实际习得损失连接起来，不能靠梯度／更新范数大小补上这个因果缺口。

直接再比较Adam与RITE还同时改变预条件、裁剪、epsilon、跳过规则和更新幅度；旧协议已经公开这些边界。即使出现F1收益，也只能先支持所用优化包的效用，不能唯一归因等函数不变性，更不能靠它增加完整图减重或GPU优势。

## 与旧函数补偿的区别及本轮出口

[function-neutral结果](../../onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md)比较完整继承与减去固定源BA，后者恢复冷初始输出而改变后续补偿函数族；它检验“首步A梯度非零是否充分”，结果不支持该充分性。它不是在同一BA内换可逆坐标，因此不能否定所有几何优化。反过来，这一区别也不使已做K52成为新问题。

本轮没有找到旧规则中已经具备功能证据、足以立即推进的另一项非重复问题。后续若提出新的干预，应先明确它新增识别哪项新习得功能因果；不能以“旧RITE尚无质量”或“GPU在忙”为由包装同一命题、轮换另一优化器或新模块。当前停止扩展文献和实施，不改旧科学源、协议、队列或关闭判决。
