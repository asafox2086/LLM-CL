# 答案读出代理约束的记忆巩固：候选方案，尚未实验

研究问题是：在1%多模态开放问答、许多阶段只有1–2次更新的条件下，对回放样本无条件施加历史隐藏态约束，是否会阻碍当前有效学习？已有Seq正BWT、OSFT优势更集中于域末习得，以及部分域后续正迁移，支持检验“何时施加约束”，不支持先把遗忘认定为总根因。这些观测没有证明I-LoRA约束有害；以下对照才检验它。

借用[I-LoRA论文](https://arxiv.org/abs/2402.18865)的共享基座、working/slow适配器、回放与EMA。官方[固定源码](https://github.com/which47/LLMCL/blob/a9dae1230ca63f5ec347ae16496e69157a2c86be/method/ILORA.py)在回放QA上，用working与slow各层隐藏态的逐坐标最大值作为working隐藏态MSE目标。论文已有文本生成任务，不能将开放生成本身称为新差异。

唯一新增规则是无学习参数的QA级开关。记原目标为 `T_l=max(H_work,l,H_slow,l)`；`D_i`为原整包隐藏态MSE。只在最后一个、已经经过原final norm的隐藏态上，使用同一冻结LM head和原答案预测位置计算：

`q_i = mean_answer_token_CE(head(T_last), gold)`

`p_i = mean_answer_token_CE(existing_working_logits, gold)`

`g_i = stop_gradient(1[q_i < p_i])`。

将原回放约束 `mean_i D_i` 替换为 `mean_i(g_i D_i)`；分母仍为全部回放QA数，不按接受数重归一化。CE、回放抽样、EMA、原目标、学习率和更新数不变；无温度、阈值或门率搜索。working NLL复用已有logits，新增工作只有末层目标的冻结head投影和CE归约，没有额外完整decoder。不得给中间层加head或新norm。

这修正了未实验的“slow NLL优于working就准入”草案：更好的slow回答不保证坐标max目标更好。即便新门打开，也只说明该次随机forward下的末层人工目标有较低规范答案NLL；它不认证整包中间态、梯度方向、可实现轨迹或自由生成F1。门控整包MSE是待验证的末层读出代理假说，不是安全保证。

比较直接移植I-LoRA、target-gate和接受数匹配的reference-mask shuffle；既有独立Naive-r8控制作为简单锚，不重复训练。gate与直接移植的差异检验准入整包的净效果；gate与shuffle的差异检验样本对应关系，而非仅弱化正则。Naive与任一I-LoRA分支的差异还包含回放/EMA，不能全归新门。

固定OneVision q/v-r8：working 540,672个FP32训练坐标；冻结slow另540,672个坐标，约2.063MiB，不进入优化器。新增门参数为0。回放、两次decoder、隐藏态与新增head计算另计。最终只导出working并fold，设计完整图仍893,675,552个系数；不保留slow、回放或门。它相对同样fold的普通LoRA/I-LoRA没有新推理减重，论文端侧优势必须来自相对保留专家银行等强参照的真实完整图与同卡测量，不能由“训练参数少”推出。

实施时共同解决官方源的EMA名字映射/阶段同步函数名不一致，并固定dropout与memory语义，不能将源缺陷当原法阴性。官方no_grad不等于eval；三臂复用同模式、同次forward。定向检索已有[选择性蒸馏及随机选择对照](https://aclanthology.org/2021.acl-long.504/)，不能称首次提出选择性蒸馏或损失准入；贡献只能由本场景适配、必要对照、完整回答和成本证据建立。

若gate未优于直接移植或同接受数shuffle，拒绝本固定准入规则的新增效用，不按DEV追阈值、rank、LR或memory。若只有总体收益而域间取舍，则如实报告；不要求每域最强，也不由方法有效倒推全部机制成立。
