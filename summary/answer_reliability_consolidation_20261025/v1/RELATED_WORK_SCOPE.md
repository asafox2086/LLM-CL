# 答案可靠性准入：定向先例与贡献边界

本记录仅做原文核查，不消费ARC的partial NLL、mask、生成或质量，不修改冻结方法、协议或运行。核查日期2026-10-25；以下只保留最相关的三篇，不能由这一有限范围断言不存在其他直接先例。

**结论：按教师相对学生的真实答案表现选择蒸馏、按两者CE比值选样，以及随机同数对照，均有明确先例。** 当前可检验的适配区别是：对I-LoRA实际坐标max隐藏态目标的末层，用原冻结head读出规范答案NLL，作为整包隐藏态MSE的QA级代理开关；它不是slow教师原输出的NLL门。这个区别尚不等于原创性或有效性证明。

## 1. 最直接的样本级相对表现开关：LSSD，EMNLP 2022

[Unifying the Convergences in Multilingual Neural Machine Translation](https://aclanthology.org/2022.emnlp-main.458/)（Huang et al.；[原文PDF](https://aclanthology.org/2022.emnlp-main.458.pdf)）的§3.3、Eq.(5)–(7)，印刷页6825–6826，明确提供WHOLE、SELECTIVE及ADAPTIVE。SELECTIVE的样本权重为：教师的真实目标得分低于学生则0，否则1；ADAPTIVE使用两者得分比值调节输出分布蒸馏强度。这里作者明确把句子得分实现为**所有目标token概率的算术平均**。因此不能把Eq.(6)直接写成本文的逐token平均gold NLL比较：`mean(p_gold)`与`mean(-log p_gold)`的排序一般不等价；相等时的开关也不同。

这是“教师在该样本上更好才蒸馏”的直接思想先例。可借鉴其将无条件、选择性和自适应传递分开的实验逻辑。原文教师为按语言dev表现保留的历史最佳checkpoint，训练为多语言联合训练，蒸馏目标是该教师的真实词表输出分布；当前是固定r8、past-only回放、阶段EMA，且准入对象是人工max-hidden目标末层读出。已读§3.3与实验表1–3，未确认当前这种逐update接受数匹配的reference-mask重演对照；不据此声称随机控制是新的。表1中SELECTIVE也没有全面胜WHOLE，原论文不支持“选择必优”。

## 2. 真实teacher/student gold CE相对指标：DA-KD，ICML 2025

[DA-KD: Difficulty-Aware Knowledge Distillation for Efficient Large Language Models](https://proceedings.mlr.press/v267/he25c.html)（He et al.；[论文原文](https://openreview.net/pdf?id=NCYBdRCpw1)）§3.2.1 Eq.(1)定义`DDS(x)=CE_student(x)/CE_teacher(x)`，两项都相对于ground truth。§3.2.2将DDS用于每epoch重新排序和高/低难度分层抽样，并逐渐减少数据量；§3.3还改变蒸馏散度。故相对gold loss、动态选样及teacher可靠性直觉都不能作为本文首次提出。

它是独立大小LLM之间的输出分布蒸馏，正文实施为多epoch训练；不是1%少步顺序CL的共享基座EMA回放，也不是人工隐藏态目标的head读出。DDS排序/分层不等于`target_NLL<working_NLL`的零阈值二值MSE开关。文中的随机抽样属于分层选样机制，不能把它误称为已建立的、与当前reference每step接受数精确匹配的null控制。可借鉴“相对损失指标必须有独立效果证据”的论证，不借用其更小学生的结构减重或训练省算结论；当前三臂仍执行同量回放/诊断head。

阅读说明：PMLR出版页和原论文索引中§3.2–3.3已核；PDF直开遇OpenReview验证，PMLR的raw PDF入口出现content-type读取限制，方法核查使用同一原论文的可检索正文。没有以第三方摘要替代公式依据，也未据有限可读部分断言其全部附录不存在某种random ablation。

## 3. 随机同数量及gold置信度不充分的直接警告：ACL 2021

[Selective Knowledge Distillation for Neural Machine Translation](https://aclanthology.org/2021.acl-long.504/)（Wang et al.；[原文PDF](https://aclanthology.org/2021.acl-long.504.pdf)）§4.1–4.2、Table1，印刷页6458–6459，比较等量的高/低分token集合，并明确包含`Distill-Half(Random)`：随机50%词接受蒸馏。最终选择策略主要依靠student Word CE，batch中位数或跨step FIFO近似分布，见§5。

该表还发现teacher的真实标签概率分组在其设定下没有明显区分收益；作者说明不能只凭gold概率否定其他软标签信息。这不是对当前max-hidden代理的反证，但直接限制“较低gold NLL必然代表更有用的全部约束”的推理。当前是QA级、动态接受数量的跨轨迹reference mask，每step保持接受数并记录实际Hamming；它不同于固定50% token抽样。随机同数验证的基本思想已有先例，不能作为新算法贡献。

## 论文中可以与不能说什么

可以把当前点写成**借用I-LoRA与选择性蒸馏，检验答案读出代理能否改善少数据VLM回放巩固**。必须引用LSSD的样本相对表现开关、DA-KD的teacher/student gold CE指标和ACL 2021的等量随机控制。真实场景差异是多模态长packed轨迹、每阶段更新极少、固定r8及单最终图，不能把文本开放生成本身说成空白。

不得声称首次提出可靠教师选择、gold loss门控、选择性蒸馏、无参数mask、随机同数消融或单模型最终部署。`head(max(H_work,H_slow))`与`head(H_slow)`的不同说明原slow-NLL草案不能认证真实目标；并不自动产生新理论。即使门控最终更好，也不能反推所有中间隐藏态目标更正确、每次梯度更有用、视觉语义得到保护或遗忘是总根因。

当前原三臂对照最多识别：该固定代理相对无条件I-LoRA的整包净效用，以及相对**确实改变过QA对应关系**的同数reference-shuffle是否还有样本对应收益。它不区分所有梯度/MSE幅度混杂；reference来自另一已封轨迹，不是shuffle自己的门率。若方法有效，贡献仍需由完整回答、逐域/逐题结果、mask退化/Hamming、约束实际作用与训练开销共同建立，不改既定结果出口追先例差异。

最终fold图与同样fold的plain LoRA/I-LoRA同规模；相对强专家银行或更大骨干的GPU/完整权重优势必须另有实测，不能归因于这个训练期mask。当前文献核查不授权新实验，也不改变运行中三臂的任何设置。
