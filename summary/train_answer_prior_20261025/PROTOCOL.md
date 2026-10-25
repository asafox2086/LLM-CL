# TRAIN答案先验与选项结构：一次有限审计

人类最新明确“可以看看10%会不会也这样，有可能是数据太少”。本审计必须**并列1%与10%**，只回答合法训练监督的答案／选项位置分布，是否为已有金融选择集中现象提供了一个需要进一步功能干预的具体竞争解释。它不训练模型、不读取任何TEST文件、不生成或评分，不据不均衡认定原因，不修改正在运行的1%／10%正式实验。最终功能问题是两种数据量下各task刚习得表现／金融选项集中度，TRAIN计数不能代替该结论。

## 固定来源与读取范围

运行必须显式提供`LLMCL_ROOT`，不得从脚本所在临时目录猜根目录。唯一两个池为正式1% TRAIN1313与已冻结10% TRAIN13105；不是1213FIT，不使用正式teacher或已学权重。

- 1% manifest：`data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json`，SHA256 `36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777`。
- 10% manifest：`data/FiveDomainCL/five_domain_native_cl_pct10_20261025/pct10/manifest.json`，SHA256 `02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7`。

只打开两manifest及其task2–11医学、task16金融的`files.train`。全池1313／13105和19task仅核manifest计数；不打开其它领域TRAIN、任何TEST、预测、图像、tokenizer、checkpoint或原parquet。每个实际打开的选中TRAIN文件SHA和行数须匹配manifest。金融正式1%为89、10%为887，以绑定清单核实；88是另一FIT视图的可能数量，不能混用。

## 字段与描述量

转换器固定字段为`id/dataset/question/answer/answers/context/metadata`。FinMME的`context`是原options，须与`metadata.options`一致；其`question_type`区分single_choice／multiple_choice／numerical。SLAKE保留原`q_lang/answer_type/content_type`；closed不是天然MCQ，无显式options时不制造选项或A/B标签。

主agent已只读两个池T2／T4／T16各前12条TRAIN的脱敏syntax（本实现者不读取其原文）：金融context为str，空或行首`A:`／`B:`等，单选原答案如`A`、多选为连续字母如`ABC`／`ABCD`、数值为原字符串；SLAKE context为空、options为None、answer_type为`OPEN`／`CLOSED`，CLOSED示例为`Yes`／`No`。这仅是有限schema检查，不假定完整池全部符合；未知编码／类型、多个参考、空额外参考都保留显式flag及原频数，不能静默排除。

解析只依据上述实际schema与原编码，不用gold补全选项。按池、域、task、题型／语言／开放闭式／原选项数分别报告原主训练答案频数和原参考列表及集合频数。不同原字符串不合并；字母集合排序仅为附加描述，不改变训练目标或评分。多选不能拆成多条独立训练标签并据此替代原答案集合；numerical单列，不混进字母分布。

金融选项需单独报告无options、严格解析失败、空／重复内容、非连续或重复标号、gold无法对应原标号、多选编码不支持等覆盖量。对解析成功记录，保守检查题干引用字母、all／none-of、以上／以下、顺序／位置语义、选项间互指等风险；全部标记可重叠并报告并集。满足语法与这些固定检查仅称“未发现预定排列障碍”，不认证语义可逆。排除／失败行仍计入完整分母，不静默只展示可解析子集。

首次执行前固定区分一般集合引介与排列障碍：题干`which/what of the following`、`select/choose/identify ... following`、`下列`、`以下哪项`等只记`question_collection_rubric_present`字面描述flag，不作为候选障碍；先移除这些固定集合引介片段再检查题干的其它语义flags。题干其它`following`出现同样只作该描述计数。选项内容中的following／above／below，以及原former／latter、first／second、显式字母互指等仍保守阻挡。报告全部flag与真正障碍并集分别计数，不因一般引介把MCQ资格误清零；此分离在读取结果前完成，不据结果调规则，仍不认证语义可逆。

## 隐私、实施与费用

实现仅用Python标准库，不导入torch、模型运行时、旧审计模块或评分器。旧option审计模块含torch且读取TEST，禁止直接import。本次不调用任何外部程序或网络。

公开JSON／FINDINGS只含计数、比例、安全短字母／yes-no标签及原ID／非短答案的SHA256；不输出问题、options内容、长答案、verified_caption或related_sentences。数值答案也按原字符串哈希保存频数，不公开原数值。完整原文仍仅在已有私有TRAIN文件，不创建QA副本。

输出到`summary/train_answer_prior_20261025`，已有完成产物则拒绝覆盖；入口记录STARTED，结束一次写审计证据和FINDINGS，记录源码／协议／输入SHA与实际秒数。0模型／decoder／生成／更新／新评分／TEST文件读取。入口／解析失败应保留故障，不以自动换parser反复读取数据。

## 正负出口与结论边界

- 若TRAIN标签没有明显多数集中，不能再把当前抽样多数类复制当现成解释；这不排除预训练字母先验、条件位置偏好、末批次影响或有限优化，也不构成这些解释的独立否定。
- 若存在集中，只确认合法监督中确有该先验，不证明它造成已封TEST的集中回答，不据此自动按先验重加权或纠正logits。
- 若有效金融MCQ覆盖足够且确有可逆排列资格，后续才可能独立评审同QA／同曝光的FIT／DEV位置干预，区分位置与内容判断；训练前后都须观察。若多数题无可靠可逆结构，则不强推排列工具。没有本审计到模型实验的自动准入。
- 医学只在原闭式答案上定位一个有限问题，不将金融字母机制推广到所有域；相同标签的均衡也不等于条件判断已学会。原Answer F1、原始数据和正式主表不改。

已有[回答失效分析](../final_weight_reduction_20261024/response_failure_profile_v1/FINDINGS.md)定位金融刚习得即大量错误；[旧格式审计](../onevision_diagnostics_20261016/ANSWER_BUDGET_AND_OPTION_AUDIT.md)仅核输出内容／字母格式，未做本TRAIN先验审计。[NAACL2024选项顺序研究](https://aclanthology.org/2024.findings-naacl.130/)说明顺序干预有明确先例，其oracle顺序差不作为本方法预期收益；[ICML2021校准研究](https://proceedings.mlr.press/v139/zhao21c.html)区分示例多数／近因／常见token偏置，不能直接当VLM监督训练因果或无需实测可用的校正配方。本次没有算法性能或端侧成功声明。
