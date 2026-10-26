# 医学回答配对准备：独立源码审阅

接受root按既定路径唯一执行，未发现阻断。对象 `analysis/medical_paired_answers_20261026.py` SHA256：`4131dbdfe13d23ef3b46dd52ba8dba4f5df82bfd652d43eb35a6479524e69629`，258行，stdlib AST通过。

已实读完整源码与 `MEDICAL_PAIRED_QUESTION.md`。来源通过两bundle内相同固定formal proof、config／manifest及原prediction SHA闭合；6个payload范围限定为两个完整TEST和四个时点预测。原375＋116＝491个ID按原顺序配对，references／metadata／原生成cap／native label逐项核对，不按原分选样。四格F1均值只从既有逐题分数复制聚合并与原verified_cells绑定，不调用评分器。原proof没有EXACT均值，源码如实仅复制封存预测内EXACT、核有限范围及同raw／同reference分数一致，未伪造独立EXACT均值验证。

全部491对保留在私有账本；仅raw字符串不同时进入盲包，未变化项记UNCHANGED_RAW。A/B方向、独立blind key与打乱顺序使用三个不同ID哈希分域，均不依赖分数。盲包逐项只含key、原question及answer_A／answer_B，没有原ID／原ID摘要、task、stage、references、原分或方向字段。公开准备收据仅含来源／输出摘要、计数、费用及范围，不含QA正文；既有原分和方向仅留私有账本。

后续两位标注者只看changed-only盲包，逐项一次判断SAME_CONTENT／DIFFERENT_CONTENT／UNDETERMINED并给短理由；不看私有账本／逐题分数方向，不判医学正确性。部分重叠、信息增减或含义不明保留未知；双人分歧保留，不能由此直接选择知识模块或声称纯机制因果。

本审阅未执行prepare或任何聚合／评分／模型／服务器操作，未读取原QA／预测正文、privatepair账本或逐题分数；仅读源码、问题和此前必要字段schema。接受源码不代表已完成准备或语义标注。
