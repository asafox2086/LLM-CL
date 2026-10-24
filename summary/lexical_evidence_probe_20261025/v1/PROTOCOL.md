# 独立词面证据能否承接紧凑视觉系统的部分回答功能

## 为什么做，结果如何改变设计

前一固定小视觉/S1仿射接口完整权重少34.96%、五个匹配GPU峰下降，但自然DEV F1为29.166667，父61.5；医学和交通也下降。名称/数字变化不能直接识别OCR损伤原因。本诊断不改旧图、不训练、不改桥、不追加数据拟合，也不把旧质量失败改成成功。具体问题是：独立来源的图中文字证据，是否能通过单独词面通路补回部分回答，而不是必须重建整个父视觉特征？

单纯原图/错图NLL无法区分学生先验、空间映射与接口失配，正负均不足以选结构，已由两子agent否决。这里有具体分叉：若正确词面证据有作用，才值得单独评审较小视觉通路加真正轻量读字通路；若无作用，不继续本固定来源/格式/冻结图的词面通路。不因阴性否定所有OCR或有训练融合。

原正式1313 teacher不参与。两固定图来自前一合法1213 FIT父和已完整核验学生；其语言见过完整FIT，仅学生桥学习限定S1。本诊断只用原20自然DEV，已反复探索，单seed，不是新留出、跨seed、非劣或正式新方法。

## 来源、全部样本与固定错配

原始TextVQA TRAIN parquet的question_id/image_id/ocr_tokens三列，按原20自然DEV的source_id和original_image_id绑定，20条全保留，18非空。只此现有原始投影，未读取raw图像列、问答列；未按gold或问题挑词、补字或纠错。完整正确OCR原顺序逐词以换行拼接，不裁剪；没有位置框。OCR可能含答案，关键是来源独立于gold筛选，不删除命中答案的文本。

官方TextVQA提供Rosetta自动OCR，但本地投影的精确Rosetta版本未另外认证，不称人工无误真值或最佳识别上界。projection SHA固定为15e36a2809973a174084d4d4f12978ac0d1415ca288ce07b91e524d9bcdf742f。它是已有系统产物的预先计算词面证据，不是已实现的轻量OCR。

18个非空项按sha256('lexical-evidence-20261025:'+id)排序；每项选循环向后首个不同原image_id且不同既有SPLIT组的非空项作donor，完全不用答案判定错配。将donor完整OCR按同tokenizer编码后固定循环或截断到recipient正确OCR body的token数。两空项两臂body均空，保留在全20均值。wrong可能重复、在token边界截断且不连贯，是人为负控，不等于真实OCR错误分布。

## 两图与三个条件

父和学生各自的noOCR锚仅从前一VERIFIED及被绑定的完整预测复制既有20条分数，不重新生成/评分。全100旧源proof与预测摘要、父/学生digest/严格重载、runtime/manifest/SPLIT链冻结。不重新解释原candidate为保质。

每个新进程按原真实图和原输入重建20条，核无OCR的prepared/transport/raw pixel/pack摘要与其原预测逐ID完全相等。每QA只做一次视觉编码，固定packed视觉特征供同一题correct/wrong两种干预使用；这是诊断共享输入，不是部署缓存收益。

保留原完整输入token序列，在最后一个user im_end标记前插入固定token片段：`\nImage text (OCR; may contain recognition errors):\n`、body、`\n`。原问题/任务instruction/图像token、全部原参考、生成EOS/cap保持；不改chat模板或筛选提示词。两臂只替换同一body区间token IDs，区间外逐值一致、总长度完全相同、attention_mask全1、image token和packed视觉值相同。body不得有special token；禁止截断全部输入、问题或OCR。token区间由实际token IDs构造，不用字符索引猜测。

每图每题correct/wrong各一次gold原NLL和一次greedy生成，顺序依固定id hash奇偶交替。共40次视觉、80NLL、80生成、80次原token_f1新评分；0新noOCR请求、训练、反向、optimizer、TEST或bootstrap。NLL是原target截断512+EOS路径，辅助解释，不替代F1。

## 预定读数和出口

全20条分别报告父/学生noOCR、correct、wrong的域F1/NLL，以及全部逐题正负。报告学生correct增益、父correct增益、correct下剩余差距，防止把父被干扰导致的差距缩小说成学生恢复。空OCR两题单列，其两臂相同但header相对noOCR可能影响输出。

仅当学生correct F1同时严格高于其原noOCR和wrong，且父correct−学生correct的差距小于原32.333333pp，记为支持进入真实轻量读字通路的独立可行性评审；否则关闭本固定证据通路。任何正结果只支持部分功能可由独立词面证据承接，不识别OCR唯一因果、学生先验缺失或接口唯一错误，不宣称已承接全部自然功能。父同OCR是未来公平质量参照；不能只比未增强父。

医学/交通尚未解决，不自动全五域/CL/1313/21454，不填Ours。未来真正OCR模型的检测/识别完整权重、词典、运行内存及误差传播全部计费；本次预计算OCR不能拿来声称新增部署优势。旧两图规模/峰数据只为动机，不是本干预的真实OCR完整成本。

## 预算与完成边界

仅OLD真正空GPU1，父/学生同卡串行，各首次STARTED后900秒，总1800模型秒（含加载、摘要/逐ID输入核对、全部80NLL/80生成及保存），controller上限1900秒。metadata/必要token CPU控制、运输/唯一核验/归档及历史费用另列。首次真实失败/中断/超预算关闭本固定诊断，不续CP、换prompt/donor、扩预算或自动版本。保全两机正式和他人资源，不向GPU发信号、不占NEW卡。

源：[TextVQA官方OCR说明](https://textvqa.org/download/)、[LLaVA-Read](https://arxiv.org/html/2407.19185v1)。前者说明独立词面来源，后者提供已有功能分工先例；不证明此固定少数据冻结图有效，不把模块组合称原创。
