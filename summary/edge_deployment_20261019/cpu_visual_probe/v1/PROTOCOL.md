# CPU视觉token成本探针 v1

目标：检验端侧真正的视觉序列开销，以及复杂token选择的成本是否配得上其节省。仅使用已导出的OSFT模型和五个固定TRAIN输入；不是新方法、正式F1或手机实测。原始基线／GPU1研究不改。

三条件全部预先保留：dense、uniform50、author_divprune50。后两者对非结构性视觉token保留round(N*0.5)，图像行分隔newline全部保留，原图／问题／文字token不改。作者DivPrune两函数从固定commit 799e2d9源码AST原样抽取；这里只适配OneVision投影后packed特征，不冒充作者全部配方复现。不按TEST选ratio。实际重复索引或非法输入严格拦截留证，不能静默改作者算法。

每条件独立CPU进程、同一模型包／CPU2线程／原common401408图像上限。按seed42 ID哈希从五域各首个原生任务TRAIN选一题，共同五题。同原始processor提示词构造，忽略视觉缓存，重新图像编码，之后删减视觉embedding并按原桥接逻辑修正image占位数量。原native位置／文字逐项核验，不改图片或答案。各请求强制8新token用于成本测量，不用这些截断输出评估Answer F1。

完整计时包括PIL、processor、视觉编码、筛选、prefill、首token和后7token。筛选时长单列，进程20ms采样生成窗口峰值RSS，加载时间／前后RSS/PSS/USS与服务器CPU背景记录。五TRAIN题只提供成本可行性点估计，不是完整TEST、显著性或手机硬件优势。OSFT CPU静态导出尚未与原GPU全量质量做等价确认，不将原AP移植为这个CPU模型的已证F1。原16基线、RITE、所有负结果及完整确认保留。
