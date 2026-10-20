# FLAP参考实现

[Fluctuation-Based Adaptive Structured Pruning for Large Language Models](https://ojs.aaai.org/index.php/AAAI/article/view/28960)，AAAI 2024。

作者仓库：[CASIA-LMC-Lab/FLAP](https://github.com/CASIA-LMC-Lab/FLAP)，固定commit `3bb57db3449dd2fa04a5c2192de80e87e33be2b1`。服务器source/仅保留所需两个参考文件和LICENSE，版本及逐文件SHA见source_manifest.json；上游源码不进入本仓库Git。

本聊天CPU结构小测只借鉴WIFV与均值补偿，改为QA等权统计、视觉／语言FFN同半宽；不称作者全方法复现、新算法或已证本场景收益。没有运行其默认数据下载、CUDA入口或修改正式基线。
