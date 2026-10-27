# Qwen 1% CL-MoE完整结果

原运行完成19阶段，原观察器已唯一核验74格、21454最终QA及完整CP SHA。本次只复用缓存proof，没有再次评分或启动模型。五域宏F1为48.071713，AP19为47.122709；低于同Qwen朴素SeqFT-LoRA的48.972351。

完整S19参数文件4899367968字节，继续保存在NEW耐久磁盘，原RUN与全部回答不删。参数系数／权重布局及GPU资源仍待提取，不用checkpoint容器大小代替权重或GPU峰。原分配器已启动Qwen OSFT接替GPU2。

[原完整proof](../two_backbone_pct01_20261017/completion_observer_seven_20261018/completed/ebf69e0412424b46f33e.json) · [保存记录](RETAINED.json)
