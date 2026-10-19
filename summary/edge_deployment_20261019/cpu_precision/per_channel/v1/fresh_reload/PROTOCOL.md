# 逐通道版本独立部署重载

原v2三组的fresh_reload协议原样用于这一组：只从本部署config／state／spec构建，不从训练图或预训练权重加载；所有state逐值精确、五个固定TRAIN的8token逐值匹配。只有研究目录／worker／单条件标签与等待完成标记不同。独立进程全部CPU、2线程；新模型完成释放后再加载，和原研究流合计最多两个CPU模型，不占GPU。

相同QuantizedDynamicLinear类可严格加载保存的per-channel packed权重；实际小CPU门槛已验证其qscheme／输出精确，完整0.9B和五题重载仍必须真实完成。原型源码／READY不改，失败留证无重试。没有TEST或完整质量结论，不填主表。共享加载器和运行库依赖另列，不能以当前FP32序列化镜像为最小公平存储基线。
