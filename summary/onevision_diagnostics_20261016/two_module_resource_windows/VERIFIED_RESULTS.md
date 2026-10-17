# 原型训练窗口：修正峰值统计范围

只读已完成证明绑定的原始资源日志，活动/排队GPU源码不变；TEST文件仅校验哈希，不解析答案或分数。原mixed maxima原样保留在JSON中，不能当作纯TRAIN峰值。

| 任务/条件 | GPU allocated前→后MiB | 增长bytes | TRAIN重置allocator峰值MiB | TRAIN采样GPU峰值MiB | TRAIN采样NVML峰值MiB | TRAIN窗口秒 | 方法窗口额外评分题次 |
|---|---:|---:|---:|---:|---:|---:|---:|
| T02/control | 1789.398→1789.399 | 512 | 2700.345 | 1892.799 | 3600.000 | 13.848 | 10 |
| T02/A_only | 1757.398→1789.399 | 33554944 | 2700.444 | 1901.713 | 3868.000 | 17.885 | 40 |
| T02/B_only | 1757.398→1789.399 | 33554944 | 2700.444 | 2159.292 | 3870.000 | 13.952 | 30 |
| T02/A_B | 1757.398→1789.399 | 33554944 | 2700.444 | 1838.265 | 3870.000 | 19.327 | 60 |
| T11/control | 1789.398→1789.399 | 512 | 2700.722 | 2009.771 | 3658.000 | 10.903 | 10 |
| T11/A_only | 1757.398→1789.399 | 33554944 | 2701.038 | 1824.452 | 3868.000 | 21.831 | 40 |
| T11/B_only | 1757.398→1789.399 | 33554944 | 2701.038 | 1908.448 | 3870.000 | 15.704 | 30 |
| T11/A_B | 1757.398→1789.399 | 33554944 | 2701.038 | 1979.972 | 3870.000 | 22.902 | 60 |
| T12/control | 1789.398→1789.399 | 512 | 2401.787 | 1869.084 | 2446.000 | 5.424 | 8 |

每行来自同一PID的before/after与其间资源样本。GPU allocator峰值读取after端点，并验证源代码先reset以及before peak等于当前占用；周期采样可漏过峰值，不能替代此计数。CPU RSS/PSS/USS和NVML仍只承诺采样最大值，原始bytes/端点全部写入[RESOURCE_WINDOWS.json](RESOURCE_WINDOWS.json)。

训练后先清理优化器/缓存再取after，增长很小不代表峰值很小；参数、候选、原始梯度的CPU负担也不能只看GPU allocated。原控制额外运行过一次原训练器作为准入，这部分QA/计算在JSON单列，不包含在方法窗口，也没有被抹掉。

## 相同参数也能有不同的增长值

控制组在窗口前已经执行过原训练器准入，其他条件没有这一预热，必须单列这一差别。若两个条件最终参数、原始梯度和优化器文件SHA相同，却有不同的before与增长，不能把增长差直接归因为算法新增内存；本审计未定位具体运行时buffer，也不宣称泄漏。

- T02/A_only：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 11]。
- T02/B_only：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 11]。
- T02/A_B：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 11]。
- T11/A_only：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 12]。
- T11/B_only：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 12]。
- T11/A_B：相对control的before差-33554432bytes、after差0bytes、增长差33554432bytes；final_adapter/raw_gradient/optimizer三文件字节完全相同：True；准入QA分别[0, 12]。

这是口径修正，尚未证明某方法比LoRA更省内存或更高F1。四组目标QA/参数/初始函数与原生输入的门槛仍独立验证。
