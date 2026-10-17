# 训练窗口内存审计

只纳入before／after均已落盘的训练窗口；不公布未完成知识病例的准确率。所有原始量单位bytes，MiB=bytes/1048576。

## rank1_correct：窗口1

可训练参数67584→67584；其字节270336→270336。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2659840000 | 2689875968 | 28.645 | 2689875968 |
| Pss_bytes | 2428427264 | 2448841728 | 19.469 | 2448841728 |
| cpu_uss_bytes | 2342920192 | 2354413568 | 10.961 | 2354413568 |
| gpu_allocated_bytes | 1843037184 | 1877330944 | 32.705 | 1881961984 |
| gpu_reserved_bytes | 3269459968 | 3269459968 | 0.000 | 3269459968 |
| gpu_process_nvml_bytes | 3625975808 | 3659530240 | 32.000 | 3659530240 |

CUDA allocator重置后峰值allocated=1913477632、reserved=3269459968。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## 解释限制

- TRAIN-after points retain optimizer state; before has optimizer created but Adam states not yet initialized
- parameter bytes exclude gradients, Adam states, activations, runtime buffers and CUDA cache
- RSS/PSS/USS and processNVML peaks are periodic sampled maxima; may miss short spikes
- CUDA allocator peaks after reset come from the endpoint-after snapshot, not old pre-reset before peak
- VmHWM is whole-process lifetime and is NOT a per-training-window CPU peak
- same PID/phase/time range excludes endpoint inference between windows; retained caches can affect baseline
- memory increase with fixed parameter count is not evidence of new modules or a memory leak

参数量不变时，占用增加不能直接解释为新增模块，亦不能仅凭一次增长排除或确认泄漏。逐PID、时间范围、原始端点和追加日志前缀SHA见[JSON](MEMORY_WINDOW_AUDIT.json)。
