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

## rank1_correct：窗口2

可训练参数67584→67584；其字节270336→270336。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2689347584 | 2689380352 | 0.031 | 2689380352 |
| Pss_bytes | 2449001472 | 2449034240 | 0.031 | 2449034240 |
| cpu_uss_bytes | 2354573312 | 2354606080 | 0.031 | 2354606080 |
| gpu_allocated_bytes | 1877330944 | 1877330944 | 0.000 | 1883022336 |
| gpu_reserved_bytes | 3269459968 | 3269459968 | 0.000 | 3269459968 |
| gpu_process_nvml_bytes | 3659530240 | 3659530240 | 0.000 | 3659530240 |

CUDA allocator重置后峰值allocated=1913477632、reserved=3269459968。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank1_permuted：窗口1

可训练参数67584→67584；其字节270336→270336。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2463592448 | 2493067264 | 28.109 | 2493067264 |
| Pss_bytes | 2230179840 | 2249977856 | 18.881 | 2249977856 |
| cpu_uss_bytes | 2144374784 | 2155139072 | 10.266 | 2155139072 |
| gpu_allocated_bytes | 1843037184 | 1877348352 | 32.722 | 1881829888 |
| gpu_reserved_bytes | 3269459968 | 3269459968 | 0.000 | 3269459968 |
| gpu_process_nvml_bytes | 3625975808 | 3659530240 | 32.000 | 3659530240 |

CUDA allocator重置后峰值allocated=1913477632、reserved=3269459968。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank1_permuted：窗口2

可训练参数67584→67584；其字节270336→270336。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2493272064 | 2493276160 | 0.004 | 2493276160 |
| Pss_bytes | 2250211328 | 2250215424 | 0.004 | 2250215424 |
| cpu_uss_bytes | 2155376640 | 2155380736 | 0.004 | 2155380736 |
| gpu_allocated_bytes | 1877348352 | 1877348352 | 0.000 | 1882018816 |
| gpu_reserved_bytes | 3269459968 | 3269459968 | 0.000 | 3269459968 |
| gpu_process_nvml_bytes | 3659530240 | 3659530240 | 0.000 | 3659530240 |

CUDA allocator重置后峰值allocated=1913477632、reserved=3269459968。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank8_correct：窗口1

可训练参数540672→540672；其字节2162688→2162688。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2355724288 | 2411053056 | 52.766 | 2411053056 |
| Pss_bytes | 2120751104 | 2166384640 | 43.520 | 2166384640 |
| cpu_uss_bytes | 2034507776 | 2071162880 | 34.957 | 2071162880 |
| gpu_allocated_bytes | 1842766848 | 1880845312 | 36.314 | 1885414400 |
| gpu_reserved_bytes | 3254779904 | 3254779904 | 0.000 | 3254779904 |
| gpu_process_nvml_bytes | 3611295744 | 3644850176 | 32.000 | 3644850176 |

CUDA allocator重置后峰值allocated=1918884352、reserved=3254779904。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank8_correct：窗口2

可训练参数540672→540672；其字节2162688→2162688。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2411429888 | 2411438080 | 0.008 | 2411438080 |
| Pss_bytes | 2166761472 | 2166768640 | 0.007 | 2166769664 |
| cpu_uss_bytes | 2071539712 | 2071547904 | 0.008 | 2071547904 |
| gpu_allocated_bytes | 1880845312 | 1880845312 | 0.000 | 1897194496 |
| gpu_reserved_bytes | 3263168512 | 3263168512 | 0.000 | 3263168512 |
| gpu_process_nvml_bytes | 3653238784 | 3653238784 | 0.000 | 3653238784 |

CUDA allocator重置后峰值allocated=1918884352、reserved=3263168512。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank8_permuted：窗口1

可训练参数540672→540672；其字节2162688→2162688。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2624802816 | 2651025408 | 25.008 | 2651025408 |
| Pss_bytes | 2385718272 | 2401466368 | 15.019 | 2401466368 |
| cpu_uss_bytes | 2298302464 | 2304192512 | 5.617 | 2304192512 |
| gpu_allocated_bytes | 1842766848 | 1880862720 | 36.331 | 1886522368 |
| gpu_reserved_bytes | 3254779904 | 3254779904 | 0.000 | 3254779904 |
| gpu_process_nvml_bytes | 3611295744 | 3644850176 | 32.000 | 3644850176 |

CUDA allocator重置后峰值allocated=1918884352、reserved=3254779904。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## rank8_permuted：窗口2

可训练参数540672→540672；其字节2162688→2162688。

| 指标 | before bytes | after bytes | 增量MiB | 窗口采样最大bytes |
| --- | ---: | ---: | ---: | ---: |
| VmRSS_bytes | 2611548160 | 2611552256 | 0.004 | 2611552256 |
| Pss_bytes | 2361989120 | 2361993216 | 0.004 | 2361993216 |
| cpu_uss_bytes | 2264715264 | 2264719360 | 0.004 | 2264719360 |
| gpu_allocated_bytes | 1880862720 | 1880862720 | 0.000 | 1888583680 |
| gpu_reserved_bytes | 3263168512 | 3263168512 | 0.000 | 3263168512 |
| gpu_process_nvml_bytes | 3653238784 | 3653238784 | 0.000 | 3653238784 |

CUDA allocator重置后峰值allocated=1918884352、reserved=3263168512。这是after端点读取的重置后峰值，不能取before中此前推理留下的peak。

## 解释限制

- TRAIN-after points retain optimizer state; before has optimizer created but Adam states not yet initialized
- parameter bytes exclude gradients, Adam states, activations, runtime buffers and CUDA cache
- RSS/PSS/USS and processNVML peaks are periodic sampled maxima; may miss short spikes
- CUDA allocator peaks after reset come from the endpoint-after snapshot, not old pre-reset before peak
- VmHWM is whole-process lifetime and is NOT a per-training-window CPU peak
- same PID/phase/time range excludes endpoint inference between windows; retained caches can affect baseline
- memory increase with fixed parameter count is not evidence of new modules or a memory leak

参数量不变时，占用增加不能直接解释为新增模块，亦不能仅凭一次增长排除或确认泄漏。逐PID、时间范围、原始端点和追加日志前缀SHA见[JSON](MEMORY_WINDOW_AUDIT.json)。
