# 同一完整模型的独立重载与五窗资源观测

正式完整质量尚在评估，本页只记录已经完成的export/reload/五窗。未运行最终verifier，不把资源通过当质量—成本目标成功。

完整新CP SHA246e876954f9aa71af4fcd0d9d9315060fbc745c642c81894acd970efb25f4a1，1787572114 bytes；同c重建的全图digest3ea480aec723afb0d8660853c274312b9aab0580bc7674f96d35d6f7180c7683。export762104真实退出0/67.878184秒，旧bank/learner清除；新进程763648/start797496721独立meta构造/strict完整重载，不借外部base权重，五窗完成后继续原完整TEST。完整CP和processor已在NEW耐久保存并全SHA通过，OLD最终件保留，路径见DURABLE_MODEL.json。未把容器字节当神经权重或GPU内存。

| 同一原输入窗口 | 静态图peak allocated MiB | 原PAS peak allocated MiB | 降幅 | 静态图peak reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| 自然 | 2166.100586 | 2959.415039 | 26.8065% | 2518 |
| 医学 | 2041.105957 | 2774.937988 | 26.4450% | 2374 |
| 遥感 | 1856.979004 | 2490.443359 | 25.4358% | 1968 |
| 金融 | 2041.101074 | 2735.135254 | 25.3748% | 2356 |
| 交通 | 2166.091797 | 2882.101563 | 24.8433% | 2506 |

五窗按原顺序/每窗32新token、同OLD物理GPU1/UUID，raw/prepared/transport hash及shape逐值匹配原PAS。窗口包括prepare/vision/decode，先于全TEST生成；NVML仅前后观测，不称NVML峰。不是全TEST最坏峰、加载全程峰或速度/TTFT测量，不将推理峰与训练峰混用。

静态图893675552个FP16神经参数、新增0，纯权重1787351104B=1704.550842MiB；原PAS纯权重2350436416B=2241.550842MiB，另必要mask336个FP32标量/1344B，总必要载荷2241.552124MiB。纯神经权重少537MiB，总必要载荷少537.001282MiB/23.956672%。该差异包括原FP32专家与FP16dense表示，不能归为同dtype最优压缩或对所有低新增方法独占优势。

resource原文件SHA9bf4a520231b1296ea9ee615d913409367b3593980192c54ca4bb6547e06a515；RESOURCE_COMPLETE/STARTED/export COMPLETE/EXIT均与binding469735a796879588c1df722b12a5573010da3a1210d0453e8883c33f7c2268dc及同CP关联。research_review独立核旧PAS物理卡/输入/字节和本阶段记录，接受上述早期资源口径；没有读取全TEST部分质量或重评分。

完整F1、AP19和19小类仍待真实终态唯一核验；原100探索QA的53.760950不能填正式表，原PAS46.619976也不能移植。FWT/BWT本确认不测，不能借用。
