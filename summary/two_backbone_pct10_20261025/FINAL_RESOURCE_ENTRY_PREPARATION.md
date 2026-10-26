# 两个已完成10% SeqFT最终模型的资源测量入口

2026-10-26。补齐主表缺测所需的独立入口，不是新方法或质量结论。当前未启动GPU测量、未加入分配器；主表资源列仍留空。

入口：`exp/final_seq_resource_20261026/worker.py`，当前SHA `aedc7e7b97d4f5be22a98d022c758d3c3c285dbc805e087b2a952c09de302400`。使用原完整S19图、原加载器、五域原hash-min DEV输入、每题32输出token。原始完整CP不改写、不删除、不重建；不调用带训练和删除行为的checkpoint_probe。不新增F1评分。

TTFT从输入准备前到首token同步回调；解码速度为后31个token除以首末token间隔。保留原有限logit检查及每token同步，故标为受控测量，不能当生产吞吐。文件哈希在窗口前，张量摘要在计时及峰值读取后。模型加载时间（含原loader完整CP校验）单列，并属于总时间。峰值使用PyTorch allocated/reserved口径，不冒称NVML整进程峰。

baseline_review核过原OneVision完整generate及prompt切片接口、两sealed proof字段、decoder对象与method.layers身份检查。修改注册路径字符串断言以免alias顺序造成误拒，不改模型结构。源码审核不等于GPU加载或测量通过。

## 目标机实际元数据检查

unix `1793025772.857432`，OLD原conda环境执行了两次`validate_metadata`，均成功，Torch未导入、0模型、0生成、0评分、未读取checkpoint载荷。两条完整CP只核原proof/metadata及文件大小；GPU实际加载时仍由原loader做一次全SHA校验。

| 骨干 | 固定窗口数 | CP字节数 | metadata秒 | 绑定SHA |
| --- | ---: | ---: | ---: | --- |
| LLaVA-OneVision | 5 | 1926008036 | 0.116525 | a2b4e88ae6088332aaf60ac1cbd371b0c42de27c7722c0a240bc2ff7dba9eb60 |
| Qwen | 5 | 4562998300 | 0.036013 | d33c6006e518b888ee99d8b29db4e6b3dc4ac81f245bb7460d60bdefb36b22c5 |

OLD工作目录为`/data2/liyapeng_grp/program/LLMCL/.deployment/final_seq_resource_metadata_20261026`，含两binding及METADATA_VALIDATED.json。实际原run仍在原tmpfs，NEW耐久pin独立保留。

准备过程中两次模型前失败保留：首次运输脚本误把nvidia-smi末尾空行当GPU记录（IndexError）；第二次原资产目录缺SPLIT.json（FileNotFoundError）。前者仅修记录解析，后者复制原d8e2…清单及已有helper原字节到独立工作目录。未修改原任务、科学条件、旧source或GPU进程，也没有模型重试。第三次完整metadata成功。此记录不能代替实际GPU窗口结果。

同期只读核验：unix1793025496原分配器609661/start771089493/UID1084确实存活，条件适应三臂仍无SLOT；unix1793025550旧四正式worker均存活，LLaVA PAS已完成S12、OSFT在S6评价。10%完整正式结果仍2/16，无新分数可写主表。
