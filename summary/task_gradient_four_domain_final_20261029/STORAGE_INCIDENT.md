# OLD测量保存异常与基线退出

真实检查：/dev/shm free=4096B。resource_s16_v1的STARTED完整，但windows.json、FAILURE.json、EXIT.json为0B，worker439844已gone，无法声称资源测量成功或获知其实际退出码。三旧baseline284532/284533/288261均gone；PaRSP最近完整S13、OSFT最近完整S6。故障与满盘同时观察，具体各进程末尾仍需恢复审计，不把stale run_status当live。完整Ours质量及CP eeaf...不受此空文件尝试影响。

已按既有人类授权清理PaRSP S11/S12早期payload释放11752270612B、OSFT S01..04释放12129157634B；分别全SHA核保S13/S6及其它指定保护payload，全部QA/JSON保留，receipt写/data2持久目录。当前tmpfs实测23881445376B可用，persistent data2约2.7TB可用。

原resource_s16_v1全部现场保留。相同测量worker/binding/最终CP于新独立输出/data2/liyapeng_grp/program/LLMCL/exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/resource_s16_storage_retry_v2已启动；unit llmcl-taskgrad-resource-s16-storage-retry-20261029.service。只改输出位置，不改图/输入/dtype或科学配置，不覆盖失败文件。尚无已核资源数值。

下一步从完整S13/S6恢复旧基线（原异常前缀保留，新分支处理截尾logs），Qwen PaRSP须先查真实最新CP。研究probe完成后仍优先执行已准备Seq S16终评。资源/基线新身份须查真实收据，不重复已执行清理。
