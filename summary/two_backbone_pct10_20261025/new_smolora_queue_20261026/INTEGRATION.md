# 两组10% SMoLoRA接入范围

NEW一次只读实查（unix 1793014350.8718238）确认：CPU1221743运行原pct01-only协调器；10% `queue_new.json` 与 `dispatch_new.lock` 均不存在，也无10%模型/协调器进程。原1%8260/8265真实身份保全。因此原 `waiting_native_bf16_on_new` 是未接入等待，并非已可自动执行。

新外层 `analysis/baseline_pct01_pct10_capacity_20261026.py --watch` 保留原capacity主循环、四把原锁、活worker接管和1%优先。只在同一分配器的GPU1/3真实ownerless、显存/利用率满足原guard且本轮内存states无1%eligible时，派发两项白名单10% SMoLoRA；GPU0不分配，GPU2不给10%，原FOREVER规则不变。两项队列位于原10% `queue_new.json`。直接使用原 `dispatch.py --phase-request` 的preflight→reload→fresh training及真实EXIT/proof，不启动 `--machine new` 主循环。缺少EXIT或可能尚存worker保留orphan占位，不自动重试。

原10%原生25个py与BUILD目标哈希逐项一致，原BF16/数据/预算/方法/预检均不修改。`--once` 只读输出；本地仅AST检查，无模型、QA、评分、服务变化。READY的 `activated:false` 表示发布前状态，后续实际激活以独立ACTIVATION为准。

完整入表可直接复用原consumer的 --host new3090 --mode verify --run REL --write 单run入口；它不调用discover，实际完整exit0后才执行。14key限制仅存在于discover/once/watch，不能对本两项queue使用这些扫描入口；无需新增observer或核验代码。目前未消费、未评分。Qwen10% CL原预检失败保持关闭，不转移重试。

## 实际接管

2026-10-26实际补齐两10% SMoLoRA队列：发布6513d1d后，NEW原CPU1221743由1739063/start35800472/UID1004替代，原unit active/running；原1%8260/8265身份保全、0GPU信号、冻结科学源/policy全不变。queue_new真实只有onevision/qwen两SMo，均waiting，仍无预检/质量；只原1%优先后的GPU1/3空闲机会，NEW0和FOREVER2规则保留。证据new_smolora_queue_20261026/ACTIVATION、AFTER。首次ExecStop缺省字段及其次wchan窄匹配均在任何stop/marker前拒绝，外层字段读取/已知poll等待修正后接管成功，wrapper不改。

--once真实return0、两行只读metadata通过；随后原服务接管完成，两项queue真实持久化。首次交接KeyError: ExecStop、其次AssertionError: poll_schedule_timeout.constprop.0均在HANDOFF_STARTED与信号前；外层改为.get和接受Python原等待状态，未更改冻结源、运行数据或模型。原失败工具输出保留于对话；成功与完整原unit字节见HANDOFF_STARTED。
