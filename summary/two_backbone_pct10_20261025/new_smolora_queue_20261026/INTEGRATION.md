# 两组10% SMoLoRA接入范围

NEW一次只读实查（unix 1793014350.8718238）确认：CPU1221743运行原pct01-only协调器；10% `queue_new.json` 与 `dispatch_new.lock` 均不存在，也无10%模型/协调器进程。原1%8260/8265真实身份保全。因此原 `waiting_native_bf16_on_new` 是未接入等待，并非已可自动执行。

新外层 `analysis/baseline_pct01_pct10_capacity_20261026.py --watch` 保留原capacity主循环、四把原锁、活worker接管和1%优先。只在同一分配器的GPU1/3真实ownerless、显存/利用率满足原guard且本轮内存states无1%eligible时，派发两项白名单10% SMoLoRA；GPU0不分配，GPU2不给10%，原FOREVER规则不变。两项队列位于原10% `queue_new.json`。直接使用原 `dispatch.py --phase-request` 的preflight→reload→fresh training及真实EXIT/proof，不启动 `--machine new` 主循环。缺少EXIT或可能尚存worker保留orphan占位，不自动重试。

原10%原生25个py与BUILD目标哈希逐项一致，原BF16/数据/预算/方法/预检均不修改。`--once` 只读输出；本地仅AST检查，无模型、QA、评分、服务变化。READY的 `activated:false` 表示发布前状态，后续实际激活以独立ACTIVATION为准。

后续完整入表仍有一个明确consumer接入口缺口：原native observer的discover固定遍历14项并直接索引queue所有key，而此NEW队列严格只有两SMo。正式完成后须使用只发现这两项的受限入口，复用原单run完整核验；本次不新增observer、不冒认已完成或已核验。Qwen10% CL原预检失败保持关闭，不转移重试。
