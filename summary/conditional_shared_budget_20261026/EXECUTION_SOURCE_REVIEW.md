# 执行接口源码审阅

结论：**accepted，当前未见阻断性的producer/consumer接口错误。** 接受仅指可进入root的执行决策，不表示已运行、质量成立、GPU可用或论文目标完成。

本次只读交叉审阅新verify.py、queue_adapter.py及其实际依赖的controller/worker/adaptation，原position wrapper、dynamic dispatcher与原dispatch.main。已与两位实现者对齐当前字段并确认候选版本；仅AST/compile检查，0模块main、0Torch/模型、0QA/预测读取、0评分、0服务器连接或进程动作。

## 已核合同

- controller串行shared train/eval、uniform train/eval、soft train/eval；单张交付GPU，不自行找卡或建立服务。每臂5400/1800秒，worker另有本phase限时。独占新output，各arm独立目录；失败停止且仅可终止本批身份匹配的worker进程组。
- PHASE_*_EXIT在终态phase项增加proof字段之前落盘。verify正确比较EXIT与终态phase去掉started_sha256、complete_sha256、checkpoint后的对象，未要求两者逐字全等。started_sha256实际指worker的train/eval_STARTED.json，外层PHASE STARTED单独核验。
- 六phase真实返回0、身份消失、顺序、GPU、时间、源/data binding、完整CP/receipt/mode、96张量初始化/87更新/19stage、38个before/acquired事件及输入配对按实际producer字段读取。metadata全通过后才O_EXCL写VERIFY_STARTED，原tokens/score AST只对900份新答案调用一次；没有旧F1重评、模型或TEST消费。
- shared/uniform终态是实际GPU FP32合并后的FP16图，soft是显式FP32 r8小银行；meta strict assign重建及全部buffer/别名/配置收据按两种实际布局核，不把纯FP16 receipt错误套给soft。五资源窗来自独立eval进程、先五窗后100 DEV；新习得仍来自原训练图。before/acquired的任务为空时保留空项。
- 原coordinator仍使用baseline root与原位置admission；仅research child获得独立science root。queue wrapper的source_dir严格等于science root中的新目录，未将两个root混用。
- 新wrapper先调用原position.Slot.wrap，再包conditional idle。position首次占卡或存活时保留原优先级；只有position ledger已released才允许conditional首次O_EXCL claim。其他空卡继续走原baseline idle，未修改native/FOREVER科学源或队列job状态。
- 原baseline子phase继续通过原position.__file__入口，保持dynamic/native/FOREVER路由，不重入新的研究hook。原dispatch.main正常return后才以同一dispatch_old.lock和已捕获的pipeline.idle链收尾；原main异常时不另起分配过程。
- conditional controller活着时跨六phase保留同卡。只有单一终态proof、正确source/protocol、真实controller和所有已尝试worker均不存在、NVML无其PID才释放；失败还核last_attempted_worker。无终态或身份/NVML不确定时保留orphaned_hold，不重启、不发GPU信号。

## 来源与作者披露

本review作者曾编写本批adaptation.py、worker.py、controller.py，因此不把这三源自审称完全独立。verify.py由plain_r8_control编写，queue_adapter.py由research_review编写；本次对这两源是跨作者接口审阅。此前plain已独立审过前三源。本次未改任何科学/调度源，不新增准入层或模型门。

| 本次实际源 | SHA256 |
|---|---|
| `exp/conditional_shared_budget_20261026/v1/adaptation.py` | `251a2d490f9ce29069de141686a16da549c28de82f7c1ce58fab2df8da31252a` |
| `exp/conditional_shared_budget_20261026/v1/worker.py` | `912d5e08edb2642268a483f98f01da7440f7fed730e7b2e91d32609127e190c8` |
| `exp/conditional_shared_budget_20261026/v1/controller.py` | `2ac4aff2b01117fdea47f45014ffa0d6b312f7610c21dc197155b1dee01c001a` |
| `exp/conditional_shared_budget_20261026/v1/verify.py` | `837f209778f2cd413cd010ea3c9938977ce35d4aa9fad56a60bc1b498bc2f4da` |
| `exp/conditional_shared_budget_20261026/v1/queue_adapter.py` | `1eaea047c6b6399b922387a80fc47b00a90728c021e50e006d11ee6de1f7910c` |

PROTOCOL.md SHA256：`c45d44aba093fdb86073e1b936d72808102b7cfb8c5d3965d89607ec377d9309`。

原position wrapper固定SHA：`23a7a6fff3f33766a22d636c632476008560023c62da1b0c504f7b9913a9f833`；原dynamic wrapper固定SHA：`6c5006c6ece0df9550b30eabbe1949ac50a89df4ca7ff3ff0aec656bb4cdc690`。这些为来源绑定，不是新的live进程观察。
