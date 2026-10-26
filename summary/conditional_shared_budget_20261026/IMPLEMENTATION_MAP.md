# 最小实施映射（未执行）

以本目录PROTOCOL.md的shared/uniform/soft为准；最初Hard/原型提议不实现。只新增`exp/conditional_shared_budget_20261026/v1`文件，不改旧runner、任何在跑任务或正式1%/10%队列。

## 可直接复用的路径

| 功能 | 已有来源／接口 | 新代码的最少差异 |
|---|---|---|
| 原1% TRAIN | `exp/osft_increment_screen_20261019/v1/study.py:original_train()` | 核原manifest/19task/1313；只按冻结SPLIT的fit_ids/dev_ids取1213/100，不再重建分组或全图哈希 |
| 冻结划分 | `summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json` | 保留其原SHA和完整分母；只对实际模型输入按已有image_sha256核图片 |
| r8初始化 | `exp/plain_r8_gpu_control_20261025/v1/control_adaptation.py:make_plain` | 新`adaptation.make_method`调用原函数后仅绑定forward；保原48模块、96个A/B对象、540672坐标、原A字节/B0、scale4、dropout.1/RNG |
| 条件规则 | 新`adaptation.py` | shared原SeqLinear；uniform一次A响应的后4维乘.5；soft用同一post-dropout A响应两r2能量FP32 softmax(tau1)，正常链式反传；无新参数/encoder/原型 |
| 训练段 | `study.training_ast()`＋plain-r8 `worker.py:train` | 原训练prefix不改；仅在每stage prefix前读before、其后保存adapter并读acquired。GPU AdamW、87更新、1213次原current backward保持 |
| GPU输入 | plain-r8 `common.py:Batcher`及其GPU-only视觉prepare | 只复用Batcher类，不实例化旧Context；新worker有自己的root/run/source账目，无旧READY、旧CP或旧point生命周期 |
| 生成 | plain-r8 `readouts.py:_batch/_generate/evaluate_dev/resource_windows` | before/acquired按task原DEV逐条调用前两函数；末态沿后两函数5窗后100DEV；记录phase字段由新worker显式写，不假索producer字段 |
| 完整导出/重建 | plain-r8 `control_deployment.py`的meta/alias-aware transfer/buffer/GenerationConfig逻辑 | 不能直接调用旧receipt/load_complete：它拒绝A/B且要求全FP16。新worker独立小封装完整core state/config/mode/所有buffers；meta上`adaptation.install_meta`后strict assign；保FP32 RoPE、tie及bridge别名 |
| 最终表示 | 新`adaptation.deploy` | shared/uniform GPU FP32 merge→FP16；soft保显式r8。实际最终图严格重载后才测终态，不能把fold图文件配未fold质量 |
| 阶段读出 | plain-r8训练保存接口＋`exp/acquisition_readout_20261025/v1/worker.py`的逐task读数结构 | 直接在当前训练图前后读，不另建22个模型、不保存19份全CP。保存/恢复Python/NumPy/Torch CPU＋CUDA RNG，恢复训练mode及requires_grad原mask，不改变dropout轨迹 |

## 新独立CLI与最少文件

`worker.py --root ABS --run ABS --arm shared|uniform|soft --gpu N --phase train|eval`

实际最少依赖：root包含原1%port/study、冻结SPLIT、原plain-r8五个被复用源码、position-coordinate BudgetQV、storage helper与原OneVision资产；run是新建的各臂私有目录。新worker不实例化旧Context、不读取旧READY/旧CP。它记录自身与实际依赖源SHA，硬核manifest/SPLIT/training-AST/storage原SHA；eval只读本臂新train_COMPLETE及新complete.pt。`--root`必须指上述1%资产视图，不能拿10%manifest替换。

- `adaptation.py`：上述初始化/模式/meta安装/最终转换；模块导入时无Torch/模型计算。接口为`make_method(model,asset,storage,arm,inherited_source)`、`install_meta(model,arm,deployed=...,plan=...)`、`metadata`与`deploy`。
- `worker.py`：自有数据绑定、训练及stage读出、完整包导出/重建、独立eval入口。只依赖原plain-r8 Batch/readout/初始化和原study/port/storage源码，不读旧CP和旧质量。
- root另负责固定协议、外层一次监督及评分消费；不在本任务另造service、allocator或多级准入系统。

train首次启动预算5400秒：fresh W0，19stage的before共100＋acquired共100，1213FIT/87更新，19个小A/B快照，最后转换并保存一个完整部署包。无独立初态100，不读TEST，不训练DEV。训练前/后的读出中，task无DEV则保留空项；before→acquired是该固定stage整包功能变化。RNG只操作本worker所用CUDA设备，不为读数遍历/初始化他卡。

eval是另一个全新进程、首次1800秒：从新完整包meta strict reload，先5个固定hash-min ID各32新token，再100最终DEV；前后state/buffer不变。每臂305生成/300待评分，三臂915生成/900待评分、3639FIT backward/261 optimizer更新；实际phase时间、训练监控与推理窗口分列。评分不在worker里执行。

保存的是完整core权重＋mode＋base config＋generation public config＋全部named_buffers（含非持久buffer），processor配置/词表作为同目录实际文件绑定。shared/uniform实际部署无A/B，soft有全48组A/B；重建meta时只初始化结构，不随机构造完整CUDA图，不读取原基座权重补缺。存储、所有别名/FP32派生buffer及输出图格式按实际核，不借旧pure-FP16 receipt假定成立。

## 空卡队列：接口可复用，当前没有接入

现有`exp/finance_position_diagnostic_20261026/queue_adapter.py`展示在原`dispatch_dynamic.prepare().pipe.idle`中保留一次研究slot的可行接口，且phase子入口保持原baseline路由、controller存活期间保留同卡。它硬绑定已有position四phase/ORDER/admission/终态schema，**不能把新三臂直接塞进原slot或改旧文件**。

最短下一步是root在决定实施后：使用本独立CLI，准备一个串行六phase的有限controller；等待现有已预留研究任务终态，再由原分配器的下一真正空闲机会启动一次，并在全部六phase间保留同卡。若需接coordinator，只能复制该slot wrapper到新目录并换本批明确phase列表/来源，在安全CPU交接后接入；保全所有live GPU身份、原native/FOREVER队列。这里不创建/部署该wrapper，不承诺当前已有空卡，不另开扫描GPU的服务，也不因GPU忙停止源码准备。

## 当前已知必须保留的边界

- 训练中新增读出会消耗RNG：必须恢复，不仅`model.train()`。
- 原pure-FP16 deployment不支持soft小银行：必须改新封装而非放宽旧assert。
- 训练冻参摘要只核W0和原buffer；本首轮没有原型/动态buffer，不能继承Hard设计额外状态。
- uniform/private系数和soft都使用一次实际dropout后A响应，非原PAS未dropout评分的逐字复现。
- soft/uniform是条件功能主对照，shared只提供普通共享锚；部署fold舍入是公开整包差异。

本文仅实施定位。0模型、0新QA读取、0评分、0服务器访问；新代码即使静态通过也不等于已执行或取得质量结果。
