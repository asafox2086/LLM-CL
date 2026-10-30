# 当前恢复与存储交接（2026-10-30）

本文件优先于旧 PID 叙述；身份实核时间为 Unix 1793363867（OLD）与 1793362571（NEW）。不把旧锁、队列的 training 字段或 systemd 启动回执当成模型存活。

## 实际进程与恢复点

| 机器 / 卡 | 基线 | 最近真实身份与事实 | 当前已确认资产 |
|---|---|---|---|
| OLD / 0 | LLaVA 10% PAS | worker 1557042 / start803671912 / UID1084，真实 live | 原正式 RUN training_gpu0_20261030；未冒报完整终态 |
| OLD / 1 | LLaVA 10% SpaRTA | 已从原暂停14恢复；worker **1946673 / start806193331 / UID1084 / PGID1946673**，CPU1946636/start806193288，unit主包装1946635；实核原native training / completed14 / active15 | 原S14 2,011,680,367B / SHA8ba5a626336f4531539cd44efdfdf0bb5e8002ad0e17bb2c7abec9bcb6a662da；sameRUN原882源码 --resume，无新预检/重S0。原strictload与RNG恢复之后已进入15，并非额外GPU重载探针 |
| OLD / 2 | LLaVA 10% CL | worker1650061 / start804287271 / UID1084，真实 live | 原正式RUN training_gpu2_20261030；无旧CE暂停请求 |
| OLD / 3 | Qwen 10% PaRSP | worker1341844 / start802262974 / UID1084，真实 live | 最近独立资产检查已完整S2：19,548,056,668B / SHAef2028e0476b50359e2a386eb194ad79cb05a6ce262cdc9d6d884b70bf35591d。此为最后核过的资产，不据此断言之后没有新阶段 |
| NEW / 1 | LLaVA 1% SMo | **worker1401697 / start70894838 / UID1004 / PGID1401697**，实际原native training16 / completed15；原261406已gone | 原S15完整SHA166653b6f70783c4d05c11aca8dc23182f09f20b5e5e877db13b7fd00b145b6e为恢复origin；新branch recovery_enospc_20261030_S15_v4。原strict loader/RNG路径已实际通过，无新预检/重S0 |
| NEW / 2 | Qwen 1% PaRSP | 原969135已gone，不是live | 原S0初评无checkpoint。下一次只能明确记原W0 fresh分支，不能冒称checkpoint续训；原初评/故障原件保 |
| NEW / 3 | LLaVA 1% FullFT | **worker1411303 / start70936723 / UID1004 / PGID1411303**，父1401680；实际completed11 / evaluation12（1793366801） | 新branch formal_failure_recovery_S10_20261030由真实完整S10 SHA0bf45b5fd68e48004d94c0ef747395ec3435d4268e9c9f11d2110248c72af516恢复；original failed RUN/0B EXIT FAILURE/S11partial全保，旧969192/969180不live。643f adapter仅入口1个AST guard允许明确manual状态，逆变全原AST相等，原29科学源未改，不伪native pause/不重原offer |

NEW当前唯一增量CPU owner为 **1401680/start70894816/UID1004/PGID1401680**，systemd包装1401679，source8db06c9cf4c6b76ffeb500a742158b6103196296add6f56113c7bfd920187b86；持原七锁，已启动SMo与FullFT。仅固定三offer，后继完整admission最后原子发布后才由同owner的inotify接入；子终态用pidfd，不另起竞争allocator。NEW0保留。SMo真实启动证据6cc267e9baee2daf75fe6da02b168a9b4fe2500fe9c81fa34446acda371d6ce6，FullFT真实恢复证据6e7503bdcd500d037dca2fdb096000c7e2e59c904424617904af4b86e0ceac11。

NEW原CPU owner969129同样已gone；llmcl-two-backbone-native.service实核MainPID0 / ExecMainStatus1 / failed。两canonical队列JSON完整但旧training/PID字段已陈旧，不得直接称正在运行。证据：NEW `summary/storage_recovery_20261030/NEW_ENOSPC_REAL_IDENTITIES_AND_QUEUE_AFTER_RETIREMENT_20261030.json`，SHA b8dceed034a2eaca60ddb8af9b87bdc8b4ac300302f9c5220c4e519016ef2699。

## 本次已闭合事件

- prefix path_compat_v1两个原GPU worker均EXIT0，parent1911209与两worker1911236/1922218全部gone，原unit实际EXIT0/inactive，SLOT complete/released。原控制终态闭合 `summary/native_prefix_function_observation_20261030/path_compat_v1/TERMINAL_CLOSED_FOR_BASELINE_RESUME_20261030.json` SHA976261c80c5d66bdd72cb90364eed2456c05d4f190514fa0541f5c17689c437c。唯一kernel事件流72884已真实终态，禁止重复建watcher或由它再启动模型。原premodel失败完整保留；科学readout由observation_review唯一执行，本交接未评分/解释其数值。
- SpaRTA原native暂停实际S14、GPU EXIT0，旧S19-only CPU收尾实际EXIT1；trace无权限，原因只作源路径推断，不冒称亲见trace。39原小记录已NEW耐久，manifest f28961ae508fac1c0c00b4c05b46812d00c77aeb1cd3829c9ba000c3d07d6bbd。旧1408250/1408233均gone，已被上表新恢复身份取代。
- SpaRTA恢复源38d6efd6281b73fa1ec630ce033982b78adc1ac5907408d1bb4b3ae997ded6da已root/literature全文接受；原同RUN --resume14→15，原25源/数据/生成/optimizer阶段生命周期未变。原请求已进pause_history，SHA4841caa3b0d3d0453c2d1710c6a41868fadf66da4df0d58da710a7153e69daec。新unit `llmcl-sparta-resume-native-prefix-20261030.service`，原旧no_retry控制器未重放。

## 已实际退休；禁止重放

所有下列操作都仅去非最终历史payload；全部原metadata、QA、预测、分数、source、主表记录及指定当前/最终/研究锚保留。

| 操作 | 实际退休字节 | 原收据SHA / 耐久位置 |
|---|---:|---|
| OLD SpaRTA S02..10九精确路径 | 18,096,734,643 | aaad11290387fee4cd0f3dc91e34f277bca9ece604c7e7ac6a4afa24e2b883da；NEW OLD_SPARTA_S02_S10_ORIGINAL_DURABLE_20261030/manifest795e5f8352a31572f0d15c8a6911447b970fcb503cc9ac98e49441be24b32197 |
| OLD旧legacy Seq S11..13三路径（重新以1%/10%维护准入，不是100%） | 13,289,386,530 | ebecc3c6f06dd55759684f5db8c000cca4a1bf99ce54127cdb3171136f713e53；NEW OLD_LEGACY_SEQ_S11_S13_PRIMARY_ORIGINAL_DURABLE_20261030/manifest9fb713a8c42c1e0a56edb60eb5c0e14164817f0fe540a71e250deac22ea93384。此前100%计划下未执行的历史记录不改 |
| NEW弃用legacy PaRSP诊断CP + 完成QwenSeq S16双link | 10,770,264,032 | c9a2dd61203acf60e57ec5194c0f62515ee7519b2cb923a92078622dd80fb943；NEW_LEGACY_GATE_AND_SEQ16_RETIREMENT_ONCE_20261030 |
| NEW FullFT正式S01..08八历史CP（紧急释放已满物理盘） | 28,656,628,192 | OLD先耐久STARTED3a7b87ffd9052dc4937f7063b071e163bfb119ed7c2737dff1ee8e5256cea933、RECEIPT2a64d055ebc59582efa0840879dd7b4e03ec581c0963a46af9884c58ed131b76；全部tmpfs原件已镜像NEW NEW_FULLFT_S01_S08_EMERGENCY_ORIGINALS_20261030/DURABLE_MANIFEST.json，SHA63479c1ea285d76578d1864bbd7bfca290fd12fae16df5d67824f56cffe47010。S10全SHA前后相同，9/10/11partial及0B终态均保 |

以上耐久位置均相对相应服务器ROOT的 `summary/storage_recovery_20261030`。更早V7、Seq01..15、旧legacy02..10等已有退休同样不可重放；本表不是允许重复执行的列表。

## 正在做与尚未做

- OLD Qwen S1-only退休已唯一完整成功：仅一个S1非final payload 15,411,559,868B实际gone；S2完整SHA前后ef2028e0476b50359e2a386eb194ad79cb05a6ce262cdc9d6d884b70bf35591d、所有其它CP与固定完成前缀不变，active/partial未触。原source97b916、同O STARTED/RECEIPT全保；原RECEIPT SHA1a2b417c3fbbdb416df7e2e8c25a7bc6502e3307c5358ca7f4e579bacf66e5b7。六原小件已NEW耐久 `OLD_QWEN_S01_AFTER_S02_ORIGINAL_DURABLE_20261030/DURABLE_MANIFEST.json` SHA30a3a1032159e004fddbf4fce71bda56ecb676e086dd47ce427f9833af913807。禁止重放；不把已retire的S1继续声称本地保留。
- NEW已完成QwenSeq S17/S18历史payload已按原人类许可直接退休：四精确links/两inode，共9,125,998,904B，RECEIPT `NEW_QWEN_SEQ_S17_S18_RETIREMENT_ONCE_20261030/RECEIPT.json` SHAd27fce82caeaff9a07f4e36cf2eefb369d5965a485663c8ff7baa8c734d84104；S19实际全SHA前后d3df...及所有quality/lineage/provenance/QA/65proof原件相同。此前要求额外OLD备份是actor过度保守的保留策略，已明确撤销；**不是归档、没有另份备份，不再声称17/18仍本地**。原qualityproof只依S19，17/18历史lineage已一次性完成，不重放。原archive默认预算失败eea030...与0copy事实保留，不是降门后重跑archive。
- NEW只读readout源曾因ENOSPC写成0B，原错误须保；不能以空文件AST通过当源有效。源修复与唯一readout归observation_review负责。
- NEW恢复实际已有两臂：SMo15→16和FullFT10→11均已严格加载并进入原科学循环，见上表。Qwen PaRSP无CP，仍待原W0 fresh分支（原初评完整T01..15与合法partial只按manifest-ID前缀兼容），未冒称checkpoint续训。其准备源597e41...正在审查/原raw UTF8前置核查，未发布PaRSP offer、未启动Qwen新worker。不得重activate_once、旧no_retry offer或为恢复改变训练/数据/生成设置。
- 后续基线需用单一kernel阶段提交事件承接checkpoint增长，及时按原授权手工退已替代非final；保current、真正恢复origin锚、最终、全部小记录。不再把一张35GB瞬时free当成全程reserve，不能再次让FullFT历史CP累计10份填满盘。

## 容量口径与不变范围

紧急退休后NEW physical当次为28,656,521,216B，账号quota当次688134676/1073741824 KiB；二者不可互换，也不是长期承诺。OLD S2已19.548GB、S1已15.412GB驻盘，预算不能把已占用完整/partial再当全额未来需求重复扣。大Qwen preflight只有局部fakehistory，不是正式CP或部署内存上界。

PaRSP CE/geometry/保护预算研究全关闭，旧借卡请求撤回；100%仅可选准备、未启动。1%/10%设置、原方法部署形态冻结；不替LoRA/OSFT做merge/物化优化。README/AGENTS与主表只由root发布，本文件不增加科研结论或新质量数值。

本次17/18退休后physical值仅见原receipt；后续FullFT已开始写新阶段，空间必须按事件与实际allocated滚动维护，不能用此历史值重复承诺。OLD Qwen S2→S3单路径退休237df750...已双审并由唯一actor启动，当前尚待原RECEIPT终态，禁止再起同O；它不会删除S3/current/partial/研究18/final19。
