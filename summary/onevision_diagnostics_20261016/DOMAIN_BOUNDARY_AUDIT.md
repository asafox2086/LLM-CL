# 19阶段：同域后续学习与跨域后续学习的逐题拆分

读取已完成确定性运行，核验预测文件与完成证明SHA、原始ID/答案、相同目标token数。不使用在跑的全量预测。

**这是已有轨迹的事后描述，不是匹配任务数/预算的领域因果实验。** 两锚六机制GPU复核仍在等待。500题NLL与155题生成不是完整测试。

| 领域 | NLL题数 | 后续同域NLL变化 | 后续跨域NLL变化 | 生成题数 | 刚学完→最终正确数 | 总丢失/新增 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| natural | 100 | +0.0000 | -0.0242 | 10 | 5→4 | 1/0 |
| medical | 100 | -0.3673 | +0.3193 | 73 | 20→25 | 3/8 |
| geography | 100 | +0.0180 | +0.0107 | 32 | 19→15 | 5/1 |
| finance | 100 | +0.0000 | +0.0319 | 10 | 3→3 | 0/0 |
| traffic | 100 | -0.0086 | +0.0000 | 30 | 0→0 | 0/0 |

“刚学会时”组合不同任务各自刚完成时的模型输出，不是单一快照；“域结束→最终”才是两固定权重快照。不能将同域与跨域的差额解释为纯领域效应。

NLL为后减前，负值表示原答案概率改善。完整19任务的当前学习、历史起点差异、后续同域/跨域NLL及95%区间见JSON。中间阶段只生成当时任务，域结束点没有全部固定生成ID，因此对应EM/转移保持null；上表只报告完整可测的“刚学完→最终”转移。原运行也没有测“首个跨域阶段的旧域”，保持null。不选取可用交集来冒充原固定队列，不重新解析答案。

## 19个原生任务

| 任务 | TRAIN | 生成QA | 刚完成→域结束→最终正确数 | 同域NLL变化 | 跨域NLL变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| T01 natural | 347 | 10 | 5→5→4 | +0.0000 | -0.0242 |
| T02 medical | 11 | 10 | 3→None→8 | -0.8573 | +0.0525 |
| T03 medical | 17 | 10 | 2→None→1 | -0.7432 | +0.2906 |
| T04 medical | 26 | 10 | 3→None→4 | -0.4135 | +0.7403 |
| T05 medical | 6 | 6 | 4→None→3 | -0.3076 | +0.3075 |
| T06 medical | 14 | 10 | 3→None→4 | -0.1780 | +0.1934 |
| T07 medical | 4 | 5 | 2→None→2 | -0.0343 | -0.1076 |
| T08 medical | 5 | 6 | 0→None→0 | -0.0618 | +0.0913 |
| T09 medical | 1 | 2 | 0→None→0 | -0.0740 | +0.3407 |
| T10 medical | 3 | 4 | 2→None→2 | -0.0323 | +0.5508 |
| T11 medical | 12 | 10 | 1→1→1 | +0.0000 | +0.1612 |
| T12 geography | 6 | 2 | 2→None→2 | +0.0112 | +0.0289 |
| T13 geography | 170 | 10 | 9→None→5 | +0.0634 | +0.0010 |
| T14 geography | 171 | 10 | 3→None→3 | -0.0023 | +0.0302 |
| T15 geography | 226 | 10 | 5→5→5 | +0.0000 | +0.0028 |
| T16 finance | 89 | 10 | 3→3→3 | +0.0000 | +0.0319 |
| T17 traffic | 49 | 10 | 0→0→0 | -0.0155 | +0.0000 |
| T18 traffic | 107 | 10 | 0→0→0 | -0.0093 | +0.0000 |
| T19 traffic | 49 | 10 | 0→0→0 | +0.0000 | +0.0000 |

## 局限

- Retrospective already-inspected 500NLL/155generation QA, not full21454 or an untouched confirmatory cohort.
- Chronological within/cross changes differ in number of later stages, exposure, content and order; no isolated abstract domain-count causality.
- Earlier independent vs sequential models have different prior training costs and initial functions; history benefit is not free.
- Loss uses original gold tokens; low NLL need not imply decoded accuracy. Generated subset and primary scoring remain unchanged.
- One seed/backbone/native diagnostic image protocol; not formal shared V2 model ranking.
- Paired image-cluster bootstrap has 2000 draws seed42, uncorrected across all contrasts.
- First cross-domain old-task cells are unmeasured in the original run and remain null; do not reconstruct them from another weight state.
- Intermediate generation covers only current native task. Domain-end EM/transitions for the full predeclared cohort are null unless all fixed IDs were actually generated; do not select an intersection subset.
- Two-anchor six-mechanism deterministic GPU study remains the required matched control, not replaced by this audit.
