# 冷启动继承与功能质量调用：四格结果

完整格：0/4；温启动19阶段：1/1。缺失为待测，不是0分。

## full21454

| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏平均 | AP | FWT | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 冷启动＋原几何 | — | — | — | — | — | — | — | — | — |
| 10 完整继承＋原几何 | — | — | — | — | — | — | — | — | — |
| 01 冷启动＋质量硬调用 | — | — | — | — | — | — | — | — | — |
| 11 完整继承＋质量硬调用 | — | — | — | — | — | — | — | — | — |

## diagnostic155

| 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | 域宏平均 | AP | FWT | BWT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 冷启动＋原几何 | 58.571 | 15.841 | 46.875 | 20.000 | 31.421 | 34.542 | 31.025 | — | — |
| 10 完整继承＋原几何 | 58.571 | 25.339 | 46.875 | 20.000 | 31.421 | 36.441 | 35.025 | — | — |
| 01 冷启动＋质量硬调用 | 60.714 | 30.282 | 46.875 | 30.000 | 31.421 | 39.859 | 36.704 | — | — |
| 11 完整继承＋质量硬调用 | — | — | — | — | — | — | — | — | — |

完整与诊断分表；全部原始ID/标注/Answer F1逐题核验后才填格，不拼接不同协议。

[两问题、模块、判据及成本](PROBLEM_AND_METHOD.md) · [全部证明与资源](FOUR_CELL_EVIDENCE.json)

- Local inherited-control evidence does not prove all-method or knowledge-only causes.
- 155 diagnostic generated answers are not complete21454.
- No fullFWT/BWT from final-only. Missing rows remain pending, not zero.
- Both quality cells charge43 historical TRAIN,172NLL forwards and16parameter CPU fit.
- Interaction is a descriptive four-cell contrast, not multi-seed significance.
- No cold/warm gains are added together to invent combined performance.
