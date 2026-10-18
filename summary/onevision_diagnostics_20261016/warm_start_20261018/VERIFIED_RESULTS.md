# 初始化单变量控制：封存结果

新增温启动训练完成：1/1；温启动完整确认：0/1；原冷启动完整确认：0/1。缺失是待测。

| 范围 | 方法 | 自然 | 医学 | 遥感 | 金融 | 交通 | AP | FWT | BWT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 155诊断 | learned_cosine_K3 | 58.571 | 15.841 | 46.875 | 20.000 | 31.421 | 31.025 | — | — |
| 155诊断 | warm_inherit_K3_v1 | 58.571 | 25.339 | 46.875 | 20.000 | 31.421 | 35.025 | — | — |

只接受完整SHA证明，逐题核对冻结原ID/标注并复算原始Answer F1。冷启动既有诊断不是新增温启动结果；未完成不能写成0分或收益。

[协议与六栏](PROTOCOL.md) · [全部证明、资源与实际更新](VERIFIED_EVIDENCE.json)

- 155 diagnostic generated QA are separate from complete21454; no fullFWT/BWT from final-only.
- Only new-expert initialization changes; geometry/allocation must match cold runtime gates.
- A result here does not establish initialization x unique-data-size interaction or knowledge sufficiency.
- CPU/GPU resident, optimizer, temporary clones/history/probes/metric and compute remain charged.
- Outside diagnostic500 is not claimed untouched by every earlier study.
