# 训练质量校准与软聚合：固定专家池对照

四视图诊断完成：1/1；完整确认：0/2。缺失为待测。

| 范围 | 视图 | 生成题数 | 19任务AP（Answer F1） |
|---|---|---:|---:|
| pilot | original_hard | 155 | 31.025 |
| pilot | quality_hard | 155 | 36.704 |
| pilot | geometry_soft | 155 | 35.996 |
| pilot | quality_soft | 155 | 36.387 |

[全部领域、路由、融合强度与激活证据](VERIFIED_EVIDENCE.json)。原LoRA全部固定；只使用原池实际保留的TRAIN向量/QA做16参数质量校准，不补样、不使用测试答案，实际QA/前向数按证明记录。

- All four fixed-pool views; no TEST chooses calibration, router or full candidates.
- 16 new parameters, zero new LoRAs; inherited pool/projection/cache/temporary costs remain charged.
- A routing gain would identify an access/aggregation component, not prove knowledge is sufficient.
- Diagnostic500NLL/155generated differ from full21454; no full transfer metrics from final-only.
- Adaptation of published principles, not original MVP/D-MoLE reproduction or proven novel method.
