# 保留源参数、抵消初始旧输出：完整507题控制

| 条件 | 最终507 Answer F1 | 最终507 NLL |
| --- | ---: | ---: |
| cold | 14.681 | 7.1256 |
| within_full | 18.113 | 5.4776 |
| cross_full | 18.468 | 5.6855 |
| cold_compensated | 14.681 | 7.1256 |
| within_compensated | — | — |
| cross_compensated | — | — |

三补偿条件必须全部测，不用测试挑选赢家。初始32精确函数门槛、真实梯度／更新门槛、冷组两步和507逐项等价均绑定于各完成证明。尚待项不是零收益。

[预定协议](PROTOCOL.md) · [所有配对／SHA／边界](VERIFIED_EVIDENCE.json)
