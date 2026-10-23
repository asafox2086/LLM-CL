# 独立v2源码已审，只允许唯一CPU门

v1 CPU门在构造期因视觉SDPA不支持而失败，原源/review/marker/log保留，不能重跑v1。v2仅固定三处统一使用语言SDPA、视觉eager并核实际子配置，三科学条件/数据/数值dtype/预算/出口不变。实际attention路径变化公开，不称逐值相同或速度收益。

复用12份已核资产，不再次下载；没有真实模型STARTED。独立SOURCE_REVIEW仅接受v2唯一CPU门，尚无门/READY/激活；READY后仍需独立准入才能激活。再失败不滚动v3或换backend凑通过。
