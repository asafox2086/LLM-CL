# 末态温度与加载路径：文字纠正，不重跑

独立源审先前将stage起点tau=1.5直接用于S19公式而写/192，不严谨，已撤回。一般两active expert式是g1=sigmoid((||A1x||²−||A2x||²)/(128*tau))。

原router_function_moments_20261030_v2 observer run.py99–104加载原method-graph-v1完整method/model对象；checkpoints.py35–39整体cloudpickle，51–64直接取回，仅改IO路径并恢复RNG。该路径未重建PAS、未begin_stage、未改tau；普通模块Python属性随对象保留，未发现重建默认温度的路径。READOUT和既存小metadata没有实际gate_tau数值，不能把source schedule末步应到0.7的推断当实测。

本次只看源码和既存小metadata，没有读取CP载荷或新模型。原M2/READOUT与新只读因子分解均不因这次文字纠错重跑。若后续查成因，需要直接记录实际路由能量与温度，而非由源码均值/默认参数给出原因。
