# LLMCL：1%与10%数据下的多领域多模态持续学习

**研究目标：完整模型权重与 GPU 内存显著更小，同时保持有竞争力的总体 Answer F1。** 人类最新允许比最高分略低，按质量—内存取舍判断；允许域间取舍，当前五域分别报告，重点是论文方法与实测资源优势，不做端侧设备适配。

**部署参数优先级（2026-10-29）：少的是最终新增部署参数，不是训练更新空间。** 目标是总部署参数接近单骨干、仅增加很少参数，并实测完整权重内存和GPU推理峰值；训练可以更新全参数或使用更大的临时空间，训练坐标只作补充。无需比同骨干全量微调更小。朴素基线采用不加LoRA的全量顺序微调，旧SeqFT-LoRA保留为小增量参照。FOREVER等低新增方法与可合并普通LoRA必须公平比较，不能仅凭新增0参数宣称优势。[比较口径与全量基线要求](summary/deployment_total_parameters_20261028/REQUIREMENTS.md)

**方法方向：抽象强方法的有效功能，用简洁机制实现，或从已验证的共同问题设计新方法。** 不局限于改造某个现有算法，不把折叠、普通扩容或换基线本身当论文贡献。新假说必须给出功能理由及实验对照，最终按原F1和完整资源实测判断。

**当前共同问题：训练后留下的附加结构，哪些是端侧推理真正需要的？** 区分训练辅助状态、输入无关表示与输入条件功能；PAS只作首个探针，CL-MoE检验另一种条件机制，SpaRTA考察保留动态功能时的充分表示。OSFT／普通LoRA的静态物化与FOREVER低新增作为必要参照，不预设所有方法都能压缩。[八法前向与成本审计](summary/deployment_total_parameters_20261028/CROSS_METHOD_FUNCTION_AUDIT_20261029.md)

**已获得的模型事实：专家参数更多，输入相关的局部功能变化未必更大。** PAS／CL新增神经权重140,771,328／37,158,912，相对各层固定全局平均gate混合的局部差值平方比为0.010457%／2.952336%，五域排序一致。测量范围为同80条历史TRAIN、168投影；不是F1损失或可删参数比例。[外层定量记录](#探索命题与结论) · [完整功能证据](summary/router_function_moments_20261030_v2/FINDINGS.md)

**研究重点（人类最新指示）：只考虑新习得，不研究遗忘。** 后续问题是现有方法为什么没有学好新任务、如何使1%与10%数据下的学习更有效；内存／参数量作为最终评估，不替代学习原因证据。

**当前主实验：恢复五域（含交通）、19个子任务。1%为1313 TRAIN，10%为13105 TRAIN，均为完整21454 TEST。TaskGradReuse已关闭，不再追加训练；四域负结果完整保留。** [恢复决定与候选门槛](summary/five_domain_restored_20261029/DECISION.md)

**设置确定后固定。** 沿用既定数据、训练、生成与评分条件，研究方法在这些条件下的新习得能力、失败和适用边界，不为改善结果反复调整设置。

**最新完整部署结果：PAS-static五域宏46.332，原PAS为46.620，低0.288pp；同卡固定五窗GPU最高2166.101MiB，原PAS2959.415MiB，权重少537.001MiB。** 已通过原完整核验，新增部署参数0；作为1%主表中的末态部署行单独报告。现有基线完整结果为1%九组、10%三组（双骨干SeqFT及LLaVA PaRSP），静态表示不计作新增训练基线。当前只证明这个检查点的质量—资源取舍，跨方法完整确认及FullFT参照仍缺，论文目标尚未完成。[完整结论与证明](summary/pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)

**已关闭候选：TaskGradReuse四域最终宏F1为44.127；省内存未弥补质量不足。后续先过FOREVER等低开销方法的质量门槛。** [实验状态](summary/task_gradient_four_domain_final_20261029/STATUS.md)

本文只保留主表、探索命题和结论。[补充说明](SUPPLEMENTARY_NOTES.md#探索过程与证据索引)保存完整探索过程、正负证据、故障、成本及运行交接。

<!-- FIVE_DOMAIN_RESTORED_MAIN_BEGIN -->
# 1%训练：五域主表

## 表1：五域／19小类最终Answer F1

原19任务完整训练终点S19；所有已填值沿已核验原始结果，新增交通不重采样。Ours旧S16候选已关闭，另存负结果。

<table>
<thead>
<tr><th rowspan="2">方法</th><th>自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th>金融</th><th colspan="4">交通</th><th rowspan="2">五域宏 F1 ↑</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th><th>T17</th><th>T18</th><th>T19</th><th>Avg</th></tr>
</thead><tbody>
<tr><th colspan="24">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>75.632</td><td>60.747</td><td>43.366</td><td>37.445</td><td>52.339</td><td>49.770</td><td>43.820</td><td>27.148</td><td>5.682</td><td>42.615</td><td>16.455</td><td>40.758</td><td>65.000</td><td>48.122</td><td>17.509</td><td>66.717</td><td>46.711</td><td>15.891</td><td>33.587</td><td>30.842</td><td>32.750</td><td>31.938</td><td>42.186</td></tr>
<tr><th>FOREVER</th><td>64.486</td><td>66.199</td><td>32.646</td><td>51.645</td><td>49.597</td><td>53.223</td><td>43.820</td><td>61.022</td><td>14.881</td><td>72.821</td><td>16.579</td><td>46.103</td><td>78.000</td><td>75.770</td><td>4.140</td><td>65.467</td><td>50.570</td><td>9.623</td><td>30.016</td><td>33.662</td><td>36.067</td><td>33.370</td><td>40.830</td></tr>
<tr><th>CL-MoE</th><td>74.657</td><td>62.026</td><td>42.636</td><td>34.270</td><td>44.892</td><td>52.726</td><td>47.191</td><td>27.443</td><td>14.015</td><td>49.077</td><td>22.227</td><td>41.139</td><td>73.000</td><td>57.597</td><td>18.256</td><td>67.316</td><td>50.050</td><td>14.914</td><td>34.330</td><td>31.592</td><td>35.297</td><td>33.110</td><td>42.774</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>70.697</td><td>75.000</td><td>41.147</td><td>33.413</td><td>48.522</td><td>49.802</td><td>51.685</td><td>41.523</td><td>21.111</td><td>82.667</td><td>24.492</td><td>44.202</td><td>44.000</td><td>77.056</td><td>20.699</td><td>66.092</td><td>55.738</td><td>25.223</td><td>34.455</td><td>39.755</td><td>34.379</td><td>37.241</td><td>46.620</td></tr>
<tr><th>PAS-static（末态部署）</th><td>70.661</td><td>73.684</td><td>41.733</td><td>32.868</td><td>46.102</td><td>50.799</td><td>51.685</td><td>46.408</td><td>19.563</td><td>79.590</td><td>24.093</td><td>44.152</td><td>44.000</td><td>72.014</td><td>20.597</td><td>66.092</td><td>54.218</td><td>25.281</td><td>34.496</td><td>39.896</td><td>34.475</td><td>37.347</td><td>46.332</td></tr>
<tr><th>PaRSP</th><td>73.346</td><td>58.904</td><td>37.123</td><td>40.064</td><td>50.786</td><td>42.467</td><td>40.449</td><td>22.730</td><td>15.530</td><td>46.615</td><td>16.883</td><td>38.743</td><td>44.000</td><td>69.983</td><td>21.751</td><td>66.417</td><td>54.088</td><td>20.609</td><td>30.975</td><td>31.218</td><td>37.562</td><td>32.655</td><td>43.888</td></tr>
<tr><th>OSFT</th><td>73.907</td><td>66.096</td><td>38.700</td><td>35.723</td><td>56.989</td><td>53.277</td><td>53.933</td><td>33.921</td><td>45.833</td><td>71.385</td><td>24.941</td><td>43.894</td><td>44.000</td><td>60.880</td><td>23.278</td><td>66.067</td><td>51.709</td><td>23.982</td><td>36.765</td><td>36.774</td><td>34.326</td><td>36.196</td><td>45.937</td></tr>
<tr><th>SpaRTA</th><td>73.596</td><td>60.395</td><td>42.465</td><td>35.552</td><td>51.505</td><td>45.899</td><td>46.067</td><td>26.868</td><td>28.301</td><td>62.410</td><td>21.327</td><td>40.921</td><td>44.000</td><td>39.120</td><td>24.601</td><td>66.742</td><td>45.942</td><td>21.328</td><td>36.591</td><td>33.708</td><td>34.271</td><td>34.520</td><td>43.261</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="24">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>86.312</td><td>72.061</td><td>45.827</td><td>41.170</td><td>56.989</td><td>53.606</td><td>56.180</td><td>42.002</td><td>18.056</td><td>62.923</td><td>27.650</td><td>47.654</td><td>72.000</td><td>65.076</td><td>14.116</td><td>67.516</td><td>51.110</td><td>26.576</td><td>37.849</td><td>32.617</td><td>29.905</td><td>33.210</td><td>48.972</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>85.821</td><td>72.003</td><td>46.777</td><td>42.709</td><td>54.839</td><td>51.463</td><td>50.562</td><td>41.523</td><td>18.056</td><td>62.154</td><td>26.260</td><td>47.279</td><td>75.000</td><td>57.597</td><td>18.324</td><td>68.791</td><td>50.680</td><td>23.557</td><td>37.125</td><td>32.358</td><td>30.414</td><td>33.023</td><td>48.072</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

域内Avg按TEST题数加权；五域宏对五个域等权。医学10类、遥感4类、交通3类，自然和金融各1类。T17–T19为原交通任务顺序，原名称及数据哈希见冻结manifest。—为未完成／未核验，非0分。

## 表2：持续学习与部署开销

| 方法 | AP19 ↑ | FWT ↑ | BWT ↑ | 完整保存系数 M ↓ | 新增权重参数（保存态）M ↓ | 权重 MiB ↓ | 权重增量 MiB ↓ | GPU五窗峰 MiB ↓ | TTFT s ↓ | tok/s ↑ | 适配器数 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LLaVA-OneVision-Qwen2-0.5B-OV** | | | | | | | | | | | |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8） | 40.286 | 14.783 | 1.775 | 894.216 | 0.541 | 1706.613 | 2.062 | — | — | — | — |
| FOREVER | 45.245 | 17.748 | 5.642 | 894.216 | 0.541 | 1706.613 | 2.062 | — | — | — | — |
| CL-MoE | 42.288 | 19.039 | 0.012 | 930.877 | 37.159 | 1846.463 | 141.912 | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | 46.406 | 14.726 | -1.732 | 1034.447 | 140.771 | 2241.552 | 537.001 | 2959.415† | — | — | — |
| PAS-static（末态部署） | 46.002 | — | — | 893.676 | 0.000 | 1704.551 | 0.000 | 2166.101† | — | — | 0 |
| PaRSP | 40.390 | 18.503 | -0.484 | 893.676 | 0.000 | 2273.345 | 568.794 | — | — | — | — |
| OSFT | 46.357 | 18.932 | -1.616 | 990.914 | 97.238 | 2757.984 | 1053.434 | — | — | — | — |
| SpaRTA | 41.829 | 17.145 | -2.992 | 904.077 | 10.402 | 1744.231 | 39.680 | — | — | — | — |
| Ours（待新候选） | — | — | — | — | — | — | — | — | — | — | — |
| **Qwen2-VL-2B-Instruct** | | | | | | | | | | | |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8） | 47.812 | 13.258 | 4.415 | 2210.075 | 1.090 | 4217.462 | 4.156 | 4954.604† | — | — | — |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | 47.123 | 14.023 | 0.075 | 2286.975 | 77.930 | 4510.812 | 297.506 | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours（待新候选） | — | — | — | — | — | — | — | — | — | — | — |

PAS-static为原1% PAS检查点的固定末态部署表示：80条历史TRAIN及答案校准后，系数冻结，独立完整重载并评测21454 TEST；不是一次新的持续训练。19小类／五域宏及AP19均来自其自身完整评估，FWT/BWT未测。总部署893,675,552神经参数、新增0，权重1704.551MiB；普通buffer另9032B。GPU为同卡同五窗各32新token的最大allocated，不是全TEST峰或速度。成本包含原PAS训练及校准/export；FOREVER和可合并LoRA／OSFT仍是公平参照。[完整核验、资源与边界](summary/pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)。

资源沿原S19保存态、原dtype及原固定五窗实测。完整保存系数包含必要持久状态，不等同训练坐标；原基座可重建缓存排除。新增权重参数含推理必需U/S/V基，排除mask与CL历史anchor/prior；这些非参数状态仍计入完整必要系数和字节。普通LoRA／OSFT可物化为原dense形状的0新增，保存态不是最优部署下界。[参数证据](summary/deployment_total_parameters_20261028/ADDED_MODEL_WEIGHT_PARAMETERS_20261030.json)。MiB=2²⁰字节，M=10⁶。FOREVER的权重增量约2.062MiB，属于必须比较的低成本强参照。Ours旧四域S16实测另存，不填本S19主表。

<details>
<summary>10%训练：五域主表</summary>

# 10%训练：五域主表

## 表1：五域／19小类最终Answer F1

原19任务完整训练终点S19；所有已填值沿已核验原始结果，新增交通不重采样。Ours旧S16候选已关闭，另存负结果。

<table>
<thead>
<tr><th rowspan="2">方法</th><th>自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th>金融</th><th colspan="4">交通</th><th rowspan="2">五域宏 F1 ↑</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th><th>T17</th><th>T18</th><th>T19</th><th>Avg</th></tr>
</thead><tbody>
<tr><th colspan="24">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>73.069</td><td>64.722</td><td>48.004</td><td>41.141</td><td>73.118</td><td>53.813</td><td>48.315</td><td>27.586</td><td>28.373</td><td>39.465</td><td>18.802</td><td>45.294</td><td>52.000</td><td>78.680</td><td>23.549</td><td>66.142</td><td>57.157</td><td>24.602</td><td>35.266</td><td>38.272</td><td>36.678</td><td>37.189</td><td>47.462</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>76.399</td><td>77.047</td><td>47.579</td><td>64.043</td><td>69.489</td><td>56.328</td><td>57.303</td><td>31.609</td><td>14.015</td><td>65.231</td><td>32.179</td><td>55.291</td><td>73.000</td><td>86.091</td><td>25.416</td><td>78.011</td><td>64.854</td><td>25.728</td><td>41.268</td><td>39.827</td><td>37.427</td><td>39.601</td><td>52.375</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="24">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>83.825</td><td>68.294</td><td>40.804</td><td>41.269</td><td>63.710</td><td>53.366</td><td>49.438</td><td>29.105</td><td>12.500</td><td>64.205</td><td>24.935</td><td>45.392</td><td>62.000</td><td>77.191</td><td>25.925</td><td>81.959</td><td>63.844</td><td>33.087</td><td>35.139</td><td>39.664</td><td>37.634</td><td>38.120</td><td>52.854</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

域内Avg按TEST题数加权；五域宏对五个域等权。医学10类、遥感4类、交通3类，自然和金融各1类。T17–T19为原交通任务顺序，原名称及数据哈希见冻结manifest。—为未完成／未核验，非0分。

## 表2：持续学习与部署开销

| 方法 | AP19 ↑ | FWT ↑ | BWT ↑ | 完整保存系数 M ↓ | 新增权重参数（保存态）M ↓ | 权重 MiB ↓ | 权重增量 MiB ↓ | GPU五窗峰 MiB ↓ | TTFT s ↓ | tok/s ↑ | 适配器数 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LLaVA-OneVision-Qwen2-0.5B-OV** | | | | | | | | | | | |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8） | 45.874 | 21.156 | -2.933 | 894.216 | 0.541 | 1706.613 | 2.062 | 2176.659† | 0.315 | 31.543 | 1 |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | — | — | — | — | — | — | — | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | 52.526 | 25.734 | 0.214 | 893.676 | 0.000 | 2205.537 | 500.986 | — | — | — | 0 |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours（待新候选） | — | — | — | — | — | — | — | — | — | — | — |
| **Qwen2-VL-2B-Instruct** | | | | | | | | | | | |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8） | 48.634 | 14.864 | -5.218 | 2210.075 | 1.090 | 4217.462 | 4.156 | 4954.604† | 0.515 | 24.873 | 1 |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | — | — | — | — | — | — | — | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours（待新候选） | — | — | — | — | — | — | — | — | — | — | — |


PaRSP10%已核验全部19阶段／74格与完整21454TEST：五域宏52.375，新增权重参数0，保存态权重2205.537MiB；GPU推理峰未测。对Qwen SeqFT的0.479pp差仅指最终宏，不能当作新习得接近。[完整结果与参数证明](summary/two_backbone_pct10_20261025/PARSP_COMPLETE_FINDINGS_20261030.md)。

资源沿原S19保存态、原dtype及原固定五窗实测。完整保存系数包含必要持久状态，不等同训练坐标；原基座可重建缓存排除。新增权重参数含推理必需U/S/V基，排除mask与CL历史anchor/prior；这些非参数状态仍计入完整必要系数和字节。普通LoRA／OSFT可物化为原dense形状的0新增，保存态不是最优部署下界。[参数证据](summary/deployment_total_parameters_20261028/ADDED_MODEL_WEIGHT_PARAMETERS_20261030.json)。MiB=2²⁰字节，M=10⁶。FOREVER的权重增量约2.062MiB，属于必须比较的低成本强参照。Ours旧四域S16实测另存，不填本S19主表。

</details>
<!-- FIVE_DOMAIN_RESTORED_MAIN_END -->

<details>
<summary>已关闭的四域探索：原S16结果与负证据</summary>

<!-- FOUR_DOMAIN_MAIN_TABLES_BEGIN -->
## 1%：四域／16小类结果

### 表1：16个子任务的最终Answer F1

**已跑结果回填：带〔S19〕的行来自已核验的原19任务训练终点，本表展示其中四域／16小类，不是重新训练得到的S16结果。未来S16结果单独标记，不覆盖原记录。**

<table>
<thead>
<tr><th rowspan="2">方法</th><th colspan="1">自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th colspan="1">金融</th><th rowspan="2">四域宏<br>F1 ↑</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th></tr>
</thead>
<tbody>
<tr><th colspan="20">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）〔S19〕</th><td>75.632</td><td>60.747</td><td>43.366</td><td>37.445</td><td>52.339</td><td>49.770</td><td>43.820</td><td>27.148</td><td>5.682</td><td>42.615</td><td>16.455</td><td>40.758</td><td>65.000</td><td>48.122</td><td>17.509</td><td>66.717</td><td>46.711</td><td>15.891</td><td>44.748</td></tr>
<tr><th>FOREVER〔S19〕</th><td>64.486</td><td>66.199</td><td>32.646</td><td>51.645</td><td>49.597</td><td>53.223</td><td>43.820</td><td>61.022</td><td>14.881</td><td>72.821</td><td>16.579</td><td>46.103</td><td>78.000</td><td>75.770</td><td>4.140</td><td>65.467</td><td>50.570</td><td>9.623</td><td>42.695</td></tr>
<tr><th>CL-MoE〔S19〕</th><td>74.657</td><td>62.026</td><td>42.636</td><td>34.270</td><td>44.892</td><td>52.726</td><td>47.191</td><td>27.443</td><td>14.015</td><td>49.077</td><td>22.227</td><td>41.139</td><td>73.000</td><td>57.597</td><td>18.256</td><td>67.316</td><td>50.050</td><td>14.914</td><td>45.190</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE〔S19〕</th><td>70.697</td><td>75.000</td><td>41.147</td><td>33.413</td><td>48.522</td><td>49.802</td><td>51.685</td><td>41.523</td><td>21.111</td><td>82.667</td><td>24.492</td><td>44.202</td><td>44.000</td><td>77.056</td><td>20.699</td><td>66.092</td><td>55.738</td><td>25.223</td><td>48.965</td></tr>
<tr><th>PaRSP〔S19〕</th><td>73.346</td><td>58.904</td><td>37.123</td><td>40.064</td><td>50.786</td><td>42.467</td><td>40.449</td><td>22.730</td><td>15.530</td><td>46.615</td><td>16.883</td><td>38.743</td><td>44.000</td><td>69.983</td><td>21.751</td><td>66.417</td><td>54.088</td><td>20.609</td><td>46.697</td></tr>
<tr><th>OSFT〔S19〕</th><td>73.907</td><td>66.096</td><td>38.700</td><td>35.723</td><td>56.989</td><td>53.277</td><td>53.933</td><td>33.921</td><td>45.833</td><td>71.385</td><td>24.941</td><td>43.894</td><td>44.000</td><td>60.880</td><td>23.278</td><td>66.067</td><td>51.709</td><td>23.982</td><td>48.373</td></tr>
<tr><th>SpaRTA〔S19〕</th><td>73.596</td><td>60.395</td><td>42.465</td><td>35.552</td><td>51.505</td><td>45.899</td><td>46.067</td><td>26.868</td><td>28.301</td><td>62.410</td><td>21.327</td><td>40.921</td><td>44.000</td><td>39.120</td><td>24.601</td><td>66.742</td><td>45.942</td><td>21.328</td><td>45.447</td></tr>
<tr><th>Ours〔S16〕</th><td>69.685</td><td>61.126</td><td>36.538</td><td>33.991</td><td>50.538</td><td>49.395</td><td>25.843</td><td>29.454</td><td>9.722</td><td>74.103</td><td>18.129</td><td>39.155</td><td>45.000</td><td>32.724</td><td>18.968</td><td>69.940</td><td>43.683</td><td>23.984</td><td>44.127</td></tr>
<tr><th colspan="20">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）〔S19〕</th><td>86.312</td><td>72.061</td><td>45.827</td><td>41.170</td><td>56.989</td><td>53.606</td><td>56.180</td><td>42.002</td><td>18.056</td><td>62.923</td><td>27.650</td><td>47.654</td><td>72.000</td><td>65.076</td><td>14.116</td><td>67.516</td><td>51.110</td><td>26.576</td><td>52.913</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE〔S19〕</th><td>85.821</td><td>72.003</td><td>46.777</td><td>42.709</td><td>54.839</td><td>51.463</td><td>50.562</td><td>41.523</td><td>18.056</td><td>62.154</td><td>26.260</td><td>47.279</td><td>75.000</td><td>57.597</td><td>18.324</td><td>68.791</td><td>50.680</td><td>23.557</td><td>51.834</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>

医学：Mod.=模态，Pos.=位置，Organ=器官，Size=尺寸，Abn.=异常，Qty.=数量，Plane=切面，Shape=形状，Color=颜色，KG=知识；地理：R/U=城乡，Exist=存在性，Count=计数，Compare=比较。自然和金融各只有一个原生任务，不重复列相同的域内Avg。

小类指标为原始Answer F1（0–100）；〔S19〕行逐项采用已核验的R[19,t]，t=1…16。医学和遥感Avg按域内TEST题数加权，四域宏对四域等权，AP16对这16项等权。原S19的FWT19/BWT19保留原19任务口径，不伪装为16任务指标。—表示尚无相应完整核验或测量，不是0。

Ours〔S16〕为独立四域终点完整19331题的原生分数一致性核验结果；16阶段训练前缀与来源边界见FINAL_RESULT。FWT19/BWT19不适用于此行，留空。总部署参数893,675,552，新增部署参数0；FP16神经权重1,787,351,104 B（1704.551 MiB），权重增量0。上述按已保存S16单dense图、48层合并证明与同骨干账本登记；同一S16完整CP已独立严格重载并完成四窗资源核验；结果不代表完整TEST最坏峰。主训练540,672坐标是训练参数，不能填作新增部署参数。

### 表2：持续学习与部署开销

| 方法 | AP16@所列终点 ↑ | FWT19 ↑ | BWT19 ↑ | 完整保存系数<br>M ↓ | 新增部署参数<br>M ↓ | 完整权重<br>MiB ↓ | 权重增量<br>MiB ↓ | GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8）〔S19〕 | 41.766 | 14.783 | 1.775 | 894.216 | — | 1706.613 | 2.062 | — | — | — | — |
| FOREVER〔S19〕 | 47.495 | 17.748 | 5.642 | 894.216 | — | 1706.613 | 2.062 | — | — | — | — |
| CL-MoE〔S19〕 | 43.890 | 19.039 | 0.012 | 930.877 | — | 1846.463 | 141.912 | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE〔S19〕 | 48.321 | 14.726 | -1.732 | 1034.447 | — | 2241.552 | 537.001 | 2959.415† | — | — | — |
| PaRSP〔S19〕 | 41.729 | 18.503 | -0.484 | 893.676 | — | 2273.345 | 568.794 | — | — | — | — |
| OSFT〔S19〕 | 48.307 | 18.932 | -1.616 | 990.914 | — | 2757.984 | 1053.434 | — | — | — | — |
| SpaRTA〔S19〕 | 43.136 | 17.145 | -2.992 | 904.077 | — | 1744.231 | 39.680 | — | — | — | — |
| Ours〔S16〕 | 40.571 | — | — | 893.676 | 0 | 1704.551 | 0 | 2174.597‡ | 0.283 | 42.346 | 0 |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8）〔S19〕 | 50.504 | 13.258 | 4.415 | 2210.075 | — | 4217.462 | 4.156 | 4954.604† | — | — | — |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE〔S19〕 | 49.715 | 14.023 | 0.075 | 2286.975 | — | 4510.812 | 297.506 | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours | — | — | — | — | — | — | — | — | — | — | — |

‡ Ours〔S16〕实测为同一完整CP独立重载后四域各一固定输入、每窗32 token的allocated峰2174.597 MiB（reserved峰2478.000 MiB）；TTFT与解码速度为四窗中位数0.283 s／42.346 tok/s，含有限logit检查与逐token同步，不是生产吞吐。PAS比较仅采用同卡同raw/prepared前四窗，逐窗allocated低25.07%–26.52%；PAS没有时延读数，且其质量属S19五域训练，不据此声称同四域终点Pareto或速度优势。[同CP资源核验证书](summary/task_gradient_four_domain_final_20261029/resource_storage_retry_v2/VERIFIED.json)。

〔S19〕资源沿用历史主表原最终保存态、原dtype的已核验记录；完整保存系数包含必要持久状态，不等同可训练参数或纯神经参数。新增部署参数未单独核定时留空，不用权重字节差代替。† GPU为历史同卡固定五窗（含交通输入）、每题32 token的推理allocated峰；不是新四域固定窗测量。原权重、GPU、延迟及速度的测量边界与来源见README历史主表脚注。训练峰、最终checkpoint及完整证据另存。MiB=2²⁰字节，M=10⁶。

<details>
<summary>10%训练：性能与开销</summary>

# 10%训练：四域主表

## 表1：16个子任务的最终Answer F1

**已跑结果回填：带〔S19〕的行来自已核验的原19任务训练终点，本表展示其中四域／16小类，不是重新训练得到的S16结果。未来S16结果单独标记，不覆盖原记录。**

<table>
<thead>
<tr><th rowspan="2">方法</th><th colspan="1">自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th colspan="1">金融</th><th rowspan="2">四域宏<br>F1 ↑</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th></tr>
</thead>
<tbody>
<tr><th colspan="20">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）〔S19〕</th><td>73.069</td><td>64.722</td><td>48.004</td><td>41.141</td><td>73.118</td><td>53.813</td><td>48.315</td><td>27.586</td><td>28.373</td><td>39.465</td><td>18.802</td><td>45.294</td><td>52.000</td><td>78.680</td><td>23.549</td><td>66.142</td><td>57.157</td><td>24.602</td><td>50.030</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="20">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）〔S19〕</th><td>83.825</td><td>68.294</td><td>40.804</td><td>41.269</td><td>63.710</td><td>53.366</td><td>49.438</td><td>29.105</td><td>12.500</td><td>64.205</td><td>24.935</td><td>45.392</td><td>62.000</td><td>77.191</td><td>25.925</td><td>81.959</td><td>63.844</td><td>33.087</td><td>56.537</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody>
</table>

医学：Mod.=模态，Pos.=位置，Organ=器官，Size=尺寸，Abn.=异常，Qty.=数量，Plane=切面，Shape=形状，Color=颜色，KG=知识；地理：R/U=城乡，Exist=存在性，Count=计数，Compare=比较。自然和金融各只有一个原生任务，不重复列相同的域内Avg。

小类指标为原始Answer F1（0–100）；〔S19〕行逐项采用已核验的R[19,t]，t=1…16。医学和遥感Avg按域内TEST题数加权，四域宏对四域等权，AP16对这16项等权。原S19的FWT19/BWT19保留原19任务口径，不伪装为16任务指标。—表示尚无相应完整核验或测量，不是0。

## 表2：持续学习与部署开销

| 方法 | AP16@所列终点 ↑ | FWT19 ↑ | BWT19 ↑ | 完整保存系数<br>M ↓ | 新增部署参数<br>M ↓ | 完整权重<br>MiB ↓ | 权重增量<br>MiB ↓ | GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8）〔S19〕 | 47.586 | 21.156 | -2.933 | 894.216 | — | 1706.613 | 2.062 | 2176.659† | 0.315 | 31.543 | 1 |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | — | — | — | — | — | — | — | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours | — | — | — | — | — | — | — | — | — | — | — |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8）〔S19〕 | 50.726 | 14.864 | -5.218 | 2210.075 | — | 4217.462 | 4.156 | 4954.604† | 0.515 | 24.873 | 1 |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | — | — | — | — | — | — | — | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours | — | — | — | — | — | — | — | — | — | — | — |

〔S19〕资源沿用历史主表原最终保存态、原dtype的已核验记录；完整保存系数包含必要持久状态，不等同可训练参数或纯神经参数。新增部署参数未单独核定时留空，不用权重字节差代替。† GPU为历史同卡固定五窗（含交通输入）、每题32 token的推理allocated峰；不是新四域固定窗测量。原权重、GPU、延迟及速度的测量边界与来源见README历史主表脚注。训练峰、最终checkpoint及完整证据另存。MiB=2²⁰字节，M=10⁶。

</details>
<!-- FOUR_DOMAIN_MAIN_TABLES_END -->
</details>


<details>
<summary>历史五域两张主表及原口径（完整保留）</summary>

<!-- FORMAL_MAIN_TABLE_BEGIN -->
## 历史五域主表：双骨干1%

Acc采用唯一的**Answer F1（0–100）**。训练集为冻结的1%（1313条），测试集为完整21454条；五个数据集按19个原生任务顺序训练，seed=42。空白表示尚无完整核验结果，不能视作0分。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | 五域宏 F1 ↑ | AP19 ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 固定五窗GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ | 完整保存系数<br>个 ↓ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 小参数顺序微调参照 | 标准对照（无独立发表年份） | 86.312 | 47.654 | 51.110 | 26.576 | 33.210 | 48.972 | 47.812 | 13.258 | 4.415 | 4217.462 | 4.156 | 4954.604† | 未测 | 未测 | 未测 | 2210075136 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | 85.821 | 47.279 | 50.680 | 23.557 | 33.023 | 48.072 | 47.123 | 14.023 | 0.075 | 4510.812 | 297.506 | 未测 | 未测 | 未测 | 未测 | 2286975072 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 小参数顺序微调参照 | 标准对照（无独立发表年份） | 75.632 | 40.758 | 46.711 | 15.891 | 31.938 | 42.186 | 40.286 | 14.783 | 1.775 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 | 894216224 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) | 64.486 | 46.103 | 50.570 | 9.623 | 33.370 | 40.830 | 45.245 | 17.748 | 5.642 | 1706.613 | 2.062 | 未测 | 未测 | 未测 | 未测 | 894216224 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) | 74.657 | 41.139 | 50.050 | 14.914 | 33.110 | 42.774 | 42.288 | 19.039 | 0.012 | 1846.463 | 141.912 | 未测 | 未测 | 未测 | 未测 | 930877024 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) | 70.697 | 44.202 | 55.738 | 25.223 | 37.241 | 46.620 | 46.406 | 14.726 | -1.732 | 2241.552 | 537.001 | 2959.415† | 未测 | 未测 | 未测 | 1034447216 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) | 73.346 | 38.743 | 54.088 | 20.609 | 32.655 | 43.888 | 40.390 | 18.503 | -0.484 | 2273.345 | 568.794 | 未测 | 未测 | 未测 | 未测 | 893675552 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) | 73.907 | 43.894 | 51.709 | 23.982 | 36.196 | 45.937 | 46.357 | 18.932 | -1.616 | 2757.984 | 1053.434 | 未测 | 未测 | 未测 | 未测 | 990913568 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) | 73.596 | 40.921 | 45.942 | 21.328 | 34.520 | 43.261 | 41.829 | 17.145 | -2.992 | 1744.231 | 39.680 | 未测 | 未测 | 未测 | 未测 | 904077344 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  |

五个数据集列为训练结束后的完整域内QA平均F1；五域宏为这五个域均值等权，AP19为19个原生任务等权平均，FWT/BWT按已测的必要阶段计算。两种权重口径不同。[FOREVER（ACL 2026）正式运行](summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/README.md)已通过双骨干模型／保存重载门槛；OneVision已通过完整19阶段／74格／21454题及回放成本核验，Qwen尚无完整核验结果；原调度描述仅为历史，不能据此推断当前进度；Ours尚无可填入此正式表的完整结果。ER-LoRA已按用户要求移出比较，已有结果保留在历史证据中。

类型按主要抗遗忘机制归类，PASs-MoE兼有正则化；LoRA是参数化方式，不单独作为上述方法类型。会议链接指向正式发表页，年份按会议届次。

[方法分类、发表信息及状态](summary/two_backbone_pct01_20261017/BASELINES.md) · [完整主表与协议](summary/two_backbone_pct01_20261017/MAIN_TABLE.md) · [逐题核验索引](summary/two_backbone_pct01_20261017/completion_observer_seven_20261018/VERIFIED_INDEX.json) · [FOREVER额外完整核验](summary/two_backbone_pct01_20261017/forever_verified_20261020/README.md)

<!-- CPU_EDGE_METRICS_NOTE -->
前两项已按9组完整结果的最终checkpoint提取：**原保存态、原dtype、未折叠的完整权重与必需持久状态**，包括视觉、语言、连接器、全部专家/路由、OSFT冻结权重基、PAS掩码及CL-MoE模型外任务锚/任务权重，底层共享存储只计一次。配置可重建的位置/旋转编码缓存、逐次输入激活、视觉缓存、训练保护历史、RNG和文件容器开销另列。增加量相对同骨干原始预训练权重；包含表示和dtype变化，**不等于新增参数量**。MiB=1048576字节。这些数值是保存态权重占用，不是完整checkpoint文件大小、实际推理峰值内存或公平折叠后的压缩比较；没有运行合并、量化或重新评价。只有完成原正式核验并保存最终checkpoint的9组填入数值，未完成与Ours不填估算。GPU列仅填已核验的五窗实测，其余运行指标仍未测。 [逐方法参数数量、dtype、排除项与来源](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。 完整保存系数采用同一必要storage口径，包括必需状态而非仅可训练参数；[新增Qwen CL账本](summary/qwen_cl_completed_20261028/FINDINGS.md)。
<!-- CPU_EDGE_METRICS_NOTE_END -->

† GPU列为原正式checkpoint、原dtype／表示下的**固定五窗最大 `max_memory_allocated`**：同OLD GPU1、同五条原始QA，每题生成32个新token，包含输入准备与生成；各骨干保留原生输入展开。不是全TEST最坏峰、CPU／NVML总内存或最优折叠部署。[PAS实测](summary/strong_pas_resource_20261025/v1/FINDINGS.md) · [Qwen实测](summary/strong_qwen_resource_20261025/v1/FINDINGS.md)。

<!-- FORMAL_MAIN_TABLE_END -->

<!-- PCT10_MAIN_TABLE_BEGIN -->
## 历史五域第二表：双骨干10%训练／完整测试

训练10%已冻结为**13105 QA**，完整TEST仍为**21454 QA**；seed42、一轮、19任务，主任务共828次优化器调用，其余沿用1%正式设置。**2/16组完整核验：双骨干SeqFT-LoRA。** 其余基线按原队列运行，或已保存阶段checkpoint等待接续。空白不是0分，不复制1%结果或估计未测资源。

| 方法／Backbone | 类型 | 发表会议／年份 | TextVQA<br>自然／通用 | SLAKE<br>医学 | RSVQA-LR<br>遥感 | FinMME<br>金融 | CODA-LM<br>交通 | 五域宏 F1 ↑ | AP19 ↑ | FWT ↑ | BWT ↑ | 最终权重存储<br>保存态 MiB ↓ | 相对原基座增加<br>MiB ↓ | 固定五窗GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ | 完整保存系数<br>个 ↓ | 最终CP<br>MiB | 新习得五域宏F1 ↑ | 新习得AP19 ↑ | 状态 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| **Qwen2-VL-2B-Instruct** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 小参数顺序微调参照 | 标准对照（无独立发表年份） | 83.825 | 45.392 | 63.844 | 33.087 | 38.120 | 52.854 | 48.634 | 14.864 | -5.218 | 4217.462 | 4.156 | 4954.604† | 0.515 | 24.873 | 1 | 2210075136 | 4351.614 | 56.481 | 53.578 | 完整核验通过；最终模型已存 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队，待动态回放GPU预检 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 预检失败：损失／梯度有限性检查未通过 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已接NEW原队列，等待原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 预检失败：SVD初始化显存溢出 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| SeqFT-LoRA | 小参数顺序微调参照 | 标准对照（无独立发表年份） | 73.069 | 45.294 | 57.157 | 24.602 | 37.189 | 47.462 | 45.874 | 21.156 | -2.933 | 1706.613 | 2.062 | 2176.659† | 0.315 | 31.543 | 1 | 894216224 | 1836.784 | 49.996 | 48.652 | 完整核验通过；最终模型已存 |
| FOREVER | 回放 | [ACL 2026](https://aclanthology.org/2026.acl-long.1144/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队，待动态回放GPU预检 |
| CL-MoE | 架构 | [CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Huai_CL-MoE_Enhancing_Multimodal_Large_Language_Model_with_Dual_Momentum_Mixture-of-Experts_CVPR_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| SMoLoRA | 架构 | [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Wang_SMoLoRA_Exploring_and_Defying_Dual_Catastrophic_Forgetting_in_Continual_Visual_ICCV_2025_paper.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 已接NEW原队列，等待原生BF16空卡 |
| PASs-MoE | 架构（兼正则） | [ACL 2026](https://aclanthology.org/2026.acl-long.1474/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 正式运行 |
| PaRSP | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.1244/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 阶段CP已保存并耐久归档；研究期间暂停，恢复许可已排好 |
| OSFT | 正则化 | [ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/22620cccae29bbb3f18d226ea0320ff5-Abstract-Conference.html) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | S0完整CP已保存并耐久归档；已暂停让卡研究 |
| SpaRTA | 正则化 | [ACL 2026](https://aclanthology.org/2026.acl-long.334/) |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | OLD排队 |
| Ours | 自研方法 | 未发表 |  |  |  |  |  |  |  |  |  | 未测 | 未测 | 未测 | 未测 | 未测 | 未测 |  | 未测 |  |  | 空，不自动运行探索配方 |

五域列及五域宏／AP19为最终成绩；新习得两列来自各task训练结束的对角格，按任务等权或域内QA加权再五域等权，均可由原74格得到，不是同一时点的域末模型。BWT仅为继承的描述量，不作优化目标。完整保存系数沿用1%必要storage口径，共享底层存储只计一次；神经权重与必需非参数状态分列明细，派生缓存、训练保护历史与RNG排除，不等同主优化器坐标。完整系数、权重／CP与GPU峰须逐组合实测后填写。

第二主表完整覆盖两个backbone各八类基线，共16组；主要报告与1%相同的五域最终F1、宏F1／AP19、FWT／BWT及全部资源指标，另保留完整参数量与CP大小。每组S19完整模型权重、必要方法状态、processor／config及来源记录均须耐久留存；不是只记录参数量或保存LoRA。[最终模型留存](summary/two_backbone_pct10_20261025/FINAL_MODEL_RETENTION.md)

刚习得及金融集中错答是补充分析。LLaVA SeqFT的最终五域宏从1%的42.186升至10%的47.462，四域提高、自然域下降；金融刚习得F1从16.405升至25.414，单选回答却更集中。**集中不能替代新习得判断；不同题型的学习效果也不同。** 数据量、更新预算及先前课程共同变化，不能单独归因样本量。[首组完整证据](summary/two_backbone_pct10_20261025/FIRST_COMPLETE_FINDINGS.md) · [分题型观测与结论](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md)

Qwen SeqFT最终五域宏由1%的48.972升至10%的52.854，遥感、金融、交通提高，自然和医学下降；完整参数仍为2,210,075,136，保存态权重4217.462 MiB，新增训练参数1,089,536。更多训练改善总体成绩，但没有均匀改善所有域；尚不能将效果归因于纯数据量。10% GPU资源已按原最终模型完成固定五窗实测。[第二组完整证据](summary/two_backbone_pct10_20261025/SECOND_COMPLETE_FINDINGS.md)

[10%正式协议与冻结证据](summary/two_backbone_pct10_20261025/PROTOCOL.md) · [10%独立主表](summary/two_backbone_pct10_20261025/MAIN_TABLE.md)

<!-- PCT10_MAIN_TABLE_END -->

## 主表怎么读

- **评价口径：** 主表为冻结1%训练集1313条、19阶段、完整21454 TEST、seed42；五域宏是五个域均值等权，AP是19任务等权，两者不混用。已完整核验9/16组，空白不代表0分。
- **最强参照：** 当前已核验的LLaVA-OneVision内五域宏最高为PASs-MoE **46.619976**；跨骨干最高为Qwen顺序LoRA **48.972351**。新方法需同时报告这两个比较范围，不能把PAS无条件称为全体最强。
- **基线接续：** 新服务器由队列使用GPU1/2/3，只保留GPU0不由我方分配；移除了不必要的队列屏障。原基线继续运行，新完整核验结果出来后补表。[运行与队列详情](summary/baseline_capacity_20261025/v1/FINDINGS.md)。


</details>

## 探索命题与结论

**核心观测：专家参数更多，输入相关的局部功能变化未必更大。** 相同LLaVA骨干、1%正式训练后的两个S19模型，PAS新增参数更多，但相对固定gate均值混合的局部输出差异更小：

| 原模型 | 新增神经权重参数（个） | 局部差值平方和／原适应增量平方和 | 对应相对L2范数 |
| --- | ---: | ---: | ---: |
| PASs-MoE | 140,771,328 | 0.010457% | 1.0226% |
| CL-MoE | 37,158,912 | 2.952336% | 17.1824% |

**怎么测：** 冻结原模型，在相同80条历史TRAIN（五域各16条）、全部168个投影的原动态隐藏输入上，比较真实动态专家输出与固定全局平均gate混合输出；每题先平均答案预测位置（含EOS），再对题等权，最后求168投影的平方能量和之比。五域分别汇总，PAS的该比例均低于CL，完整分域数值与原始读数见[观测与证明](summary/router_function_moments_20261030_v2/FINDINGS.md)。

**这说明什么：** 新增参数规模不能替代条件功能测量；部署表示应依据实际功能偏离选择。上述比例的分母是原适应增量，**不是整个层输出，也不是F1损失或“可删除参数比例”**；均值参照并非最优静态表示。这是两个已训练模型、历史TRAIN答案位置上的反例，尚非跨骨干／全部CL方法的普适因果规律。正式PAS-static的46.332 F1使用另一套能量加权校准系数，与本表平均gate参照分开报告。

**完整部署性质：此PAS末态的大专家表示可用一个固定稠密骨干近似承载。** 全21454 TEST五域宏46.332，对原PAS46.620下降0.288pp；各域自然／医学／遥感／金融／交通为70.661／44.152／54.218／25.281／37.347。893,675,552总部署参数、新增0，权重1704.551MiB；同卡五窗GPU峰逐窗低24.84%–26.81%。不证明训练router无用或所有条件结构均可删除，也不把普通权重合并本身称新原理。[完整结果与范围](summary/pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)

**已有GPU过程反证：** 同11条TRAIN，q/v与all一步均让11/11答案CE下降，不能称该批q/v“没学动／一阶朝上”。all降幅较大也伴更大真实权重位移，原因尚未识别；它不是最终新增参数少的方法成果，已准备的范数控制未启动、不自动立项。[实际过程观测](summary/train_step_space_diagnostic_20261029/FINDINGS.md) · [观测标准](summary/five_domain_restored_20261029/MECHANISM_EVIDENCE_REQUIREMENTS.md)


连续学习先导通常采用1213 FIT／100 DEV、19阶段；其它诊断使用各自固定范围。DEV已用于多次探索，结果用于筛选，不能与正式TEST分数直接排名，也不据此宣称显著提升。下表区分实际负结果、局部支持、质量缺测和待验证假说。



| 观测的问题与探索命题 | 检验结果 | 结论与设计影响 |
| --- | --- | --- |
| **按大域取均值能否承载条件变化？** | 两法gate域内变差仍为79.24%/70.66%；域均值局部误差分别保留全局均值参照的72.83%/66.09%，函数分解的cross非零 | **大域有部分组织，但均值参照仍留下多数残差。** 不是域划分无效/最优表示下界；不把误差下降叫功能解释率。[分解、范围与设计影响](summary/router_function_domain_decomposition_20261030/FINDINGS.md) |
| **更多新增参数是否对应更大的条件响应？** | PAS／CL新增神经权重140,771,328／37,158,912；同80TRAIN／168投影，固定平均gate参照的局部差值平方比0.010457%／2.952336%，五域排序一致 | **参数规模与本次局部条件变化指标排序相反。** 作为端侧表示选择的观测依据；非最优表示下界、非整模删参或F1等价证明。[完整读数与证明](summary/router_function_moments_20261030_v2/FINDINGS.md) |
| **路由是否回避专家输出差异大的方向？** | 两法总体Rjoint/Rindep为1.572/1.065，五域汇总均>1，部分单层<1 | **原负耦合假设未获支持。** 不用抵消故事解释静态可行性。[完整正负证据](summary/router_function_moments_20261030_v2/FINDINGS.md) |
| **遥感presence的高分是否意味着学会条件区分？** | LLaVA 1%刚学完时，Seq／OSFT／PAS分别有2912／2947／2955题输出各自同一评分token标准化答案（共2955题，三者众数相同）；PAS恒输出该答案仍有2217题满分、F1为75.025 | **高分不能单独证明条件区分。** 该现象与[RSVQA已知答案先验](https://arxiv.org/html/2311.16782v1)重合，保留为诊断复现，不作为端侧创新点；尚不知其训练原因，也不能断言模型内部未使用图像。count虽低，多种方法本阶段仍有正增量。[跨方法原轨迹及边界](summary/domain_acquisition_observation_20261029/FINDINGS.md) |
| **训练附加结构是否必须原样部署？** | 八法实际前向：OSFT／LoRA输入无关部分可物化，PaRSP已是原稠密参数；CL／PAS／SMo／SpaRTA保留不同条件计算，FOREVER原新增约2.062MiB | **必须按功能区分，不能一概称大结构冗余。** PAS仅局部近似证据，CL已完成局部联合功能观测、静态替换仍未验证；不把常规合并当创新。[跨方法审计与共同准则](summary/deployment_total_parameters_20261028/CROSS_METHOD_FUNCTION_AUDIT_20261029.md) |
| **条件模块的完整参数是否都是推理必需自由度？** | SpaRTA当前eval分支的K/W矩阵只经normalize后K行和与W行和进入最终标量，实数算术可保留动态功能而收缩表示 | **代数层面的观测，未有GPU/F1结论。** 保守公式载荷可少12.455MiB，不能称零新增、训练矩阵无用或泛化压缩成功。[推导、适用分支与边界](summary/deployment_total_parameters_20261028/CROSS_METHOD_FUNCTION_AUDIT_20261029.md) |
| **最终低分能否直接说明额外结构没有新习得收益？** | Qwen CL最终五域均低于Seq，但刚习得14/19任务更高，五域宏+1.797pp；医学+8.873、金融−2.082 | **该解释被否证。** 收益不均，不能把最终差距当新习得差；完整权重多约7%，GPU仍缺测，不将坐标比写成内存比。[证据与成本](summary/qwen_cl_completed_20261028/FINDINGS.md) |
| **少量更新坐标能否借当前任务梯度获得更好回答？** | 同S15、69FIT／5更新：实际SVD金融DEV 20，随机10，共同before5；部署图／GPU相同 | **实际策略有局部正向信号，原范数匹配失败。** 10pp只来自两题，一题额外新学、一题保留旧满分；不证明纯方向、显著或全域优势。[完整正负与资源](summary/task_direction_local_20261027/DESCRIPTIVE_FINDINGS_20261028.md) |
| **强方法主要解决遗忘，还是先学得更好？** | 正式刚习得均分：OSFT47.887821、Seq38.604705；BWT分别−1.616191、+1.774861 | **支持优先研究有效习得。** OSFT在本轨迹上刚习得更强，保持更差；尚未识别其性能原因，不能把所有缺口归为遗忘。[证据](summary/final_weight_reduction_20261024/osft_initial_gap_check_v1/FINDINGS.md) |
| **低曝光新专家的冷启动能否靠继承改善？** | 局部11条医学QA／一次更新，从头19.88、完整继承29.26；独立功能中性控制没有复现完整继承收益 | **局部支持功能继承。** 不支持“所有方法每域清零”或“只开放首步A梯度就够”。[冷启动](summary/onevision_diagnostics_20261016/expert_cold_start_20261018/RESEARCH_FINDINGS.md) · [控制](summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/function_vs_factor/RESEARCH_FINDINGS.md) |
| **OSFT部分训练与保护机制是否可省？** | ShortFixed宏42.301329、原OSFT40.783801，梯度及Adam状态少21.34%；去逐步保护42.333991，但金融下降 | **局部支持简化训练状态，未证明最终模型更小。** 点估计不证明等效或保护无用；阶段刷新对照故障，质量缺测。[短侧](summary/edge_deployment_20261019/osft_short_factor_control_20261022/v2/FINDINGS.md) · [保护](summary/edge_deployment_20261019/osft_protection_control_20261022/v1/FINDINGS.md) |
| **普通all-r8能否达到强基线？** | 完整1%宏44.894423，低于OSFT/PAS，高于q/v的42.185908 | **未达原强基线门槛，但保留为简单参照。** 独立部署重载与推理峰缺测，不把训练峰当推理显存。[原完整结果](summary/edge_deployment_20261019/osft_increment_confirmation_20261022/v1/FINDINGS.md) · [比较范围](summary/task_gradient_formal_20261028/COMPARATOR_SCOPE.md) |
| **借原生FFN响应做轻量调制有净收益？** | 动态44.757603、固定44.681201；匹配宽q/v为45.085963，分别高0.328360／0.404762pp | **未支持新增调制的净质量收益。** 原生响应非退化并不等于有答题价值；宽q/v保留为容量对照，正式推进已关闭。[匹配结果](summary/native_response_selection_20261025/qv_control/FINDINGS.md) |
| **低损伤排名、低秩或局部拟合能直接承接回答功能？** | 联合删8个FFN：43.791987→19.620168；FFN低秩／功能拟合多组虽减重8.61%–25.83%，F1仍崩坏 | **这些固定替代方案保质失败。** 局部误差下降、单删低损伤都不足以证明完整图可删；不否定所有压缩。[删图](summary/ffn_direct_ablation_20261025/v1/FINDINGS.md) · [角色对照](summary/final_weight_reduction_20261024/ffn_role_retention_v1/README.md) · [读出](summary/final_weight_reduction_20261024/functional_readout_carrier_v1/README.md) |
| **换更小预训练载体，就能同时保质且省显存？** | 小视觉接口宏38.966256<父43.791987；完整SmolVLM学习后37.991536，权重少43.21%，五窗GPU峰却高14.09%–57.43% | **本组质量／资源联合目标失败。** 参数少不能代替GPU峰实测，预训练完整也不保证功能承接。[视觉接口](summary/vision_interface_transfer_20261025/v1/FINDINGS.md) · [原生小模型](summary/native_prior_learning_20261025/compatibility/INTERPRETATION.md) |
| **OSFT优势是图文配对利用整体更强？** | 61题四状态488次NLL，配对支持改善OSFT相对q/v差−0.013734 | **未支持预定解释。** 不能反推视觉无用或可以删视觉编码器。[证据](summary/final_weight_reduction_20261024/paired_image_support_diagnostic_v1/RESULT_FINDINGS.md) |
| **历史隐藏态约束应否先检查它对当前回答是否有利？** | DEV宏：朴素r8 38.775370、原法 43.703262、答案门 43.719891、同数打乱 43.671936 | **微小正向点估计，尚不足确立主创新。** 门仅高原法0.016628pp、高打乱0.047954pp；四个非交通域F1相同。主要收益已有于共同回放／EMA配方，不能归新门。[完整结果](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md) |
| **共同回放配方的终态收益来自更好的新习得？** | 两方法医学刚习得均36.857143、遥感均65；40条逐题F1全同，0高／0低／40平 | **本次两域不支持该解释。** 终态差在后续形成，不将约4.93pp宏收益归为更会学新任务。按人类要求，后续只查新习得原因，不开保持／遗忘分支。[证据与独立评判](summary/acquisition_readout_20261025/v1/FINDINGS.md) |
| **金融TRAIN标签扩大到10%后是否更均衡？** | 单选金标A/B/C/D：1%为22/18/7/6，10%为189/163/99/76；最大占比41.509%→35.863% | **监督偏斜减弱但仍存在。** 不能据计数认定模型塌缩原因或自动做先验校正；模型观测见下一行。[完整比较与独立审阅](summary/train_answer_prior_20261025/INTERPRETATION.md) |
| **10%会消除金融集中错答、均匀改善新习得吗？** | LLaVA／SeqFT刚习得F1：16.405→25.414；单选不同回答174→9，但原满分331→495、零分966→812。10%多选仍286题严格单字母且原token F1全零（非选项集合正确率），数值训练前8.640→刚学完6.974 | **集中与更高F1可以并存，题型习得并不均匀。** 不支持把集中直接当作失败或统一位置偏置原因；单组结果不推广全部基线，也不隔离纯数据因果。[完整同ID比较与独立评判](summary/train_answer_prior_20261025/PCT01_PCT10_RESPONSE_FINDINGS.md) |
| **两个医学任务的阶段降分都意味着内容退化吗？** | 全491题配对、208项回答变化经双agent盲读。Plane原F1−0.718pp而EXACT+0.862pp；Position的未知含义变化承载77.75%总失分量 | **不能统一归因。** Plane有明确表述效应，Position主因未识别；不据此选择统一格式／知识模块，EXACT也非语义正确率。[完整正负与未知](summary/train_answer_prior_20261025/MEDICAL_PAIRED_FINDINGS.md) |

**背景对照，不作为核心研究发现。** 匹配探索中朴素r8宏38.775370、宽q/v45.085963，训练坐标相差约8.14倍；这说明容量比较的量级，不能解释为什么学不好。同LLaVA／SeqFT的1%→10%刚习得19/19任务提高，五域宏42.026→49.996；阶段内增量却13高6低。这些结果是后续问题的背景，不能单独作为优化机制发现。[容量对照](summary/plain_r8_gpu_control_20261025/v1/FINDINGS.md) · [全19任务](summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION_FINDINGS.md)

**跨骨干背景核对：局部失败不固定在同一题型。** Qwen刚习得五域宏46.985→56.481，19任务中17提高、2下降（Quantity、Shape）；10%时Shape阶段仍下降1.111。LLaVA在10%中下降的Position／Plane，Qwen却分别提高3.425／7.447。因此不支持“这两类医学任务统一学不会”或一个共同根因；能保留的是固定配置下学习收益不均匀。两比例同时改变数据、更新与课程，Qwen历史训练硬件也不同，不能作纯数据因果。[Qwen完整对照](summary/train_answer_prior_20261025/QWEN_ACQUISITION_FINDINGS.md) · [LLaVA完整对照](summary/train_answer_prior_20261025/ALL_TASK_ACQUISITION_FINDINGS.md)

**PAS的优势包含跨域新习得，但不能归因于路由。** 正式1%全19任务补齐后，PAS刚习得五域宏48.349，普通LoRA为42.026；医学／遥感／金融／交通更高，自然更低，任务15高4低。但交通T18／T19刚习得更高、阶段增量反而更小：继承起点不同，不能把终点高分统一称为本阶段学得更多。[完整观测与局限](summary/conditional_shared_budget_20261026/PAS_ACQUISITION_FINDINGS.md)

**强方法改善的金融答题功能不同。** 同样1%数据，546道多选刚习得原F1：SeqFT／OSFT均为0，PAS为13.736（75题原F1满分）；严格单字母输出分别504／545／127题。OSFT的金融优势来自单选，PAS则同时获得单选和多选收益；三法数值题阶段均值都下降。不能统一解释为“数据太少学不会”，也不能把PAS差异直接归因于专家路由。[全部题型、前后读数与负贡献](summary/train_answer_prior_20261025/FINANCE_STRONG_ACQUISITION_FINDINGS_20261027.md)

**不能直接用隐藏题型或显式关键词分流。** 原输入不传入题型metadata；预定单／多选关键词在1%金融TRAIN的89题全部未命中，10%的887题仅9题命中多选提示（8多选、1单选）。这否定该字面规则作为可靠分流依据，不等于语义不可区分，也不证明输入歧义就是失败原因。[输入信号及方法约束](summary/train_answer_prior_20261025/FINANCE_TRAIN_TYPE_CUES_FINDINGS_20261027.md)

**已检验命题：同样的小预算，条件适应是否比简单权重混合值得？** 三臂完整结果不支持保留当前soft点：DEV最终宏shared／uniform／soft为38.623／39.663／39.456，soft低于uniform且显存略高。它的新习得收益集中在一条医学表述变化、一条金融选项变化，不能称普遍知识改善；uniform相对shared最终约96.1%的净提升来自同一条遥感二元题。当前点关闭，三臂完整参数及正负证据均保留，Ours仍空。[完整对照、逐题贡献与资源](summary/conditional_shared_budget_20261026/RESULT_FINDINGS_20261027.md)

**跨骨干金融题型：多选现象复现，数值降分不复现。** Qwen普通LoRA在1%／10%的多选刚习得F1为0／10.989，546题中严格单字母为403／259，与LLaVA的0／8.425及504／286有共同现象。但Qwen数值题两个比例的当前阶段均提高（+0.526／+0.789），不复现LLaVA的阶段下降。不能把跨比例终点变化当当前训练损伤，也不能据此认定统一题型混淆根因。该观察不单独证明题型监督有效。[完整题型、前后变化与边界](summary/train_answer_prior_20261025/QWEN_FINANCE_TYPES_FINDINGS_20261027.md)。

**PAS的多选收益不等于普遍学会完整组合。** 546题中，PAS刚习得输出长度只有1／2／4（127／198／221题），没有三字母输出，但合法参考含162道三字母题。75题原串匹配之外，还有176题仅多选、151题仅缺选、91题同时错选漏选；不能把它的高分直接归因精准选项组合或复杂专家必要。普通LoRA有353题输出是gold成员子集，但均匀字母本来也常是成员，不能据此断言内容已经学会或只需续写。研究FIT的17多选已覆盖2／3／4成员（9／4／4题），但不能据字符覆盖推出合法token负候选；成员辅助头和负损失均暂不推进，避免将PAS过选直接当小LoRA漏选的干预依据。[候选评判](summary/train_answer_prior_20261025/ANSWER_CONSTRUCTION_DECISION_20261027.md)。[完整观测与边界](summary/train_answer_prior_20261025/FINANCE_MULTICHOICE_LITERAL_FINDINGS_20261027.md)。

## 参数量与已测GPU优势

当前候选使用 **LLaVA-OneVision-Qwen2-0.5B-OV**：Qwen2-0.5B-Instruct语言骨干＋SigLIP视觉编码器，原完整模型 **893,675,552** 个系数。下表是已核验checkpoint的原保存表示，包含整个模型与必需持久状态，不是只计adapter。

| 骨干／方法 | 完整保存系数数量 | 权重存储 MiB |
| --- | ---: | ---: |
| LLaVA-OneVision／SeqFT-LoRA | 894,216,224 | 1706.613 |
| LLaVA-OneVision／FOREVER | 894,216,224 | 1706.613 |
| LLaVA-OneVision／CL-MoE | 930,877,024 | 1846.463 |
| LLaVA-OneVision／PASs-MoE | 1,034,447,216 | 2241.552 |
| LLaVA-OneVision／PaRSP | 893,675,552 | 2273.345 |
| LLaVA-OneVision／OSFT | 990,913,568 | 2757.984 |
| LLaVA-OneVision／SpaRTA | 904,077,344 | 1744.231 |
| Qwen2-VL／SeqFT-LoRA | 2,210,075,136 | 4217.462 |

保存dtype与表示不同，**存储增加不等于新增参数量**；普通LoRA与OSFT常规fold后都可回原稠密形状，不能拿OSFT未折叠保存态制造独占减重优势。[完整统计口径](summary/final_weight_reduction_20261024/checkpoint_weight_storage_v1/FINDINGS.md)。

**GPU实测成立的范围：** 已完成宽q/v对原完整PAS的同卡、同五题、各32新token测量。宽q/v最终893,675,552个系数，比PAS少**13.61%**；原dtype权重字节少**23.96%**；五窗口GPU峰 **1857–2166 MiB**，PAS为 **2490–2959 MiB**，逐窗低**24.84%–26.81%**。包含原FP32专家／FP16基座的dtype差异，仅是这五窗的原实现比较；未证明同dtype最优部署或全输入峰。[实测证据](summary/native_response_selection_20261025/qv_control/RESOURCE_COMPARISON.json) · [参数与dtype影响拆分](summary/edge_resource_comparison_20261025/v1/PRECISION_FAIRNESS_REVIEW.md)。

**全体最强基线的资源锚也已补齐：** 原Qwen顺序LoRA完整图在同卡／同五条原始QA／32-token窗实测为4312–4955 MiB；匹配LLaVA朴素r8为1857–2166 MiB，逐窗少55.15%–58.80%。完整系数较Qwen少59.56%，主要来自骨干与表示差异，不能称新增门的结构贡献。Qwen保原FP16基座＋FP32 LoRA、未fold；各自原生输入展开不同，非最优部署或全分布峰。[完整证据](summary/strong_qwen_resource_20261025/v1/FINDINGS.md) · [三组资源对照图](summary/edge_resource_comparison_20261025/v1/FINDINGS.md)。

**这仍不是Ours的联合成功：** 宽q/v训练参数4,399,104，比朴素r8的540,672多3,858,432，约8.14倍；合并后均无新增推理参数。其F1是DEV结果，尚未证明正式质量胜最强基线。

## 已结束探索：条件混合点关闭

**问题与方法。** PAS有跨域新习得优势但持久专家成本较高；我们检验能否用540,672个训练坐标承接这项收益。普通shared r8、固定4/2/2混合uniform、A响应能量加权soft共用初始化与训练暴露。soft没有额外router参数，但保留条件分支；简单两臂实际fold。借鉴已有机制本身不是否决理由，取舍依据是以下实测结果。

| 内部探索臂 | 刚习得宏F1 | 最终宏F1 | 完整系数 | 权重MiB | 五窗GPU峰MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| shared | 43.252 | 38.623 | 893,675,552 | 1704.551 | 2166.101 |
| uniform | 42.367 | 39.663 | 893,675,552 | 1704.551 | 2166.101 |
| soft | 43.819 | 39.456 | 894,216,224 | 1706.613 | 2168.163 |

**为什么不保留soft？** 它相对uniform最终低0.208pp，权重与allocated峰各多2.0625 MiB；前四域逐题F1相同，最终差异全在交通。新习得虽高1.452pp，但医学增益仅来自一条中英文表述变化，金融增益仅来自一题且shared已同样答对。uniform最终高shared1.041pp，也主要由一条遥感题贡献，不能包装为新框架成功。

这是1213 FIT／100 DEV、单种子的完整探索，**不是正式主表结果**。阶段起点不同，不把增量差当同起点学习效率；固定五窗峰不是全输入最坏值。关闭本固定soft方案，保留全部参数和证据；不推广为所有条件机制无效，不调设置救结果。[完整结论及核验记录](summary/conditional_shared_budget_20261026/RESULT_FINDINGS_20261027.md)

**成对选项监督：仅见弱局部信号，尚不足以进入正式实验。** 两组完整训练与评估后，刚习得宏F1由43.274升至44.252，但正收益仅来自一道金融题；最终宏F1仅由38.701升至38.772。12题置换诊断中稳定且严格正确从1题增至2题，稳定且错误均为3题。干预前15阶段参数摘要并不相同，不能把全部差异单独归因于配对监督。两组完整权重均为893,675,552系数／1704.551 MiB，固定五窗GPU峰均2166.101 MiB：无新增部署成本，也没有相对普通小LoRA的内存优势。这是已探索DEV的单seed结果，Ours保持空，不追调当前配方。[设计](summary/train_answer_prior_20261025/PAIRED_OPTION_DESIGN_20261027.md) · [结果审阅](summary/train_answer_prior_20261025/PAIRED_RESULT_REVIEW_20261027.md)

**已结束的答案门探索：** 原法、答案门、同数打乱的DEV宏分别43.703262、43.719891、43.671936；新增门只高原法0.016628pp、高打乱0.047954pp。共同配方和普通r8的医学／遥感刚习得分别完全相同（36.857143／65），因此不能将终态改善归为这两域的新习得收益，也不足以确立新增门为主创新。这条探索已结束，不是当前正在验证的方法。[完整正负结果](summary/answer_reliability_consolidation_20261025/v1/FINDINGS.md) · [新习得对照](summary/acquisition_readout_20261025/v1/FINDINGS.md)

**位置诊断已完成：换选项位置后，所选内容常发生变化，10%也未消除这一现象。** 固定45条共同TRAIN、一次循环，在金融刚学完状态，1%有32/37、10%有31/40条有效配对改变所选内容；两比例共同有效的33题中为28→24。训练前有效配对各仅2题，无法据此判定金融训练如何改变位置敏感性。稳定选择也可能选错；这不是正确率或所有金融错误的统一原因，不自动推出PIF训练方案。[完整结果与边界](summary/train_answer_prior_20261025/POSITION_FINDINGS.md)

---

[探索过程与证据索引](SUPPLEMENTARY_NOTES.md#探索过程与证据索引) · [完整实验口径与数据来源](SUPPLEMENTARY_NOTES.md) · [历史研究记录](README_HISTORY_20261024.md)

**题型辅助监督未改善目标新习得。** 同一S15起点，Plain／真实题型／打乱题型的刚习得宏F1为43.255／43.127／43.121，最终宏为38.633／38.740／38.716；金融刚习得均15，多选4题仍全为单字母且原F1为零。真实标签的最终微增全来自交通，与打乱标签仅差0.023，不能证明目标功能有效。三组完整系数893,675,552、权重1704.551 MiB、五窗GPU峰2166.101 MiB一致；训练头新增2691参数、部署移除，仅证明无新增部署成本。关闭此固定方案，不推进正式表；单seed小DEV不支持题型信息普遍无用的结论。[完整正负结果](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_FINDINGS_20261027.md) · [独立审阅](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_RESULT_REVIEW_20261027.md)

10% Qwen OSFT在SVD初始化预检发生显存溢出，保留为固定条件下的失败；它不是推理显存或质量分数。[失败证据](summary/two_backbone_pct10_20261025/QWEN_OSFT_INITIALIZATION_FAILURE_20261027.md)

[题型辅助研究首次实际启动及基线暂停留存证据](summary/train_answer_prior_20261025/TYPE_AUXILIARY_FIRST_STARTED_20261027.json)。

[题型辅助导出故障独立审阅](summary/train_answer_prior_20261025/TYPE_AUXILIARY_EXPORT_FAILURE_REVIEW_20261027.md)；完整S15与失败现场已耐久归档，141文件逐SHA核对，原始记录保留。

10% SeqFT资源列：原最终未折叠模型、同卡五条固定QA、每条32新token；GPU为五窗最大allocated峰，TTFT与解码速度为五窗中位数。计时包含同步观测开销，不是全TEST最坏峰或生产吞吐。[完整实测与核验](summary/two_backbone_pct10_20261025/FINAL_RESOURCE_FINDINGS_20261027.md)。

三臂完整最终模型、S15、答案及失败历史已耐久归档，309文件逐SHA校验。PaRSP已从S1恢复并进入S2。[归档证据](summary/train_answer_prior_20261025/TYPE_AUXILIARY_V2_ARCHIVE_20261027.json)。
