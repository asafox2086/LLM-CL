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

PAS-static为原1% PAS检查点的固定末态部署表示：80条历史TRAIN及答案校准后，系数冻结，独立完整重载并评测21454 TEST；不是一次新的持续训练。19小类／五域宏及AP19均来自其自身完整评估，FWT/BWT未测。总部署893,675,552神经参数、新增0，权重1704.551MiB；普通buffer另9032B。GPU为同卡同五窗各32新token的最大allocated，不是全TEST峰或速度。成本包含原PAS训练及校准/export；FOREVER和可合并LoRA／OSFT仍是公平参照。[完整核验、资源与边界](../pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)。

资源沿原S19保存态、原dtype及原固定五窗实测。完整保存系数包含必要持久状态，不等同训练坐标；原基座可重建缓存排除。新增权重参数含推理必需U/S/V基，排除mask与CL历史anchor/prior；这些非参数状态仍计入完整必要系数和字节。普通LoRA／OSFT可物化为原dense形状的0新增，保存态不是最优部署下界。[参数证据](../deployment_total_parameters_20261028/ADDED_MODEL_WEIGHT_PARAMETERS_20261030.json)。MiB=2²⁰字节，M=10⁶。FOREVER的权重增量约2.062MiB，属于必须比较的低成本强参照。Ours旧四域S16实测另存，不填本S19主表。
