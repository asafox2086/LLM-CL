# 1%训练：四域主表

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

Ours〔S16〕为独立四域终点完整19331题的原生分数一致性核验结果；16阶段训练前缀与来源边界见FINAL_RESULT。FWT19/BWT19不适用于此行，留空。总部署参数893,675,552，新增部署参数0；FP16神经权重1,787,351,104 B（1704.551 MiB），权重增量0。上述按已保存S16单dense图、48层合并证明与同骨干账本登记；GPU实测另列，独立重载检查待完成。主训练540,672坐标是训练参数，不能填作新增部署参数。

## 表2：持续学习与部署开销

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
| Ours〔S16〕 | 40.571 | — | — | 893.676 | 0 | 1704.551 | 0 | — | — | — | — |
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

〔S19〕资源沿用历史主表原最终保存态、原dtype的已核验记录；完整保存系数包含必要持久状态，不等同可训练参数或纯神经参数。新增部署参数未单独核定时留空，不用权重字节差代替。† GPU为历史同卡固定五窗（含交通输入）、每题32 token的推理allocated峰；不是新四域固定窗测量。原权重、GPU、延迟及速度的测量边界与来源见README历史主表脚注。训练峰、最终checkpoint及完整证据另存。MiB=2²⁰字节，M=10⁶。
