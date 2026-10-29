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
