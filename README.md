# 多领域、数据受限的多模态持续学习

1%与10%分别使用1313与13105条训练样本；均评估五域19任务、完整21454条TEST。以下为训练末态S19的原token F1；基线沿原方法、原推理图报告。域内Avg按题数加权，五域宏对五域等权；“—”表示尚未完成或核验，不是0分。

## 1%训练：五域主表

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

## 10%训练：五域主表

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
<tr><th>OSFT</th><td>73.470</td><td>74.503</td><td>51.170</td><td>62.012</td><td>68.683</td><td>53.235</td><td>53.933</td><td>34.195</td><td>41.667</td><td>76.462</td><td>31.360</td><td>55.061</td><td>63.000</td><td>71.540</td><td>16.390</td><td>81.459</td><td>59.176</td><td>27.146</td><td>37.818</td><td>40.095</td><td>35.208</td><td>38.408</td><td>50.652</td></tr>
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

## 观测结论

**已观测：PASs-MoE 1%的动态专家增量，在已测80题上比CL-MoE 1%更接近固定专家组合。** 对通过专家扩展学习多个领域的模型，训练后保留专家参数与逐输入选择专家，是需要分别检验的两件事。我们比较同一已训练模型在同一隐藏输入上的两种专家增量：一是用该输入的原路由权重组合专家输出，二是用历史样本的平均路由权重固定组合这些专家输出；专家本身保持不变。固定的是组合系数，输出仍随输入变化。这里测的是动态组合相对固定组合额外改变了多少输出，不是新增参数方向与基础模型方向是否一致，也不是专家之间是否相同。

在LLaVA、1%训练的PASs-MoE与CL-MoE末态模型中，新增专家参数分别为140,771,328与37,158,912；上述输出差值的平方和占原专家增量平方和的0.010457%与2.952336%。参数更多的PASs-MoE，在这批输入上反而更接近固定组合。因此，具体的持续学习部署问题是：**为学习多个领域保留的专家结构，在推理时有多少必须依赖逐输入的动态组合，又有多少能由固定组合承载？** 当前局部观测仅覆盖80条历史TRAIN的答案预测位置、168个投影，未比较新增方向与基础模型方向；该比例不是F1损失或可删除参数比例。是否能简化整个部署模型，须由下面的完整生成与资源对照检验。[局部专家输出观测](summary/router_function_moments_20261030_v2/FINDINGS.md)

**已观测：PASs-MoE 1%的固定部署表示，五域宏F1低0.288个百分点、必要权重载荷少23.96%，但交通域83.23%的题F1发生变化。** 完整21454题上，固定稠密表示的五域宏F1为46.332，原动态表示为46.620，相差−0.288个百分点；必要权重载荷减少23.96%，同卡五个固定输入窗口的allocated峰下降24.84%–26.81%。89.00%的题F1相同，但交通域83.23%的题发生双向变化并大幅抵消；遥感域动态表示仍高1.519个百分点。该干预同时包含固定混合、折叠与精度变化，不能单独归因为路由，也不能据此否定专家训练的价值。[完整配对观测](summary/pas_static_final_confirmation_20261029/OBSERVATION_PAS_FIXED_DYNAMIC_PAIRED_20261031.md) · [质量与资源证明](summary/pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)

**正在证明：CL-MoE 1%在固定专家组合后，完整测试质量会改变多少。** 对同一已训练检查点，固定原有效专家组合系数，保留原专家权重、精度与生成设置，比较完整21454题的五域宏F1、逐域差值与逐题双向变化。目前没有完整对照结果，本次完整验证尚待执行。局部输出接近不能替代这项完整检验。

**尚未验证：上述观测能否推广到10%训练、其它持续学习方法，以及真实端侧的时延和能耗。** 也未证明该现象由数据受限或序贯学习造成；不能将当前场景下的两个模型观测写成持续学习的普遍规律，或把宏平均接近写成动态选择无用。
