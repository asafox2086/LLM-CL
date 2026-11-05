# 多领域、数据受限的多模态持续学习

1%与10%分别使用1313与13105条训练样本；均评估五域19任务、完整21454条TEST。以下为训练末态S19的原token F1；基线沿原方法、原推理图报告。域内Avg按题数加权，五域宏对五域等权；“—”表示尚未完成或核验，不是0分。

## 1%训练：五域主表

<table>
<thead>
<tr><th rowspan="2">方法</th><th>自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th>金融</th><th colspan="4">交通</th><th rowspan="2">五域宏 F1 ↑</th><th rowspan="2">新增参数量（M）↓</th><th rowspan="2">S19可训练参数（M）</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th><th>T17</th><th>T18</th><th>T19</th><th>Avg</th></tr>
</thead><tbody>
<tr><th colspan="26">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>75.632</td><td>60.747</td><td>43.366</td><td>37.445</td><td>52.339</td><td>49.770</td><td>43.820</td><td>27.148</td><td>5.682</td><td>42.615</td><td>16.455</td><td>40.758</td><td>65.000</td><td>48.122</td><td>17.509</td><td>66.717</td><td>46.711</td><td>15.891</td><td>33.587</td><td>30.842</td><td>32.750</td><td>31.938</td><td>42.186</td><td>0.541</td><td>0.541</td></tr>
<tr><th>FOREVER</th><td>64.486</td><td>66.199</td><td>32.646</td><td>51.645</td><td>49.597</td><td>53.223</td><td>43.820</td><td>61.022</td><td>14.881</td><td>72.821</td><td>16.579</td><td>46.103</td><td>78.000</td><td>75.770</td><td>4.140</td><td>65.467</td><td>50.570</td><td>9.623</td><td>30.016</td><td>33.662</td><td>36.067</td><td>33.370</td><td>40.830</td><td>0.541</td><td>0.541</td></tr>
<tr><th>CL-MoE</th><td>74.657</td><td>62.026</td><td>42.636</td><td>34.270</td><td>44.892</td><td>52.726</td><td>47.191</td><td>27.443</td><td>14.015</td><td>49.077</td><td>22.227</td><td>41.139</td><td>73.000</td><td>57.597</td><td>18.256</td><td>67.316</td><td>50.050</td><td>14.914</td><td>34.330</td><td>31.592</td><td>35.297</td><td>33.110</td><td>42.774</td><td>37.159</td><td>37.159</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>70.697</td><td>75.000</td><td>41.147</td><td>33.413</td><td>48.522</td><td>49.802</td><td>51.685</td><td>41.523</td><td>21.111</td><td>82.667</td><td>24.492</td><td>44.202</td><td>44.000</td><td>77.056</td><td>20.699</td><td>66.092</td><td>55.738</td><td>25.223</td><td>34.455</td><td>39.755</td><td>34.379</td><td>37.241</td><td>46.620</td><td>140.771</td><td>140.771</td></tr>
<tr><th>PaRSP</th><td>73.346</td><td>58.904</td><td>37.123</td><td>40.064</td><td>50.786</td><td>42.467</td><td>40.449</td><td>22.730</td><td>15.530</td><td>46.615</td><td>16.883</td><td>38.743</td><td>44.000</td><td>69.983</td><td>21.751</td><td>66.417</td><td>54.088</td><td>20.609</td><td>30.975</td><td>31.218</td><td>37.562</td><td>32.655</td><td>43.888</td><td>0.000</td><td>134.195</td></tr>
<tr><th>OSFT</th><td>73.907</td><td>66.096</td><td>38.700</td><td>35.723</td><td>56.989</td><td>53.277</td><td>53.933</td><td>33.921</td><td>45.833</td><td>71.385</td><td>24.941</td><td>43.894</td><td>44.000</td><td>60.880</td><td>23.278</td><td>66.067</td><td>51.709</td><td>23.982</td><td>36.765</td><td>36.774</td><td>34.326</td><td>36.196</td><td>45.937</td><td>97.238</td><td>113.766</td></tr>
<tr><th>SpaRTA</th><td>73.596</td><td>60.395</td><td>42.465</td><td>35.552</td><td>51.505</td><td>45.899</td><td>46.067</td><td>26.868</td><td>28.301</td><td>62.410</td><td>21.327</td><td>40.921</td><td>44.000</td><td>39.120</td><td>24.601</td><td>66.742</td><td>45.942</td><td>21.328</td><td>36.591</td><td>33.708</td><td>34.271</td><td>34.520</td><td>43.261</td><td>10.402</td><td>10.402</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="26">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>86.312</td><td>72.061</td><td>45.827</td><td>41.170</td><td>56.989</td><td>53.606</td><td>56.180</td><td>42.002</td><td>18.056</td><td>62.923</td><td>27.650</td><td>47.654</td><td>72.000</td><td>65.076</td><td>14.116</td><td>67.516</td><td>51.110</td><td>26.576</td><td>37.849</td><td>32.617</td><td>29.905</td><td>33.210</td><td>48.972</td><td>1.090</td><td>1.090</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>85.821</td><td>72.003</td><td>46.777</td><td>42.709</td><td>54.839</td><td>51.463</td><td>50.562</td><td>41.523</td><td>18.056</td><td>62.154</td><td>26.260</td><td>47.279</td><td>75.000</td><td>57.597</td><td>18.324</td><td>68.791</td><td>50.680</td><td>23.557</td><td>37.125</td><td>32.358</td><td>30.414</td><td>33.023</td><td>48.072</td><td>77.930</td><td>77.930</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

## 10%训练：五域主表

<table>
<thead>
<tr><th rowspan="2">方法</th><th>自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th>金融</th><th colspan="4">交通</th><th rowspan="2">五域宏 F1 ↑</th><th rowspan="2">新增参数量（M）↓</th><th rowspan="2">S19可训练参数（M）</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th><th>T17</th><th>T18</th><th>T19</th><th>Avg</th></tr>
</thead><tbody>
<tr><th colspan="26">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>73.069</td><td>64.722</td><td>48.004</td><td>41.141</td><td>73.118</td><td>53.813</td><td>48.315</td><td>27.586</td><td>28.373</td><td>39.465</td><td>18.802</td><td>45.294</td><td>52.000</td><td>78.680</td><td>23.549</td><td>66.142</td><td>57.157</td><td>24.602</td><td>35.266</td><td>38.272</td><td>36.678</td><td>37.189</td><td>47.462</td><td>0.541</td><td>0.541</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>68.633</td><td>60.102</td><td>26.876</td><td>32.247</td><td>34.344</td><td>27.608</td><td>24.719</td><td>17.805</td><td>0.000</td><td>40.000</td><td>15.513</td><td>30.556</td><td>4.199</td><td>73.536</td><td>18.554</td><td>84.259</td><td>60.936</td><td>26.436</td><td>30.824</td><td>41.372</td><td>36.329</td><td>37.700</td><td>44.852</td><td>140.771</td><td>140.771</td></tr>
<tr><th>PaRSP</th><td>76.399</td><td>77.047</td><td>47.579</td><td>64.043</td><td>69.489</td><td>56.328</td><td>57.303</td><td>31.609</td><td>14.015</td><td>65.231</td><td>32.179</td><td>55.291</td><td>73.000</td><td>86.091</td><td>25.416</td><td>78.011</td><td>64.854</td><td>25.728</td><td>41.268</td><td>39.827</td><td>37.427</td><td>39.601</td><td>52.375</td><td>0.000</td><td>—</td></tr>
<tr><th>OSFT</th><td>73.470</td><td>74.503</td><td>51.170</td><td>62.012</td><td>68.683</td><td>53.235</td><td>53.933</td><td>34.195</td><td>41.667</td><td>76.462</td><td>31.360</td><td>55.061</td><td>63.000</td><td>71.540</td><td>16.390</td><td>81.459</td><td>59.176</td><td>27.146</td><td>37.818</td><td>40.095</td><td>35.208</td><td>38.408</td><td>50.652</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>75.554</td><td>74.371</td><td>53.891</td><td>58.841</td><td>72.984</td><td>50.214</td><td>51.685</td><td>32.184</td><td>25.476</td><td>68.513</td><td>28.102</td><td>53.610</td><td>80.000</td><td>76.244</td><td>20.869</td><td>80.685</td><td>61.745</td><td>26.824</td><td>37.288</td><td>39.580</td><td>34.170</td><td>37.766</td><td>51.100</td><td>10.402</td><td>10.402</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th colspan="26">Qwen2-VL-2B-Instruct</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>83.825</td><td>68.294</td><td>40.804</td><td>41.269</td><td>63.710</td><td>53.366</td><td>49.438</td><td>29.105</td><td>12.500</td><td>64.205</td><td>24.935</td><td>45.392</td><td>62.000</td><td>77.191</td><td>25.925</td><td>81.959</td><td>63.844</td><td>33.087</td><td>35.139</td><td>39.664</td><td>37.634</td><td>38.120</td><td>52.854</td><td>1.090</td><td>1.090</td></tr>
<tr><th>FOREVER</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>CL-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SMoLoRA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PASs-MoE</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>PaRSP</th><td>85.419</td><td>77.193</td><td>50.556</td><td>69.018</td><td>60.887</td><td>57.609</td><td>59.551</td><td>41.092</td><td>12.500</td><td>71.641</td><td>28.680</td><td>57.091</td><td>76.000</td><td>78.613</td><td>26.128</td><td>83.858</td><td>65.224</td><td>35.408</td><td>41.702</td><td>40.794</td><td>35.809</td><td>39.834</td><td>56.595</td><td>0.000</td><td>514.742</td></tr>
<tr><th>OSFT</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SpaRTA</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>Ours（待新候选）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
</tbody></table>

SpaRTA 10%（LLaVA）的末态19任务已完整核验；重启前历史评估记录缺失，未据此补造前向迁移、后向迁移等历史指标。[末态核验](summary/baseline_reboot_20261102/SPARTA_FINAL19_VERIFIED.json)

Qwen PaRSP 10% 的末态19任务已完整核验；S0–S11历史评估缺失，迁移指标尚不完整。[末态核验](summary/baseline_reboot_20261102/QWEN_FINAL19_VERIFIED.json)

M表示百万参数，不是MB。S19可训练参数按训练末阶段requires_grad参数元素数统计；不代表所有元素均有非零更新，也不代表全程固定值。1%取各自原始训练记录，未用10%代填。[1%记录](summary/shared_initial_and_parameter_audit_20261103/TRAINABLE_PCT01.json) · [10%记录](summary/shared_initial_and_parameter_audit_20261103/TRAINABLE_S19_RECORDS.json)

新增参数量按末态模型权重系数相对基座的净增量统计；OSFT包含作为buffer保存的固定SVD权重因子，不等于可训练参数量；不包含优化器、协方差等训练状态或精度转换增加的字节。未核验项保留“—”。[参数依据](summary/shared_initial_and_parameter_audit_20261103/ADDED_PARAMETERS.json)

1%已有F1的9个基线组合，其新增参数与S19可训练参数已逐项补齐。[checkpoint统计依据](summary/parameter_completion_20261104/FINAL_ADDED_COUNTS.json)


PASs-MoE 10% LLaVA最终S19完整测试已核验；重启前历史评估缺失，不代表完整历史指标已恢复。[末态依据](summary/baseline_reboot_20261102/PAS_FINAL19_VERIFIED.json)

## 观测结论

**已观测：在已测1%多领域持续学习设置中，动态组合相对固定组合的完整测试净收益较小。** CL-MoE同一检查点、同一专家集合的21454题对照中，动态五域宏token F1为42.774，固定为42.689，相差0.085个百分点；19998题F1相同，动态更好810题，固定更好646题。固定在自然、医疗、地理略高，动态在金融、交通略高，宏差不能解释为每域等效。PASs-MoE已有完整对照为动态46.620、固定46.332，但PAS包含折叠与精度变化，不能将差异全部归因于路由。[CL完整核验](summary/routing_mechanism_20261104/CL_FULL_VERIFIED.json)

**已观测：专家方向不同，但路由变化对最终答案排序的影响较小。** CL的80题历史TRAIN读数中，专家输出余弦中位数0.226，路由相对均匀权重尺度的RMS变化中位数约26%。另在40题TRAIN、相同真实答案前缀的三臂诊断中，动态减固定的最终中心化词表分数L2变化，为固定减关闭专家的3.56%；364个答案预测位置分别有3处、71处最高分token改变。这是输出端作用较小的机制证据，不是“路由权重恒定”，也不等于自由生成答案必然相同。[机制记录](summary/routing_mechanism_20261104/OBSERVATION.md)

**已观测：单专家低分存在作用强度混杂，不能直接证明组合互补。** 同40道TRAIN题，关闭专家F1为17.63，固定全专家41.73。八个单槽位保留原系数时为23.78–27.78；分别提高到原全组合的层系数总量后，八个均提升11.93–19.02个百分点，七个超过全固定，另一个低2.02个百分点。反过来，保持全专家组成、只把系数缩为1/8，F1降至23.78。可见这批题的收益不必依赖全专家共同启用，作用强度本身已能造成大幅变化。系数总量匹配不等于输出范数匹配；同编号跨层槽位不等于语义对齐专家。[完整干预核验](summary/routing_mechanism_20261104/MASS_PAIRED_ANALYSIS.json)

**已确认：输出端的小扰动在预定TEST样本复现。** 使用与完整TEST相同的固定系数，每域按题目ID哈希选择20题，共100题。动态减固定的中心化词表分数L2变化为固定减关闭专家的3.50%；864个受测答案位置分别有10处、175处最高分token改变。五域相对L2均在2.29%–5.60%，不是仅由池化平均造成。这里仍使用真实答案前缀，不将其当作自由生成F1。[TEST确认](summary/routing_mechanism_20261104/HELDOUT_VERIFIED_ANALYSIS.json)

**已观测：提高整体作用强度的收益，大部分固定组合也能获得。** 同100道TEST题，动态和固定的系数均放大2倍后，生成F1分别由40.627、40.526升至42.876、42.604；动态相对固定的优势由0.101升至0.272个百分点，仅扩大0.171个百分点。动态放大后20题改善、14题退化；固定18题改善、14题退化，不能只报告正向变化。这支持整体强度与选择策略应分开分析，不证明数据受限是成因，也不证明应优先加权最新专家。[四臂核验](summary/routing_mechanism_20261104/STRENGTH_VERIFIED.json)
