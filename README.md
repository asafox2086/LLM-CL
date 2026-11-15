# 多领域、数据受限的多模态持续学习

1%与10%分别使用1313与13105条训练样本；均评估五域19任务、完整21454条TEST。以下为训练末态S19的原token F1；基线沿原方法、原推理图报告。域内Avg按题数加权，五域宏对五域等权；“—”表示尚未完成或核验，不是0分。

## 1%训练：五域主表

<table>
<thead>
<tr><th rowspan="2">方法</th><th>自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th>金融</th><th colspan="4">交通</th><th rowspan="2">五域宏 F1 ↑</th><th rowspan="2">新增参数量（M）↓</th><th rowspan="2">S19可训练参数（M）</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th><th>T17</th><th>T18</th><th>T19</th><th>Avg</th></tr>
</thead><tbody>
<tr><th colspan="26">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>0.573</td><td>0.000</td><td>0.200</td><td>0.040</td><td>0.067</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.026</td><td>0.041</td><td>0.056</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.000</td><td>0.276</td><td>15.435</td><td>6.976</td><td>12.758</td><td>10.330</td><td>2.247</td><td>0.000</td><td>893.676</td></tr>
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
<tr><th>CL-MoE</th><td>75.770</td><td>70.863</td><td>48.190</td><td>45.187</td><td>65.726</td><td>55.230</td><td>51.685</td><td>28.592</td><td>45.833</td><td>58.923</td><td>24.280</td><td>48.350</td><td>72.000</td><td>78.714</td><td>23.889</td><td>66.717</td><td>57.697</td><td>24.057</td><td>38.058</td><td>35.744</td><td>35.293</td><td>36.182</td><td>48.411</td><td>37.159</td><td>37.159</td></tr>
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

CL-MoE 10% LLaVA末态已核验；历史S0缺失，未补造迁移指标。[末态依据](summary/baseline_reboot_20261102/CL_FINAL19_VERIFIED.json) · [参数依据](summary/baseline_reboot_20261102/CL_PCT10_PARAMETERS.json)

FullFT 1% LLaVA报告实际S19 BF16末态输出；训练使用FP32参数/BF16 AMP。三次恢复来源及最终21454题已核对，进程在测试后的资源日志解析阶段失败。该行不代表完整历史迁移指标已核验，也未分离最终精度转换的影响。[末态分数](summary/fullft_final_20261105/FINAL_OUTPUT_CLOSURE.json) · [恢复审计](summary/fullft_final_20261105/S18_CONTINUATION_AUDIT.json) · [参数依据](summary/fullft_final_20261105/PARAMETERS.json)

## 观测结论

**已观测：在已测1%多领域持续学习设置中，动态组合相对固定组合的完整测试净收益较小。** CL-MoE同一检查点、同一专家集合的21454题对照中，动态五域宏token F1为42.774，固定为42.689，相差0.085个百分点；19998题F1相同，动态更好810题，固定更好646题。各领域的收益方向不同，不能将小宏差解释为每域等效。PASs-MoE完整对照为动态46.620、固定46.332，但其固定化包含折叠与精度变化，不能把差异全部归因于路由。[CL完整核验](summary/routing_mechanism_20261104/CL_FULL_VERIFIED.json)

**已观测：跨域动态收益较小的结论依赖固定参照，尚不能直接归结为“多领域使路由失效”。** 在相同1280题训练预算、80次更新的同域／跨域持续学习对照中，共同医学目标任务完整228题TEST上，动态相对均匀固定组合的收益分别为2.193、0.439个百分点；换成各自TRAIN问题上的平均路由权重作为固定组合后，收益分别为0.439、0.658个百分点，排序反转。两种固定系数均未按TEST选择，TRAIN均值也不是最优固定组合。该对照使用LLaVA-OneVision-Qwen2-0.5B、单种子；不能外推到全部领域、算法或数据比例。[完整阶段分数](summary/domain_cl_factorial_20261107/VERIFIED_BD_STAGE_T02.json) · [固定参照敏感性](summary/routing_history_intervention_20261107/EVIDENCE_SYNTHESIS_20261111.json)

**已观测：冻结跨域训练后的专家，重新学习组合能够恢复旧任务表现。** 固定基座及全部晚期专家，用同样256条旧任务TRAIN、64次更新分别学习动态组合与固定组合，完整228题TEST上得到76.754和72.588，动态高4.167个百分点，图像簇配对95%区间为[0.628，7.675]；原动态为71.491。固定组合已明显离开均匀权重，但仍不能称最优固定上界。该结果说明本次原动态表现没有充分体现重新学习组合可以达到的表现；旧TRAIN也可能向路由学习知识，因此不证明专家完整保留旧知识，也不是无回放CL方法成绩。[校准对照](summary/routing_recalibration_followup_20261113/SYNTHESIS.json)

**已识别的局部原因：推理时混入历史平均权重，会压制已校准路由的收益。** 原实现训练时只用当前输入的路由，推理时使用“50%当前路由＋50%历史任务平均权重”。冻结专家及重训后的原结构路由，仅去掉历史项，跨域组完整228题F1从73.684升至76.316，增加2.632个百分点，图像簇配对95%区间为[0.441，4.915]。但同域也有此现象；回到未经校准的原模型，去掉历史混合后，跨域专家更新仍使动态相对均匀固定的收益下降2.193个百分点。因此历史混合是本次校准收益受限的一个原因，尚不足以解释原始跨域收益下降，更不是已证明的多领域特有机制。[历史项干预](summary/routing_history_after_refit_20261113/VERIFIED_PAIRED.json) · [原模型阶段干预](summary/routing_history_expert_stage_20261113/VERIFIED_PAIRED.json)

**已观测：同域任务顺序训练使多个专家路径共同退化，但尚未证实它专门削弱动态选择的额外收益。** 在相同医学领域、完全相同1280条TRAIN与80次更新、相同五个优化器及合并边界下，只改变样本的时序组织。完整医学2094题TEST中，混合顺序／任务顺序的动态F1为59.279／54.576，均匀固定为57.110／52.622；八条单专家路径全部下降3.678–5.203点，说明损失在不使用动态选择的路径中也存在。动态相对固定的收益为2.168／1.954点，变化−0.215，图像簇配对95%区间[−1.307，0.895]，不能说顺序已证明专门损害路由。两组专家共同改善题占改善并集均约89%，这种重叠在同域混合训练中已存在。该全医学结果不能套到共同T02：其完整228题上两组动态F1相同。单种子对照尚不证明跨领域原因或共同退化的具体更新机制。[完整同域时序对照](summary/domain_cl_factorial_20261107/VERIFIED_AB_20261114.json)

**已观测：跨域降低动态收益没有在第二目标任务复现，不能将多领域本身视为充分原因。** 医学完整228题四组已齐：同域混合、同域顺序、跨域混合、跨域顺序的动态减均匀固定收益依次为0.000、2.193、0.877、0.439点；跨域效应随训练顺序改变方向。交通完整500题、每组1280条独立TRAIN与80次更新、相同辅助任务数的复核中，顺序训练的同域／跨域动态收益为−0.080／0.218点；TRAIN均值固定参照下为0.113／0.385点。跨域减同域分别为+0.298、+0.272点，图像簇95%区间均包含零，没有复现医学的收益压缩。交通跨域动态绝对F1虽低2.874点，固定组合也下降，不能将绝对性能降低当成动态选择额外收益降低。两目标均为单种子，尚不能断言跨域无影响或有稳定正效应。[医学四组完整结果](summary/domain_cl_factorial_20261107/VERIFIED_ABCD_RECOVERY_20261113.json) · [交通四组完整结果](summary/traffic_factorial_matched_tasks_20261113/VERIFIED_ABCD.json)

**正在证明：医学中的收益变化由哪一次任务转换触发，以及专家更新与历史组合如何共同影响它。** 原医学轨迹中，同域／跨域组第四阶段的动态减均匀固定收益均为1.535点，最后阶段分别变为2.193、0.439点；这只定位差距形成的阶段，不证明两组此前模型状态相同。从同一个第四阶段检查点替换最后256条任务数据的配对实验已实际启动，保持相同16次更新，并同时评估两种固定参照。任务替换仍伴随回答形式变化；数据量因素及跨种子稳定性尚未验证。[原阶段读数](summary/domain_cl_factorial_20261107/TRAIN_SUPERVISION_STAGE_READOUT.json)
