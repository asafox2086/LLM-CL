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

**已观测结论一：已测PASs-MoE 1%存在部署冗余——不必原样保留动态多专家结构，也能取得接近的总体质量。** 在LLaVA、五域19任务的完整21454题上，将训练后的专家增量转为固定稠密表示，五域宏F1从46.620变为46.332，必要权重载荷减少23.96%，同卡五个固定输入窗口的allocated显存峰下降24.84%–26.81%。这里“部署冗余”指该检查点存在更紧凑、以0.288个百分点宏F1损失为代价的表示，并非严格等价或所有专家都可删除。它说明：用于持续学习的多专家结构，不一定要原样保留到部署阶段；专家学到的增量仍可通过固定表示参与推理。该替换包含固定混合、折叠与精度变化，尚不能把效果单独归因于取消动态路由，也没有证明训练时无需专家。[质量与资源证明](summary/pas_static_final_confirmation_20261029/FINAL_FINDINGS_20261030.md)

**已观测结论二：这个模型的总体保分掩盖了领域差异，不能据此认定简化后的模型能无差别替代原模型。** 总体89.00%的题F1不变，但交通域83.23%的题发生双向变化，得失相抵后域均分几乎不变；遥感域则由原动态表示领先1.519个百分点。因此，多领域持续学习的部署简化需要同时核对各域损失与逐题变化，单一宏平均分不足以说明哪些领域和请求受到影响。这是完整配对支持的具体结论，不是动态路由无用的证明。[完整配对观测](summary/pas_static_final_confirmation_20261029/OBSERVATION_PAS_FIXED_DYNAMIC_PAIRED_20261031.md)

**正在证明：训练后动态组合专家的作用，是否在CL-MoE 1%中也能由固定组合近似承载。** 比较的是同一批专家、同一已训练检查点的动态组合与固定组合；固定系数后，专家输出仍随输入变化。已有80条历史TRAIN、168个投影的局部观测显示，固定组合与动态组合的输出差值平方和，占原专家增量平方和的比例，在PASs-MoE和CL-MoE中分别为0.010457%与2.952336%。这为完整干预提供了依据，但不是F1损失、可删除参数比例，也不说明新增方向与基础模型方向一致。CL-MoE完整21454题对照尚无结果，本次完整验证尚待执行，不能将局部近似写成整模型保分。[局部专家输出观测](summary/router_function_moments_20261030_v2/FINDINGS.md)

**尚未验证：部署冗余及其领域差异能否成为跨方法、跨1%/10%设置的共同观测。** 当前完整部署证据仅覆盖上述PASs-MoE检查点；真实端侧时延和能耗、非专家扩展方法的对应证据，以及数据受限或序贯学习是否造成该现象，均未证明。论文正在检验的是多领域知识的学习与保留，是否必须依赖同样复杂的部署结构；目前不能将单模型实例提升为持续学习的普遍规律。
