# 1%训练：四域主表

## 表1：16个子任务的最终Answer F1

<table>
<thead>
<tr><th rowspan="2">方法</th><th colspan="1">自然</th><th colspan="11">医学</th><th colspan="5">地理／遥感</th><th colspan="1">金融</th><th rowspan="2">四域宏<br>F1 ↑</th></tr>
<tr><th>TextVQA</th><th>Mod.</th><th>Pos.</th><th>Organ</th><th>Size</th><th>Abn.</th><th>Qty.</th><th>Plane</th><th>Shape</th><th>Color</th><th>KG</th><th>Avg</th><th>R/U</th><th>Exist</th><th>Count</th><th>Compare</th><th>Avg</th><th>FinMME</th></tr>
</thead>
<tbody>
<tr><th colspan="20">LLaVA-OneVision-Qwen2-0.5B-OV</th></tr>
<tr><th>FullFT（朴素全量）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
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
<tr><th>SeqFT-LoRA（q/v-r8）</th><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td><td>—</td></tr>
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

各小类均为统一四域终点R[16,t]的原始Answer F1（0–100）。Avg按该域TEST题数加权；四域宏对四个域等权。AP16对16个终点任务分数等权，不能与四域宏互换。FWT/BWT沿原定义限制在16任务；保留报告但研究分析聚焦新习得。—表示尚未完成该四域口径核验，不是0。历史五域和阶段刚习得分数不得直接填入。

## 表2：持续学习与部署开销

| 方法 | AP16 ↑ | FWT ↑ | BWT ↑ | 总部署参数<br>M ↓ | 新增部署参数<br>M ↓ | 完整权重<br>MiB ↓ | 权重增量<br>MiB ↓ | GPU峰值†<br>MiB ↓ | 首词延迟<br>s ↓ | 生成速度<br>tok/s ↑ | 常驻适配器<br>数量 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **LLaVA-OneVision-Qwen2-0.5B-OV** |  |  |  |  |  |  |  |  |  |  |  |
| FullFT（朴素全量） | — | — | — | — | — | — | — | — | — | — | — |
| SeqFT-LoRA（q/v-r8） | — | — | — | — | — | — | — | — | — | — | — |
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
| SeqFT-LoRA（q/v-r8） | — | — | — | — | — | — | — | — | — | — | — |
| FOREVER | — | — | — | — | — | — | — | — | — | — | — |
| CL-MoE | — | — | — | — | — | — | — | — | — | — | — |
| SMoLoRA | — | — | — | — | — | — | — | — | — | — | — |
| PASs-MoE | — | — | — | — | — | — | — | — | — | — | — |
| PaRSP | — | — | — | — | — | — | — | — | — | — | — |
| OSFT | — | — | — | — | — | — | — | — | — | — | — |
| SpaRTA | — | — | — | — | — | — | — | — | — | — | — |
| Ours | — | — | — | — | — | — | — | — | — | — | — |

† GPU峰值为统一四域固定输入、同卡同生成长度下的推理allocated峰；reserved峰、实际dtype、训练峰、完整最终checkpoint与来源在证据表另存。MiB=2²⁰字节，参数M=10⁶；新增参数/权重增量相对同backbone。部署量包括推理必需的共享权重、专家/适配器及必要状态，不含optimizer，不用训练坐标或checkpoint容器体积替代。
