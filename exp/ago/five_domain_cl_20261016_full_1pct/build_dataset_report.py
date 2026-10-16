"""Five original multimodal sources, traceable examples and root README."""
import html
import importlib.util
import json
import shutil
from pathlib import Path
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
ID=HERE.name
DATA=ROOT/'data/FiveDomainCL'/ID
DATA.mkdir(parents=True,exist_ok=True)
SAMPLES=ROOT/'sample'/ID;SAMPLES.mkdir(parents=True,exist_ok=True)
spec=importlib.util.spec_from_file_location('sources',HERE/'prepare_data.py')
n=importlib.util.module_from_spec(spec);spec.loader.exec_module(n)
sources=[
 ('TextVQA','自然／通用',34602,347,5000,'Amanpreet Singh 等 · CVPR 2019','https://textvqa.org/',
  'CoIN（NeurIPS 2024）；SMoLoRA（ICCV 2025）','官方 train／validation；公开 validation 作完整测试池'),
 ('SLAKE','医学',9835,99,2094,'Bo Liu 等 · ISBI 2021','https://www.med-vqa.com/slake/',
  'Uncertainty-Driven Expert Control（ICCV 2025）','官方 train／test；保留中英文、OPEN／CLOSED，val 2,099 不用'),
 ('RSVQA-LR','地理／遥感',57223,573,10004,'Sylvain Lobry 等 · TGRS 2020','https://zenodo.org/records/6344334',
  'GeoChat（CVPR 2024）；LHRS-Bot（ECCV 2024）；RSAdapter（TGRS 2024）','官方 active 掩码 train／test；val 10,005 不用，保留原数值答案'),
 ('MME-Finance','金融',935,10,236,'Ziliang Gan 等 · ACM MM 2025','https://github.com/HiThink-Research/MME-Finance',
  '本次未补足独立多篇复用核验，见调研记录','英文 MMfin 原始 1,171 QA；本研究按图表组 seed42 做 80/20，无额外验证池'),
 ('AQUA','艺术文化',69812,699,4912,'Noa Garcia 等 · ECCV VISART Workshop 2020','https://github.com/noagarcia/ArtVQA',
  'CL-CrossVQA（WACV 2025）；ArtQuest（WACV 2024）','官方 train／test；知识、视觉 QA 全保留并输入原画作，val 5,124 不用'),
]
examples=[next(n.native_parquet('TextVQA','train')),next(n.native_slake('train')),next(n.native_rsvqa('train')),
          next(x for x in n.native_mme()['train'] if x['metadata']['image_type']=='candlestick'),next(n.native_aqua('train',True))]
examples[-1]['images']=[str(Path('data/AQUA/selected_images')/examples[-1]['metadata']['original_image'])]
examples[-1]['metadata']['conditioning']='original painting; knowledge QA'
samplepage=['# 五领域图文问答原始样例','', '问题、参考和图像均来自源数据；用于展示，不代表当前1%训练选择。','']
sample_rows=[];audit=[]
for i,(source,example) in enumerate(zip(sources,examples),1):
 name,domain,tr,small,te,origin,url,reuse,split=source
 original=ROOT/example['images'][0]
 with Image.open(original) as im:
  if im.format=='TIFF':
   destination=SAMPLES/(name+'.png');im.save(destination)  # lossless display copy; original TIFF unchanged
  else:
   destination=SAMPLES/(name+('.jpg' if im.format=='JPEG' else '.png'));shutil.copyfile(original,destination)
 image=str(destination.relative_to(ROOT));slug=name.lower()
 audit.append({'task':i,'dataset':name,'modality':'image+text question','domain':domain,
               'full_train':tr,'pct01_train':small,'full_test':te,'source':url,'introduction':origin,
               'reuse':reuse,'split':split,'sample':example,'sample_image':image})
 def esc(value,limit=160):
  value=str(value);return html.escape(value[:limit]+('…（节选）' if len(value)>limit else '')).replace('|','&#124;').replace('\n','<br>')
 sample_rows.append(f'| [T{i} · {name}](data/FiveDomainCL/{ID}/samples.md#{slug}) | {esc(example["question"])} | {esc(example["answer"])} | <img src="{image}" width="180" alt="{name}原始样例"> |')
 samplepage += [f'<a id="{slug}"></a>',f'## T{i} · {name}','',f'源 ID：`{example["id"]}`。','',example['question'],'','```json',json.dumps(example['answers'],ensure_ascii=False,indent=2),'```','']
 if example.get('context'):samplepage += ['<details><summary>原始上下文</summary>','',example['context'],'','</details>','']
 samplepage += [f'![{name}](../../../{image})','']
(DATA/'samples.md').write_text('\n'.join(samplepage))
(DATA/'dataset_audit.json').write_text(json.dumps({'experiment_id':ID,'task_count':5,'count_unit':'QA pairs','datasets':audit},ensure_ascii=False,indent=2)+'\n')
root=ROOT/'README.md';old=root.read_text();a=old.index('## 方法 × 方向二维表');b=old.index('\n## ',a+5);papers=old[a:b]
text=['# LLMCL：小模型的五领域多模态持续学习','',
'当前采用 **5 个领域、5 个图文问答任务、5 个数据集**，每领域只保留一个来源；不再单独寻找纯文本数据集，也不删除图片造出文本任务。基座 **Qwen2-VL-2B-Instruct**；seed42；每任务一轮；不做验证；**不存阶段／优化器 checkpoint，全部训练结束保存最终模型与方法参数**。','',papers,'',
'## 当前五任务协议','',f'独立实验目录：`exp/{ID}`；新报告：`summary/{ID}`；五任务数据：`data/FiveDomainCL/{ID}`。旧十任务预测和统计不复用。五源原始资产已到达，固定清单、源文件哈希和新任务 GPU 预检通过后再进入正式训练；实际进度见 [运行状态](summary/'+ID+'/progress.md)。','',
'| 条件 | 训练 | 测试 | GPU |','|---|---|---|---|',
'| full | 各来源完整训练池 | 完整公开有标注测试池 | GPU3，方法串行 |',
'| pct01 | 同训练池按原始题型分层取 ceil(1%) | 与 full 相同文件和 SHA256 | GPU1／2，方法并行 |','',
'GPU0 保留。九方法：SeqFT-LoRA 控制、SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。两组从相同基座独立开始，按 T1→T5 训练。先评测全部五任务，每完成一个任务再次评测全部五任务，输出 **6×5** 矩阵。统一报告 EM 正确数／总数、Token F1、ROUGE-L，再计算 AP、ForgetRate、FWT、BWT；统一词面分数不冒称各数据集原生指标。','',
'## 数据集：来源、数量与实际使用论文','',
'所有任务输入均为原始图像＋原始问题／上下文；保留原始是非、选择、开放式、计数等题型和答案。表中单位都是 QA 对。单任务不足十万，五任务训练总量 **172,407**，1% 组 **1,728**，共同测试 **22,246**；总量与单个来源数量分开报告。','',
'| 任务／领域 | 数据集（均为多模态） | 原始发表／资源 | 全量训练 | 1%训练 | 完整测试 | 划分与配置 | 谁实际使用 |','|---|---|---|---:|---:|---:|---|---|']
for i,(name,domain,tr,small,te,origin,url,reuse,split) in enumerate(sources,1):text.append(f'| T{i} · {domain} | {name} | [{origin}]({url}) | {tr:,} | {small:,} | {te:,} | {split} | {reuse} |')
text += ['',
'复用论文、发表链接、正文位置和适配限制见 [多领域数据集调研](data/MULTIDOMAIN_DATASET_SURVEY_20261016.md)。注意：后续使用论文在 ICCV 发表，不等于数据集首发 ICCV；AQUA 的原始发表为 workshop。MME-Finance 当前仍缺少本次核验的多篇独立复用证据，不把它写成已满足该条件。','',
'1% 按源训练池原始题型比例，用最大余数分配 ceil(0.01×N) 名额，seed42 固定哈希选 ID。所有方法使用相同清单和训练顺序。RSVQA-LR 遵守作者 active 掩码，保留原始数字而非自行分箱。AQUA 两类 QA 在同一个图文任务训练，不再将知识题去图构造文本任务。MME-Finance 唯一采用本研究图表分组 holdout；它不是作者官方 train/test。','',
'## 原始数据 sample','',
'下面是源数据示例，长回答只作展示节选。[完整问题／参考／上下文](data/FiveDomainCL/'+ID+'/samples.md)；[机器审计](data/FiveDomainCL/'+ID+'/dataset_audit.json)。JPEG／PNG 复制原字节，RSVQA 原 TIFF 仅另存无损 PNG 用于页面展示；训练仍读取原 TIFF。','',
'| 数据集 | 原始问题 | 原始参考答案 | 图像 |','|---|---|---|---|',*sample_rows,'',
'## 参数、日志与后续评定','',
f'每个方法结束后保存 `exp/{ID}/runs/<condition>/<method>/<run_id>/final_parameters/`：完整模型 state_dict（包含冻结旧专家和合并后的权重）、方法和路由状态、基座文件哈希、模型／处理器配置、任务清单及运行代码。`artifact_manifest.json` 记录大小和 SHA256。它需要按自定义方法架构加载，正式后续评定前必须做新进程重载等价核验；不能直接当作标准 Transformers 模型目录。','',
'两组所有方法继续记录逐步 loss、裁剪前后梯度、学习率、耗时；每5秒及阶段边界记录 CPU RSS/PSS/USS 与 GPU allocated/reserved/NVML。训练前、最后阶段训练后释放优化器的具体占用、差值与峰值单列；最终参数导出在该内存测量之后，不污染训练增长统计。保留逐题预测、完整矩阵和最终指标，禁止混入旧运行。','',
f'[实验说明](exp/{ID}/README.md) · [执行策略](exp/{ID}/execution_policy.json) · [最新协议](rules/015_five_multimodal_final_parameters_20261016.md) · [作者方法代码](code/README.md)。旧报告放 `summary/ago/`；旧实验放 `exp/ago/`。主仓库 `/data1/liyapeng/program/LLMCL`，Tailscale SSH 端口8666。','']
root.write_text('\n'.join(text))
print('Updated five-domain root README and original examples')
