"""Refresh six-source documentation without overwriting the paper table."""
import html
import json
import shutil
from pathlib import Path
from PIL import Image
import prepare_data as n

ROOT=n.ROOT
ID=n.HERE.name
DATA=n.DEST
CATALOG=[
 ('TextVQA','自然／通用',34602,347,5000,'Amanpreet Singh等，CVPR 2019','https://textvqa.org/',
  'CoIN（NeurIPS 2024）、SMoLoRA（ICCV 2025）','官方train/validation；公开validation作测试',
  '自然照片＋问题：读取场景文字并结合图像作答'),
 ('SLAKE','医学',9835,99,2094,'Bo Liu等，ISBI 2021','https://www.med-vqa.com/slake/',
  'Uncertainty-Driven Expert Control（ICCV 2025）','官方train/test；双语；val2099不用',
  '医学影像＋中英问题；保留视觉题、知识题及OPEN/CLOSED'),
 ('RSVQA-LR','地理／遥感',57223,573,10004,'Sylvain Lobry等，TGRS 2020','https://rsvqa.sylvainlobry.com/',
  'GeoChat（CVPR 2024）、LHRS-Bot（ECCV 2024）、RSAdapter（TGRS 2024）','官方active掩码train/test；val10005不用；不自行分箱',
  '遥感图像＋问题：计数、存在性、比较与区域判断'),
 ('FinMME','金融',8864,89,2233,'Junyu Luo等，ACL 2025主会','https://aclanthology.org/2025.acl-long.1426/',
  'Beyond Real Weights（WACV 2026）实际评测；AgentFinVQA（2026预印本）实际评测',
  '源池11099；2条缺答案单列；11097有标注QA按原图SHA256分组seed42研究80/20，非官方划分',
  '真实金融图表＋问题、选项；单选、多选和数值题全保留'),
 ('AQUA','艺术／文化',69812,699,4912,'Noa Garcia等，ECCV VISART Workshop 2020','https://github.com/noagarcia/ArtVQA',
  'CL-CrossVQA（WACV 2025）、ArtQuest（WACV 2024）；具体子集见调研',
  '官方train/test；知识和视觉题都提供原画作；val5124不用','画作＋问题：视觉内容和艺术知识问答'),
 ('CODA-LM','交通／自动驾驶',20495,205,2123,'Kai Chen等，WACV 2025','https://github.com/DLUT-LYZ/CODA-LM',
  'MPDrive（CVPR 2025）实际评测；RoboTron-Drive（ICCV 2025）实际训练和评测',
  '作者English LLaVA Train/Test；源卡20495/2123；Val18455、Mini193不用',
  '真实道路图像＋作者指令：场景理解、区域理解、驾驶建议；区域题保留作者红框图'),
]


def main():
    DATA.mkdir(parents=True,exist_ok=True)
    samples=ROOT/'sample'/ID;samples.mkdir(parents=True,exist_ok=True)
    old=(ROOT/'README.md').read_text(encoding='utf-8')
    a=old.index('## 方法 × 方向二维表');b=old.index('\n## ',a+5);paper=old[a:b].rstrip()
    status_file=DATA/'preparation_status.json'
    ready=status_file.exists() and json.loads(status_file.read_text())['status']=='ready'
    actual=json.loads((DATA/'task_counts.json').read_text()) if ready else []
    examples=[next(n.native_parquet('TextVQA','train')),next(n.native_slake('train')),next(n.native_rsvqa('train')),
              n.native_finmme()['train'][0],next(n.native_aqua('train',True)),next(n.native_coda('test'))]
    examples[4]['images']=[str(Path('data/AQUA/selected_images')/examples[4]['metadata']['original_image'])]
    entries=[];sample_rows=[];details=['# 六领域原始图文问答示例','','原问题、选项、答案保留；本页示例不是1%采样清单。','']
    lines=['# LLMCL：小模型的六领域多模态持续学习','',
      '**六个领域、六个图文问答任务、每领域一个数据集**。金融替换为FinMME；第六领域新增交通／自动驾驶CODA-LM。','',paper,'',
      '## 当前六任务协议','',f'独立实验：`{ID}`；代码／参数在`exp/{ID}/`，日志报告在`summary/{ID}/`，QA清单在`data/FiveDomainCL/{ID}/`（父目录名保留，内容是六任务）。旧五／十任务服务停用，不复用其预测。',
      '当前状态：'+('六任务清单已生成；GPU预检和正式运行以进度报告为准。' if ready else '下载／逐图审计中，正式六任务清单尚未冻结，不代表已开始训练。'),
      '基座Qwen2-VL-2B-Instruct；seed42；一轮；无验证。九方法：SeqFT-LoRA、SMoLoRA、PASs-MoE、Hyper-LLaVA、PaRSP、OSFT、SpaRTA、FV-guided Training、ALER。GPU0保留；GPU3跑全量；GPU1/2并行跑1%组，均完整测试。',
      '每个方法独立从基座开始，按T1→T6训练；初始及每阶段结束测全部六任务，输出**7×6矩阵**。保留逐题结果、EM正确数/总数、Token F1、ROUGE-L及AP、ForgetRate、FWT、BWT。统一词面指标不等于原生评分：FinMME涉及数值容差和多选；CODA-LM原文采用LLM评审，EM不是它的官方准确率。','',
      '## 数据集：来源、数量与复用依据','',
      '单位均为QA，每个来源训练QA低于十万。'+('全部来源已按完整文件计数并冻结清单。' if ready else 'FinMME已做完整文件与逐图审计；CODA-LM下载中，表中为作者English配置源卡数量，下载后与Parquet逐文件复核。')+'保留原始选择、是非、计数等构成。','',
      '| 任务／领域 | 数据集（图文） | 提出者／首发 | 全量训练 | 1%训练 | 完整测试 | 划分 | 独立实际使用 |',
      '|---|---|---|---:|---:|---:|---|---|']
    for i,(item,x) in enumerate(zip(CATALOG,examples),1):
        name,domain,tr,small,te,origin,url,reuse,split,visual=item
        if actual:tr,small,te=(actual[i-1][k] for k in ['full_train','pct01_train','full_test'])
        lines.append(f'| T{i} · {domain} | {name} | [{origin}]({url}) | {tr:,} | {small:,} | {te:,} | {split} | {reuse} |')
        with Image.open(ROOT/x['images'][0]) as im:
            target=samples/(name+('.jpg' if im.format=='JPEG' else '.png'))
            if im.format in ['JPEG','PNG']:shutil.copyfile(ROOT/x['images'][0],target)
            else:im.save(target)  # lossless display copy, never replace original TIFF
        image=str(target.relative_to(ROOT));anchor=name.lower()
        entries.append(dict(task=i,dataset=name,domain=domain,modality='image+text',full_train=tr,pct01_train=small,
            full_test=te,introduction=origin,source=url,reuse=reuse,split=split,visual_evidence=visual,sample=x,sample_image=image,
            count_status='frozen_manifest' if actual else 'file_audited' if name=='FinMME' else 'author_metadata' if name=='CODA-LM' else 'previous_native_file_audit'))
        def esc(v):
            v=str(v);return html.escape(v[:160]+('…（节选）' if len(v)>160 else '')).replace('|','&#124;').replace('\n','<br>')
        sample_rows.append(f'| [T{i} · {name}](data/FiveDomainCL/{ID}/samples.md#{anchor}) | {esc(x["question"])} | {esc(x["answer"])} | <img src="{image}" width="180" alt="{name}原始图像"> |')
        details += [f'<a id="{anchor}"></a>',f'## T{i} · {name}','',f'源ID：`{x["id"]}`。','',x['question'],'']
        if x.get('context'):details += ['原始选项／上下文：','',x['context'],'']
        details += ['参考答案：','',x['answer'],'',f'![{name}](../../../{image})','']
    totals=[sum(e[k] for e in entries) for k in ['full_train','pct01_train','full_test']]
    lines += ['',f'六任务合计（当前源文件／源卡口径，冻结后复核）：全量训练 **{totals[0]:,}**，1%训练 **{totals[1]:,}**，共同完整测试 **{totals[2]:,}**。',
      'FinMME的ID3698、9806金答案为空，原记录单独保留审计，不伪造、不以空串监督。作者仅发布名为train的评测池；本研究按图像字节SHA256分组，组哈希seed42排序，20%组留作测试。原选项作为输入，verified_caption/related_sentences只作来源元数据，不输入模型，避免泄漏答案。',
      '1%组按源题型分层，最大余数分配ceil(1%×N)，seed42哈希选ID。所有方法共享ID、顺序及测试SHA256；不自行删掉选择、分类、是非题。CODA-LM用作者原问题、答案和区域红框图，不自行造图或改写问答。','',
      '## 多模态依据','',
      '六个来源都有真实图像和问题文本，模型同时接收两种输入。**图文任务身份不等于每一道题都必须看图**：SLAKE知识题、AQUA知识题及语言偏置可能允许捷径。原始构成照常保留；若主张模型实际利用图像，需要额外的固定测试集遮图／错配图消融，当前未声称已做。','',
      '| 数据集 | 视觉来源与任务 |','|---|---|',*[f'| {x[0]} | {x[9]} |' for x in CATALOG],
      '', '原始发表、独立复用正文与限制见[六域多模态核验](data/MULTIMODAL_SIX_DOMAIN_AUDIT_20261016.md)。医学首发ISBI、遥感首发TGRS、艺术首发workshop，不把后续顶会复用写成首发顶会。','',
      '## 原始数据 sample','',f'[完整问题、选项、答案和原图](data/FiveDomainCL/{ID}/samples.md) · [机器审计](data/FiveDomainCL/{ID}/dataset_audit.json)。长答案在下表节选；旧样例不在本页展示。','',
      '| 数据集 | 原始问题 | 参考答案 | 原图 |','|---|---|---|---|',*sample_rows,'',
      '## 参数与资源记录','',f'各方法在全部六阶段结束后仅保存一次`exp/{ID}/runs/<condition>/<method>/<run_id>/final_parameters/`：完整最终模型（含冻结专家、合并权重）、方法与路由状态、配置、处理器、代码和SHA256。不存阶段、优化器、恢复checkpoint；参数不进Git。下游评定前必须新进程重载并验证输出等价。',
      '两组每个方法记录loss、裁剪前后真实梯度范数、CPU RSS/PSS/USS、GPU allocated/reserved/NVML；单列训练前、训练后具体占用、变化与峰值。训练后测量在释放优化器后、最终参数导出前。',
      f'[执行策略](exp/{ID}/execution_policy.json) · [协议](rules/016_six_multimodal_finmme_coda_20261016.md) · [进度](summary/{ID}/progress.md)。','']
    (ROOT/'README.md').write_text('\n'.join(lines).rstrip()+'\n',encoding='utf-8')
    (DATA/'samples.md').write_text('\n'.join(details).rstrip()+'\n',encoding='utf-8')
    n.write_json(DATA/'dataset_audit.json',dict(experiment_id=ID,task_count=6,count_unit='QA',formal_manifests_generated=ready,datasets=entries))


if __name__=='__main__':main()
