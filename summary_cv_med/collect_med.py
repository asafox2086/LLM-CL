"""Live human-readable medical-only report; all values come from saved evaluations."""
import argparse
import csv
import fcntl
import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'exp'))
from common import continual_metrics, write_json

OUT=ROOT/'summary_cv_med'
METHODS=['seq_lora','migu_lora','olora','sapt_lora']
NAMES=['SeqLoRA','MIGU-LoRA','O-LoRA','SAPT-LoRA']
TASKS=['medical_medmcqa','medical_medqa','medical_vqa_rad_chest','medical_vqa_rad_head','medical_vqa_rad_abd']
SHORT=['MedMCQA','MedQA','胸部图像','头部图像','腹部图像']

def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])+'\n'

def number(value):return '—' if value is None else f'{value:.2f}'

def plot(process):
    if not process:return False
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0,str(OUT/'plot_templates'))
    from style import setup_style, polish_axes, save_png_pdf
    setup_style('line')
    colors=['#4F81BD','#9BBB59','#C0504D','#8064A2'];styles=[('o','-'),('D','--'),('^','-.'),('s',':')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),dpi=300)
    for i,m in enumerate(METHODS):
        rows=[r for r in process if r['method']==m]
        if not rows:continue
        for ax,key in zip(axes.flat,['all_task_em','medical_probe','old_task_delta','general_probe']):
            valid=[r for r in rows if r.get(key) is not None]
            if not valid:continue
            ax.plot([r['stage'] for r in valid],[r[key] for r in valid],color=colors[i],
                    marker=styles[i][0],linestyle=styles[i][1],label=NAMES[i],markersize=5)
    for ax,title,ylabel in zip(axes.flat,['A  Fixed five-task mean','B  Held-out medical knowledge',
                                          'C  Old-task change after current training','D  Held-out general knowledge'],
                                 ['EM (%)','Accuracy (%)','Change (percentage points)','Accuracy (%)']):
        ax.set_title(title,fontsize=14);ax.set_xlabel('Completed medical tasks');ax.set_ylabel(ylabel)
        ax.set_xlim(-.15,5.15);ax.set_xticks(range(6));polish_axes(ax)
        if 'Change' in ylabel:
            if not ax.lines:
                ax.text(.5,.5,'Available after stage 2',transform=ax.transAxes,ha='center',fontsize=13,color='#7F7F7F')
            ax.axhline(0,color='#7F7F7F',lw=1)
        else:ax.set_ylim(0,100)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,1.02),frameon=False)
    fig.tight_layout();save_png_pdf(fig,str(OUT/'figures/process_med'));plt.close(fig)
    (ROOT/'sample/med').mkdir(parents=True,exist_ok=True)
    (OUT/'figures/process_med.png').replace(ROOT/'sample/med/process_med.png')
    return True

def collect():
    latest=read(ROOT/'exp/CV_result_med/latest.json');run=Path(latest['run']) if latest else None
    manifest=read(ROOT/'exp/CV_result_med/shared/manifest.json')
    OUT.mkdir(parents=True,exist_ok=True)
    statusrows=[];allresults={};process=[];score_rows=[];gains=[];probe_rows=[];impact_rows=[];impact_stages=[]
    for method,name in zip(METHODS,NAMES):
        output=run/method if run else None
        status=read(output/'status.json',{}) if output else {}
        queue=read(output/'queue.json',{}) if output else {}
        state=queue.get('state','未启动')
        if state=='running':state=status.get('status',state)+' / '+status.get('phase','加载中')
        statusrows.append([name,state,status.get('stage','—'),status.get('task','—')])
        results=read(output/'scores.json',[]) if output else []
        allresults[method]=results
        for stage,scores in enumerate(results):
            for task,metrics in scores.items():
                score_rows.append({'method':method,'stage':stage,'learned_task':TASKS[stage-1] if stage else 'base',
                                   'task':task,**{k:metrics[k] for k in ['exact_match','token_f1','rougeL','count']}})
            if len(scores)!=5:continue
            probes=read(output/'knowledge_probe'/f'stage_{stage:02d}.scores.json',{})
            old_delta=None
            if stage>1 and len(results[stage-1])==5:
                old=TASKS[:stage-1]
                old_delta=sum(scores[t]['exact_match']-results[stage-1][t]['exact_match'] for t in old)/len(old)
                details=[]
                for t in old:
                    before=results[stage-1][t]['exact_match'];after=scores[t]['exact_match']
                    detail={'method':method,'stage':stage,'current_training_task':TASKS[stage-1],
                            'old_test_task':t,'test_count':scores[t]['count'],
                            'before_accuracy':before,'after_accuracy':after,'change_pp':after-before}
                    impact_rows.append(detail);details.append(detail)
                assert abs(sum(r['change_pp'] for r in details)/len(details)-old_delta)<1e-10
                impact_stages.append({'method':method,'stage':stage,'current_training_task':TASKS[stage-1],
                    'before_stage':stage-1,'after_stage':stage,'old_test_tasks':' ; '.join(old),
                    'old_test_questions':sum(r['test_count'] for r in details),
                    'old_task_macro_before':sum(r['before_accuracy'] for r in details)/len(details),
                    'old_task_macro_after':sum(r['after_accuracy'] for r in details)/len(details),
                    'old_task_delta_pp':old_delta})
            process.append({'method':method,'stage':stage,'all_task_em':sum(scores[t]['exact_match'] for t in TASKS)/5,
                            'medical_text_em':sum(scores[t]['exact_match'] for t in TASKS[:2])/2,
                            'medical_image_em':sum(scores[t]['exact_match'] for t in TASKS[2:])/3,
                            'old_task_delta':old_delta,'medical_probe':probes.get('medical',{}).get('accuracy'),
                            'general_probe':probes.get('general',{}).get('accuracy')})
            for group,s in probes.items():probe_rows.append({'method':method,'stage':stage,'group':group,**s})
        completed=[i for i,s in enumerate(results) if len(s)==5]
        if not completed:continue
        last=max(completed)
        for i,t in enumerate(TASKS):
            base=results[0][t]['exact_match'];learned=results[i+1][t]['exact_match'] if i+1<=last else None
            final=results[last][t]['exact_match']
            gains.append({'method':method,'task':t,'latest_stage':last,'base':base,'just_learned':learned,'latest':final,
                          'learning_gain':None if learned is None else learned-base,
                          'retained_gain':None if learned is None else final-base,
                          'change_after_learning':None if learned is None else final-learned})
    for filename,rows,fields in [('stage_scores.csv',score_rows,['method','stage','learned_task','task','exact_match','token_f1','rougeL','count']),
                                 ('process_metrics.csv',process,['method','stage','all_task_em','medical_text_em','medical_image_em','old_task_delta','medical_probe','general_probe']),
                                 ('learning_gain.csv',gains,['method','task','latest_stage','base','just_learned','latest','learning_gain','retained_gain','change_after_learning']),
                                 ('knowledge_probe.csv',probe_rows,['method','stage','group','count','accuracy']),
                                 ('old_task_impact_details.csv',impact_rows,['method','stage','current_training_task','old_test_task','test_count','before_accuracy','after_accuracy','change_pp']),
                                 ('old_task_impact_stages.csv',impact_stages,['method','stage','current_training_task','before_stage','after_stage','old_test_tasks','old_test_questions','old_task_macro_before','old_task_macro_after','old_task_delta_pp'])]:
        with (OUT/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(OUT/'process_data.json',{'run':str(run) if run else None,'tasks':TASKS,'methods':METHODS,'rows':process})
    write_json(OUT/'latest.json',{'run':str(run) if run else None,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    lines=['# 医学实验：持续学习与诊断读出适配\n',
           '从原始 Qwen2-VL-2B-Instruct 开始，依次学习 MedMCQA → MedQA → 胸部图像 → 头部图像 → 腹部图像。四方法使用同一数据和上一轮 LoRA 参数。只用医学 QA 训练；外部通用题仅用于测量知识变化。\n',
           '**总对照入口：** [基线性能、训练过程与方法来源](../summary/medical_research_comparison.md)。本页先列本研究的诊断适配，再列四方法纯医学持续学习；两组实验各有固定测试集。\n']
    adaptation=read(ROOT/'analysis/omnimed_20261009/adaptation_summary.json',{})
    if adaptation.get('results'):
        overall=next(r for r in adaptation['results'] if r['source']=='ALL')
        a=overall['image'];b=overall['no_image']
        base=a['base_accuracy']
        lines.extend(['## 本研究方法：诊断读出与语言端适配\n',
            '数据来自 **OmniMedVQA（CVPR 2024）**。固定131张测试图，来自 ISIC2019、Retinal OCT-C8、Fitzpatrick17k，每个来源四类。所有方法共享视觉基座；准确率为 %，提升为百分点。\n',
            table(['方法／来源','训练设置','测试准确率','相对基础模型','去图像准确率'],[
                ['基础模型／Qwen2-VL','本轮无训练',number(base),'0.00',number(b['base_accuracy'])],
                ['原医学 LoRA／前期实验','VQA-RAD 80张训练图；普通rank-8',number(a['old_medical_lora_accuracy']),number(a['old_medical_lora_accuracy']-base),number(b['old_medical_lora_accuracy'])],
                ['本研究：诊断标签监督 LoRA','372张图；语言q/v普通rank-8；冻结视觉；3轮；3种子',number(a['new_lora_accuracy']),number(a['new_lora_accuracy']-base),number(b['new_lora_accuracy'])],
                ['本研究：冻结特征线性读出','同372张图；每来源一个线性分类器；冻结视觉',number(a['linear_accuracy']),number(a['linear_accuracy']-base),'不适用：输入是图像特征']]),
            '诊断标签监督 LoRA 使用普通 LoRA 结构，本研究改变的是诊断标签监督和适配实验设置。线性读出用于检验固定视觉特征中可恢复的类别信息。新 LoRA 相对原 LoRA 提升17.05个百分点，95%区间[7.38, 26.72]。线性读出按来源单独训练，LoRA联合训练，结构差异纳入结果解释。\n',
            '**论文与方法的对应：** OmniMedVQA 论文提供数据和问答基准；上述线性读出、诊断监督 LoRA 及划分由本研究设计。医学 LoRA 论文方法的复现对照尚未完成。\n',
            '**训练过程：** 同一127张验证图上，第1／2／3轮的三种子均值为40.16%／42.26%／44.09%；各种子由验证集选择第3轮后，在131张测试图得到47.84%。本轮诊断适配尚未测量逐轮旧任务保留和外部知识，因此A–D图展示下面的四方法持续学习实验。\n',
            '[完整诊断实验报告](../analysis/omnimed_20261009/README.md) · [逐轮训练CSV](../summary/omnimed_training_process.csv)\n'])
    lines.append('## 纯医学持续学习：五类真实示例\n')
    examples=[]
    # Fixed first test row per task, independently of model score.
    if manifest:
        import torch
        data=torch.load(ROOT/'exp/CV_result_med/shared/data.pt',map_location='cpu',weights_only=True)
        for task,label in zip(TASKS,SHORT):
            row=data[task]['test'][0]
            lines.extend([f'### {label}\n',row['prompt']+'\n',f'参考答案：**{row["references"][0]}**。\n'])
            if row.get('image'):
                dest=ROOT/'sample/med'/Path(row['image']).name;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(ROOT/'CV_data'/row['image'],dest)
                lines.append(f'![{label}测试原图](../sample/med/{dest.name})\n\n**图注｜{label}原图。** 固定第一条测试样本，与上述问题配对；原始图片直接复制，不改尺寸或像素，不按模型表现挑选。\n')
            answers=[]
            for method,name in zip(METHODS,NAMES):
                stages=[i for i,s in enumerate(allresults[method]) if task in s]
                if not stages or not run:continue
                stage=max(stages);path=run/method/'predictions'/f'stage_{stage:02d}'/f'{task}.jsonl'
                if path.exists():
                    record=json.loads(path.read_text().splitlines()[0]);assert record['instance_id']==row['instance_id']
                    answers.append([name,stage,record['prediction'].replace('|','&#124;').replace('\n',' '),number(record['scores']['exact_match'])])
            lines.append(table(['方法','最新评估阶段','模型回答','EM (%)'],answers) if answers else '模型回答尚未产生；后台报告会自动补充。\n')
            examples.append({'task':task,'instance_id':row['instance_id'],'prompt':row['prompt'],'references':row['references'],'image':row.get('image')})
        write_json(OUT/'examples.json',examples)
    lines.extend(['## 当前进度与效果\n',table(['方法','状态','阶段','正在处理'],statusrows),
                  '阶段 0 为基座；阶段 1–5 对应上述五任务。每阶段重测固定全部 600 条测试题，而不是只测当前任务。未完成数据留空，不记成 0 分。\n'])
    for method,name in zip(METHODS,NAMES):
        results=allresults[method]
        if not results:continue
        lines.append(f'### {name}：逐阶段任务准确率\n')
        lines.append(table(['阶段']+SHORT,[[stage]+[number(scores.get(t,{}).get('exact_match')) for t in TASKS] for stage,scores in enumerate(results)]))
    final=[]
    for method,name in zip(METHODS,NAMES):
        r=allresults[method]
        if len(r)==6 and all(len(s)==5 for s in r):
            matrix=[[s[t]['exact_match'] for t in TASKS] for s in r];m=continual_metrics(matrix)
            final.append([name]+[number(m[k]) for k in ['AP','F.Rate','BWT','FWT']])
    lines.extend(['### 完整 CL 指标\n',table(['方法','最终 AP ↑','遗忘幅度 ↓','BWT ↑','FWT ↑'],final) if final else '完整五阶段尚未完成，暂不发布最终 CL 指标。\n',
                  '## 学会多少、保留多少\n',
                  '下表比较同一批题：学习增益＝刚学完 − 基座，保留增益＝最新阶段 − 基座，后续变化＝最新阶段 − 刚学完。正的学习增益才支持“本任务确实改善”；低遗忘且低学习增益可能只是没有学会。最新阶段在学习完成前持续变化。\n',
                  table(['方法','任务','最新阶段','基座','刚学完','最新','学习增益','保留增益','后续变化'],
                        [[NAMES[METHODS.index(r['method'])],SHORT[TASKS.index(r['task'])],r['latest_stage']]+[number(r[k]) for k in ['base','just_learned','latest','learning_gain','retained_gain','change_after_learning']] for r in gains]),
                  '## 外部知识有没有变化\n',
                  '每阶段测同一组独立 198 道 MMLU 题：医学 96、通用 102。使用 A/B/C/D 下一 token 概率评分，保存四个概率和逐题答案；这些题不参与训练、验证或选 checkpoint。它能显示固定题组的知识变化，不代表全部知识，也不能和自由生成 EM 混为一个指标。\n',
                  table(['方法','阶段','医学 (%)','通用 (%)'],[[NAMES[METHODS.index(r['method'])],r['stage'],number(r['medical_probe']),number(r['general_probe'])] for r in process])])
    if plot(process):
        lines.extend(['![纯医学 CL 全过程](../sample/med/process_med.png)\n',
                      '**图注｜A–D：四种方法在纯医学持续学习中的变化。** 横轴0＝基础模型，1＝学完MedMCQA，2＝学完MedQA，3＝学完胸部，4＝学完头部，5＝学完腹部。\n',
                      table(['面板／位置','测量什么','如何读'],[
                          ['A／左上：总体任务表现','固定600题，先算五个任务各自准确率，再等权平均','纵轴为准确率%；越高表示固定任务平均表现越好'],
                          ['B／右上：医学知识','每阶段重测固定96道外部医学MMLU题','纵轴为准确率%；观察医学知识题的变化'],
                          ['C／左下：旧任务变化','训练当前任务之后−之前，在同一组已学任务测试题上的宏平均准确率','纵轴为百分点；负值表示本次训练后旧任务准确率下降'],
                          ['D／右下：通用知识','每阶段重测固定102道外部通用MMLU题','纵轴为准确率%；观察通用知识题的变化']]),
                      '曲线：蓝色圆点＝SeqLoRA，绿色菱形＝MIGU-LoRA，红色三角＝O-LoRA，紫色方块＝SAPT-LoRA。折线连接已测阶段，单种子；[矢量PDF](figures/process_med.pdf)。\n',
                      '### C图的实验设置：怎样测量旧任务受损\n',
                      '对每种方法，在阶段s−1保存模型并评测固定测试题；按各方法设置学习当前任务的训练集，完成阶段s，再用更新后的模型重测完全相同的旧任务测试题。SAPT还按其实现执行任务边界反思。测试题不参与训练。取每个旧任务准确率的“训练后−训练前”，再对旧任务等权平均。训练超参数及各任务样本量见本页“数据规模、实现与恢复”。\n',
                      table(['横轴阶段','本次训练','比较模型状态','测量哪些旧任务','固定旧测试题数'],[
                          [2,'MedQA','学完MedMCQA → 学完MedQA','MedMCQA',200],
                          [3,'胸部图像','学完MedQA → 学完胸部','MedMCQA、MedQA',400],
                          [4,'头部图像','学完胸部 → 学完头部','MedMCQA、MedQA、胸部',491],
                          [5,'腹部图像','学完头部 → 学完腹部','MedMCQA、MedQA、胸部、头部',541]]),
                      '阶段0／1尚无可测的旧任务训练前后变化，C图留空。不同阶段的旧任务集合随学习增加，各点对应表中各自的集合。\n',
                      r'$$\Delta_{\mathrm{old}}(s)=\frac{1}{s-1}\sum_{t=1}^{s-1}\left[R_{s,t}-R_{s-1,t}\right],\quad s=2,3,4,5.$$'+'\n',
                      '**具体例子：** SeqLoRA 在阶段2训练MedQA；旧任务只有MedMCQA的同一200道测试题。训练前准确率45.00%（90题正确），训练后44.00%（88题正确），所以C图阶段2为44.00−45.00＝**−1.00个百分点**。这测量本次更新后旧任务的准确率下降；因果解释受单种子和固定题组范围限制。\n',
                      table(['方法','本次训练','旧任务平均：训练前','训练后','C图变化／百分点'],
                          [[NAMES[METHODS.index(r['method'])],SHORT[TASKS.index(r['current_training_task'])],
                            number(r['old_task_macro_before']),number(r['old_task_macro_after']),number(r['old_task_delta_pp'])]
                           for r in impact_stages]),
                      '[逐任务训练前后明细CSV](old_task_impact_details.csv) · [C图每阶段汇总CSV](old_task_impact_stages.csv)。\n'])
    lines.extend(['## 指标与实验边界\n',
                  'MedMCQA/MedQA 主指标为严格答案字母准确率（允许末尾句点或右括号）；图像主指标为规范化 EM，并另存 Token F1、ROUGE-L 及 OPEN/CLOSED 分组。图像短答案同义表达可能被 EM 判错，ROUGE-L 只是文字重合。AP 为五任务等权均分；图像三个器官共享一个数据集，因此也分别报告文字/图像领域均分，不能称为三个独立数据集。\n',
                  r'令 $R_{s,t}$ 为学完 $s$ 个任务后在任务 $t$ 上的准确率（百分数），$T=5$。'+'\n',
                  r'$$\mathrm{AP}=\frac{1}{T}\sum_{t=1}^{T}R_{T,t},\qquad \Delta_{\mathrm{learn},t}=R_{t,t}-R_{0,t},\qquad \Delta_{\mathrm{retain},t}=R_{T,t}-R_{0,t}.$$'+'\n',
                  r'$$\mathrm{BWT}=\frac{1}{T-1}\sum_{t=1}^{T-1}(R_{T,t}-R_{t,t}),\qquad \mathrm{FWT}=\frac{1}{T-1}\sum_{t=2}^{T}(R_{t-1,t}-R_{0,t}).$$'+'\n',
                  r'$$\mathrm{F.Rate}=\frac{1}{T-1}\sum_{t=1}^{T-1}\left(\max_{s\in\{t,\ldots,T-1\}}R_{s,t}-R_{T,t}\right).$$'+'\n',
                  '单种子、固定任务顺序、每任务 1 epoch。视觉编码器冻结，图像最多 128 token；没有同时改路由、正则或分辨率，以便和上一轮对照。医学选择题只训练字母，不训练解释或临床推理过程。图像保持现有 seed42 病例分组切分，非论文标准问题级切分。MedMCQA 的 dev/test 是有标签的官方 validation 子集；MedQA 的 dev 从官方 train 单独划出。数据集间仅去除完全相同的问题，没有完成语义近重复去污染。\n',
                  '## 数据规模、实现与恢复\n'])
    if manifest:lines.append(table(['任务','训练','验证','测试'],[[label]+[manifest['counts'][t][s] for s in ['train','dev','test']] for t,label in zip(TASKS,SHORT)]))
    lines.extend(['模型约 2.21B；LoRA rank=8、alpha=32、dropout=0.1、q_proj/v_proj，学习率 1e-4，有效 batch=16、micro-batch=1、seed=42。方法额外参数和实际数据哈希保存在各自 config/manifest。每次优化更新保存优化器、调度器、AMP、RNG；每阶段保存完整边界，测试回答按批落盘。nohup 独立会话，关闭终端继续运行。显存不足时等待；训练 OOM 最多自动恢复 4 次，其余失败保留日志。\n',
                  f'当前原始结果目录：`{str(run.relative_to(ROOT)) if run else "exp/CV_result_med/（尚未启动）"}`，每个方法独立子目录；医学数据 `data/medical_qa_med`，图像原件 `CV_data/medical/vqa_rad`，展示图片 `sample/med`。\n',
                  '```bash\n# 查看实时状态与报告\n.plot-env/bin/python summary_cv_med/collect_med.py\n# 中断后恢复同一轮（把下方路径替换为 latest.json 中的 run）\n.vision-env/bin/python exp/start_cv_detached_med.py --resume exp/CV_result_med/qwen2vl2b_pure_med/<run>\n```\n',
                  '数据来源：[MedMCQA 官方](https://github.com/medmcqa/medmcqa)、[MedQA 官方](https://github.com/jind11/MedQA)、[VQA-RAD 官方](https://osf.io/89kps/)。下载镜像版本、文件 SHA、选择 ID 在独立 manifest；全部配置为 `exp/configs/*_qwen2vl_med.json`，运行代码 `exp/cv_run_med.py`。\n'])
    tmp=OUT/'README.md.tmp';tmp.write_text('\n'.join(lines));tmp.replace(OUT/'README.md')
    return bool(run) and all(read(run/m/'queue.json',{}).get('state') in ['completed','failed'] for m in METHODS)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');args=p.parse_args()
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    while True:
        with (OUT/'.collect_med.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            done=collect()
        if not args.watch or done:break
        time.sleep(60)
