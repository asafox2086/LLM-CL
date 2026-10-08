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
        for ax,key in zip(axes.flat,['all_task_em','old_task_delta','medical_probe','general_probe']):
            valid=[r for r in rows if r.get(key) is not None]
            if not valid:continue
            ax.plot([r['stage'] for r in valid],[r[key] for r in valid],color=colors[i],
                    marker=styles[i][0],linestyle=styles[i][1],label=NAMES[i],markersize=5)
    for ax,title,ylabel in zip(axes.flat,['(a) Fixed five-task mean','(b) Previously learned task impact',
                                          '(c) Held-out medical knowledge','(d) Held-out general knowledge'],
                                 ['EM (%)','Change (percentage points)','Accuracy (%)','Accuracy (%)']):
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
    statusrows=[];allresults={};process=[];score_rows=[];gains=[];probe_rows=[]
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
                                 ('knowledge_probe.csv',probe_rows,['method','stage','group','count','accuracy'])]:
        with (OUT/filename).open('w') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(rows)
    write_json(OUT/'process_data.json',{'run':str(run) if run else None,'tasks':TASKS,'methods':METHODS,'rows':process})
    write_json(OUT/'latest.json',{'run':str(run) if run else None,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    lines=['# 纯医学文字问答与图像问答 CL\n',
           '从原始 Qwen2-VL-2B-Instruct 开始，依次学习 MedMCQA → MedQA → 胸部图像 → 头部图像 → 腹部图像。四方法使用同一数据和上一轮 LoRA 参数。只用医学 QA 训练；外部通用题仅用于测量知识变化。\n',
           '## 先看五类真实示例\n']
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
                      '**图注｜纯医学 CL 全过程。** 横轴是已学医学任务数（0＝基座，1＝MedMCQA，2＝MedQA，3＝胸部，4＝头部，5＝腹部）。左上是固定五任务宏平均 EM；左下是切换到新任务后、同一组此前已学任务的平均准确率变化，负值表示受损；右上和右下分别为固定医学、通用知识题的准确率。准确率单位为 %，变化单位为百分点。蓝色圆点＝SeqLoRA，绿色菱形＝MIGU，红色三角＝O-LoRA，紫色方块＝SAPT。只画已完成评估，不插值；单种子无误差带。[矢量 PDF](figures/process_med.pdf)。\n'])
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
