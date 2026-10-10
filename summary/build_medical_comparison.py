"""Build review tables from completed SSH experiment outputs; no model inference."""
import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if ROOT.name == 'summary':
    ROOT = ROOT.parent
OUT = ROOT / 'summary'
RUN = ROOT / 'analysis/omnimed_20261009'
MED = ROOT / 'summary_cv_med'
OUT.mkdir(exist_ok=True)

def load(name):
    return json.loads((RUN / name).read_text(encoding='utf-8'))

def lines(name):
    return [json.loads(s) for s in (RUN / name).read_text(encoding='utf-8').splitlines() if s]

def csv_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def write_csv(name, rows):
    with (OUT / name).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        w.writeheader()
        w.writerows(rows)

def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(str(v) for v in row) + ' |' for row in rows])

def num(value):
    return '—' if value is None or value == '' else f'{float(value):.2f}'

def ci(value):
    return f'[{value[0]:.2f}, {value[1]:.2f}]'

adapt = load('adaptation_summary.json')
primary = load('analysis_summary.json')['primary']
scores = lines('adaptation_scores.jsonl')
updates = lines('adaptation_updates.jsonl')
assert len(updates) == 423
assert len({(u['seed'], u['step']) for u in updates}) == 423
assert all(r['candidate_accuracy'] == r['strict_accuracy'] for r in scores)
assert len(lines('adaptation_predictions.jsonl')) == 786
assert len(lines('predictions.jsonl')) == 11288
assert all(len({u['step'] for u in updates if u['seed'] == s and u['epoch'] == e}) == 47
           for s in (42, 43, 44) for e in (1, 2, 3))

baseline_rows = []
for r in adapt['results']:
    a, b = r['image'], r['no_image']
    baseline_rows.append(dict(source=r['source'], test_images=r['n'], base_accuracy=a['base_accuracy'],
        old_medical_lora_accuracy=a['old_medical_lora_accuracy'], adapted_lora_accuracy=a['new_lora_accuracy'],
        linear_readout_accuracy=a['linear_accuracy'], gain_vs_base_pp=a['new_minus_base_pp'],
        gain_vs_old_lora_pp=a['new_minus_old_pp'], gain_vs_old_ci_low=a['new_minus_old_ci'][0],
        gain_vs_old_ci_high=a['new_minus_old_ci'][1], adapted_no_image_accuracy=b['new_lora_accuracy'],
        adapted_image_gain_pp=r['visual_contribution']['new_image_minus_no_image_pp']))
write_csv('omnimed_baseline_comparison.csv', baseline_rows)

epoch_rows = []
for epoch in (1, 2, 3):
    s = {r['seed']: r['candidate_accuracy'] for r in scores if r['epoch'] == epoch and r['split'] == 'dev'}
    assert set(s) == {42, 43, 44}
    u = [r for r in updates if r['epoch'] == epoch]
    epoch_rows.append(dict(epoch=epoch, updates_per_seed=epoch * 47, evaluation_split='dev', images=127,
        seed42_accuracy=s[42], seed43_accuracy=s[43], seed44_accuracy=s[44], mean_accuracy=statistics.mean(s.values()),
        sample_sd=statistics.stdev(s.values()), mean_train_loss=statistics.mean(r['loss'] for r in u),
        mean_logged_gradient_norm=statistics.mean(r['gradient_norm'] for r in u)))
write_csv('omnimed_training_process.csv', epoch_rows)

category_names = {'Anatomy Identification': '解剖识别', 'Modality Recognition': '模态识别',
                  'Disease Diagnosis': '诊断', 'Lesion Grading': '病变分级',
                  'Other Biological Attributes': '其他生物属性', 'ALL': '全部题目'}
primary_rows = []
categories = list(dict.fromkeys(r['category'] for r in primary))
for category in categories:
    def get(arm, condition):
        return next(r for r in primary if r['category'] == category and r['arm'] == arm and r['condition'] == condition)
    b, o, no = get('base', 'image'), get('SFT_r8_mean', 'image'), get('SFT_r8_mean', 'no_image')
    primary_rows.append(dict(task=category_names.get(category, category), questions=b['n'],
        base_accuracy=b['accuracy'], old_lora_accuracy=o['accuracy'], gain_vs_base_pp=o['accuracy']-b['accuracy'],
        old_lora_no_image_accuracy=no['accuracy'], image_gain_pp=o['accuracy']-no['accuracy']))
write_csv('omnimed_task_comparison.csv', primary_rows)

process = csv_rows(MED / 'process_metrics.csv')
stage_scores = csv_rows(MED / 'stage_scores.csv')
learning = csv_rows(MED / 'learning_gain.csv')
methods = {'seq_lora': 'SeqLoRA', 'migu_lora': 'MIGU-LoRA', 'olora': 'O-LoRA', 'sapt_lora': 'SAPT-LoRA'}
methods = {m: methods.get(m, m) for m in dict.fromkeys(r['method'] for r in process)}
tasks = list(dict.fromkeys(r['task'] for r in stage_scores))
assert len(tasks) == 5 and len(process) == 24 and len(stage_scores) == 120
cl_rows = []
for m, label in methods.items():
    p = {int(r['stage']): r for r in process if r['method'] == m}
    assert set(p) == set(range(6))
    matrix = {(int(r['stage']), r['task']): float(r['exact_match']) for r in stage_scores if r['method'] == m}
    r0, rf = p[0], p[5]
    bwt = statistics.mean(matrix[5,t]-matrix[j,t] for j,t in enumerate(tasks[:-1],1))
    fwt = statistics.mean(matrix[j-1,t]-matrix[0,t] for j,t in enumerate(tasks[1:],2))
    forgetting = statistics.mean(max(matrix[s,t] for s in range(j,5))-matrix[5,t]
                                for j,t in enumerate(tasks[:-1],1))
    cl_rows.append(dict(method=label, base_AP=float(r0['all_task_em']), final_AP=float(rf['all_task_em']),
        gain_vs_base_pp=float(rf['all_task_em'])-float(r0['all_task_em']),
        base_text_accuracy=float(r0['medical_text_em']), final_text_accuracy=float(rf['medical_text_em']),
        base_image_accuracy=float(r0['medical_image_em']), final_image_accuracy=float(rf['medical_image_em']),
        medical_knowledge_gain_pp=float(rf['medical_probe'])-float(r0['medical_probe']),
        general_knowledge_gain_pp=float(rf['general_probe'])-float(r0['general_probe']),
        forgetting_pp=forgetting, BWT_pp=bwt, FWT_pp=fwt))
write_csv('medical_cl_baseline_comparison.csv', cl_rows)
write_csv('medical_cl_training_process.csv', process)
write_csv('medical_cl_learning_retention.csv', learning)

parts = ['# 医学实验：基线性能与训练过程对照',
    '所有表格由服务器已完成实验的原始结果生成。准确率单位为 %；增益、迁移和遗忘单位为百分点。不同题组各自比较固定基线。',
    '## 1. OmniMedVQA 诊断适配：与基线对比',
    'Qwen2-VL-2B-Instruct；视觉编码器冻结。原医学 LoRA 使用此前的 VQA-RAD 80 张训练图；新诊断 LoRA 从基础模型开始，使用本轮372张训练图、127张验证图，三个种子42/43/44。测试集为三个来源的131张留出图，每个来源选定四类。线性读出使用同一批标签，按来源各拟合一个分类器；LoRA 联合训练三个来源。',
    table(['数据来源', '测试图', '基础模型', '原医学 LoRA', '新诊断 LoRA', '线性读出', '新−基础', '新−原 LoRA（95%区间）'],
          [[r['source'],r['test_images'],num(r['base_accuracy']),num(r['old_medical_lora_accuracy']),num(r['adapted_lora_accuracy']),
            num(r['linear_readout_accuracy']),f"{r['gain_vs_base_pp']:+.2f}",
            f"{r['gain_vs_old_lora_pp']:+.2f} [{r['gain_vs_old_ci_low']:.2f}, {r['gain_vs_old_ci_high']:.2f}]" ] for r in baseline_rows]),
    '新诊断 LoRA 总体准确率47.84%，相对原医学 LoRA 提升17.05个百分点，95%区间[7.38, 26.72]。',
    '### 图像利用能力对照（同一131张测试图、相同问题和选项）',
    table(['模型', '有图像', '去掉图像', '图像贡献'], [[label,num(a['image'][key]),num(a['no_image'][key]),
        f"{a['image'][key]-a['no_image'][key]:+.2f}"] for a in adapt['results'] if a['source']=='ALL'
        for label,key in [('基础模型','base_accuracy'),('原医学 LoRA','old_medical_lora_accuracy'),('新诊断 LoRA','new_lora_accuracy')]]),
    '图像贡献＝有图像准确率−去掉图像准确率。新 LoRA 相对原 LoRA 的图像贡献增加15.01个百分点，95%区间[4.06, 25.70]；这支持本任务中语言端适配改善了固定视觉表示的利用。',
    '## 2. 新诊断 LoRA：训练中的能力变化',
    '每轮完成47次参数更新，每种子共141次更新；三个种子合计423次。下表每轮都评测同一127张验证图，三个种子均由验证集选择第3轮。验证集第0轮准确率未记录。',
    table(['轮次', '累计更新／种子', '验证图', 'seed42', 'seed43', 'seed44', '均值 ± 种子标准差', '平均训练损失'],
        [[r['epoch'],r['updates_per_seed'],r['images'],num(r['seed42_accuracy']),num(r['seed43_accuracy']),num(r['seed44_accuracy']),
          f"{r['mean_accuracy']:.2f} ± {r['sample_sd']:.2f}",f"{r['mean_train_loss']:.4f}"] for r in epoch_rows]),
    f"验证准确率从第1轮的{epoch_rows[0]['mean_accuracy']:.2f}%提高到第3轮的{epoch_rows[-1]['mean_accuracy']:.2f}%，增加{epoch_rows[-1]['mean_accuracy']-epoch_rows[0]['mean_accuracy']:.2f}个百分点。损失为训练中各更新记录的平均值；不同轮次对应不同参数状态，用于描述优化过程。梯度范数原始统计保存在CSV中。",
    '最终选中模型在131张测试图上，三个种子分别为45.80%、48.85%、48.85%。验证曲线与最终测试成绩分别报告。每轮分来源准确率、每轮去图像能力、旧任务保留能力在这次诊断适配中未记录。',
    '## 3. OmniMedVQA 1024题：医学能力分项基线',
    '本表使用原医学 LoRA（三种子均值）和基础模型。新诊断 LoRA 的131图测试与本表采用不同题组。候选字母准确率与大小写等价的首答案字母准确率在这1024题上完全一致。',
    table(['任务','题数','基础模型','原医学 LoRA','相对基础增益','原 LoRA 去图像','图像贡献'],
        [[r['task'],r['questions'],num(r['base_accuracy']),num(r['old_lora_accuracy']),f"{r['gain_vs_base_pp']:+.2f}",
          num(r['old_lora_no_image_accuracy']),f"{r['image_gain_pp']:+.2f}"] for r in primary_rows]),
    '## 4. 纯医学持续学习：四种方法与基础模型／SeqLoRA 对照',
    '顺序：MedMCQA → MedQA → 胸部图像 → 头部图像 → 腹部图像。四方法均从同一 Qwen2-VL 基础模型开始，seed42，每任务1轮。每阶段固定重测600题；AP为五任务等权均分，文字均分为两任务等权，图像均分为三个器官任务等权。外部医学96题、通用102题只用于测量。',
    table(['方法','基础 AP','最终 AP','相对基础','相对 SeqLoRA','文字：基础→最终','图像：基础→最终','医学知识变化','通用知识变化'],
        [[r['method'],num(r['base_AP']),num(r['final_AP']),f"{r['gain_vs_base_pp']:+.2f}",
          f"{r['final_AP']-cl_rows[0]['final_AP']:+.2f}",f"{r['base_text_accuracy']:.2f}→{r['final_text_accuracy']:.2f}",
          f"{r['base_image_accuracy']:.2f}→{r['final_image_accuracy']:.2f}",f"{r['medical_knowledge_gain_pp']:+.2f}",
          f"{r['general_knowledge_gain_pp']:+.2f}"] for r in cl_rows]),
    table(['方法','遗忘幅度 ↓','BWT ↑','FWT ↑'],[[r['method'],num(r['forgetting_pp']),num(r['BWT_pp']),num(r['FWT_pp'])] for r in cl_rows]),
    '遗忘幅度比较旧任务的历史最好成绩与最终成绩；负值表示最终成绩超过此前最好成绩。BWT比较刚学完与最终，FWT比较任务训练之前与基础模型。单种子结果按本轮观察解释。',
    '### 逐阶段：任务能力与外部知识变化',
    '阶段0＝基础模型，1＝MedMCQA，2＝MedQA，3＝胸部，4＝头部，5＝腹部。旧任务变化比较相邻阶段中同一组此前已学任务，阶段0／1留空。',
    table(['方法','阶段','五任务 AP','文字均分','图像均分','旧任务变化','医学知识','通用知识'],
        [[methods[r['method']],r['stage'],num(r['all_task_em']),num(r['medical_text_em']),num(r['medical_image_em']),
          num(r['old_task_delta']),num(r['medical_probe']),num(r['general_probe'])] for r in process]),
    '### 各任务：学会多少、最终保留多少',
    '学习增益＝刚学完−基础；最终保留增益＝最终−基础；后续变化＝最终−刚学完。',
    table(['方法','任务','基础','刚学完','最终','学习增益','保留增益','后续变化'],
        [[methods[r['method']],r['task'].replace('medical_', ''),num(r['base']),num(r['just_learned']),num(r['latest']),
          num(r['learning_gain']),num(r['retained_gain']),num(r['change_after_learning'])] for r in learning]),
    '## 5. 数据来源、范围和可核对材料',
    'OmniMedVQA 来自 [Hu 等，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Hu_OmniMedVQA_A_New_Large-Scale_Comprehensive_Evaluation_Benchmark_for_Medical_LVLM_CVPR_2024_paper.html)，[官方数据](https://huggingface.co/datasets/foreverbeliever/OmniMedVQA)。诊断适配使用其中 ISIC2019、Retinal OCT-C8、Fitzpatrick17k 的问答及标签，本轮划分由实验程序构造。',
    '纯医学持续学习的数据是 MedMCQA、MedQA 和 VQA-RAD。VQA-RAD 来自 [Lau 等，Scientific Data 2018](https://doi.org/10.1038/sdata.2018.251)。',
    '诊断适配结果限定于选定三来源、每来源四类和131张测试图；病例缺少患者编号，图像SHA独立只保证完全相同影像未跨集合。临床征象理解、跨患者泛化及医学与NLP的机制差异需独立证据。',
    '- [OmniMedVQA 完整实验报告及原始预测](../analysis/omnimed_20261009/README.md)\n- [纯医学持续学习逐任务报告](../summary_cv_med/README.md)\n- [语言持续学习报告](README.md)\n- [自然／医学图像持续学习报告](../summary_cv/README.md)',
    '### 表格下载与刷新',
    '\n'.join(f'- [{name}]({name})' for name in ['omnimed_baseline_comparison.csv','omnimed_training_process.csv',
        'omnimed_task_comparison.csv','medical_cl_baseline_comparison.csv','medical_cl_training_process.csv','medical_cl_learning_retention.csv']),
    '在服务器仓库根目录运行：\n\n```bash\n.plot-env/bin/python summary/build_medical_comparison.py\n```',
]
(OUT / 'medical_research_comparison.md').write_text('\n\n'.join(parts)+'\n', encoding='utf-8')
entry = OUT / 'README.md'
if entry.exists():
    text = entry.read_text(encoding='utf-8')
    marker = '<!-- medical-research-comparison -->'
    if marker not in text:
        pos = text.find('\n')
        text = text[:pos+1] + '\n' + marker + '\n**医学研究对照表：** [基线性能、逐轮训练能力、四种 LoRA 持续学习与知识保留](medical_research_comparison.md)。含可下载CSV和原始结果链接。\n' + text[pos+1:]
        entry.write_text(text, encoding='utf-8')
inputs = [RUN / n for n in ['adaptation_summary.json','analysis_summary.json','adaptation_scores.jsonl',
    'adaptation_updates.jsonl','adaptation_predictions.jsonl','predictions.jsonl']]
inputs += [MED / n for n in ['process_metrics.csv','stage_scores.csv','learning_gain.csv']]
audit = {'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
         'checked_updates':423,'checked_adaptation_predictions':786,'checked_primary_predictions':11288,
         'checked_cl_stage_rows':120,'checked_cl_process_rows':24,'validation_epochs':[1,2,3],
         'source_root':str(ROOT)}
(OUT / 'medical_comparison_verification.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print('Generated medical comparison report, six CSVs, verification record and README entry.')
