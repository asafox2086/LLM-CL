"""Collect and verify the multimodal run; choose examples by fixed test order."""
from captions import add_captions
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'exp'))
from common import continual_metrics, write_json
from render_report import build_report
from publish_samples import publish_samples

METHODS = ['seq_lora', 'migu_lora', 'olora', 'sapt_lora']
NAMES = {'seq_lora': 'SeqLoRA', 'migu_lora': 'MIGU-LoRA', 'olora': 'O-LoRA', 'sapt_lora': 'SAPT-LoRA'}
TASK_NAMES = {
 'task1290_xsum_summarization': ('通用语言', '新闻摘要（XSum）'),
 'task002_quoref_answer_generation': ('通用语言', '阅读理解（Quoref）'),
 'task1510_evalution_relation_extraction': ('通用语言', '关系抽取'),
 'task1729_personachat_generate_next': ('通用语言', '对话续写（PersonaChat）'),
 'task748_glucose_reverse_cause_event_detection': ('通用语言', '因果推理（GLUCOSE）'),
 'medical_mts_dialog_note': ('医学语言', '医患对话转临床记录（MTS-Dialog）'),
 'medical_iu_xray_impression': ('医学语言', '影像 Findings 转 Impression（IU-Xray）'),
 'natural_vqav2': ('自然图像', '自然图像问答（VQAv2）'),
 'medical_vqa_rad': ('医学图像', '医学图像问答（VQA-RAD）'),
}
SCORES = ['rougeL', 'exact_match', 'token_f1']

def read_jsonl(path):
    with path.open() as handle:
        return [json.loads(line) for line in handle]

def save_csv(path, rows, fields):
    temp = path.with_suffix('.tmp')
    with temp.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temp.replace(path)

def close(a, b):
    if not math.isclose(float(a), float(b), abs_tol=1e-8, rel_tol=1e-10):
        raise ValueError(f'Score mismatch: {a} vs {b}')

def fence(text):
    marker = '`' * (max([len(part) for part in str(text).split() if set(part) == {'`'}] + [3]) + 1)
    return marker + 'text\n' + str(text) + '\n' + marker + '\n'

def collect(run, output):
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((run / METHODS[0] / 'config.json').read_text())
    tasks = config['tasks']
    examples = []
    selected = defaultdict(lambda: defaultdict(list))
    for task in tasks:
        path = run / METHODS[0] / 'predictions/stage_00' / (task + '.jsonl')
        baseline = read_jsonl(path) if path.exists() else []
        if not baseline:
            continue
        choices = [(TASK_NAMES[task][1], baseline[0], 'task')]
        if task in ['natural_vqav2', 'medical_vqa_rad']:
            for category in sorted({r['answer_type'] for r in baseline}):
                row = next(r for r in baseline if r['answer_type'] == category)
                choices.append((TASK_NAMES[task][1] + ' / ' + category, row, 'answer_type'))
        for title, record, kind in choices:
            example = {'category': TASK_NAMES[task][0], 'title': title, 'kind': kind,
                'selection': 'first example in fixed baseline test order; independent of scores',
                'task_id': task, 'instance_id': record['instance_id'], 'source': record, 'responses': {}}
            examples.append(example)
            selected[task][record['instance_id']].append(example)
    rows, overview, domain_rows, groups = [], [], [], []
    checks = []
    total_records = 0
    for method in METHODS:
        directory = run / method
        cfg = json.loads((directory / 'config.json').read_text())
        assert cfg['tasks'] == tasks
        for key in ['seed','epochs','batch_size','micro_batch_size','eval_batch_size','learning_rate','rank','alpha',
                    'dropout','prompt_tokens','target_tokens','train_count','dev_count','test_count','data_manifest_sha256']:
            assert cfg[key] == config[key], (method,key)
        status = json.loads((directory / 'status.json').read_text())
        completed = status['status'] == 'completed'
        scores = json.loads((directory / 'scores.json').read_text()) if (directory / 'scores.json').exists() else []
        if completed:
            assert len(scores) == len(tasks)+1
            assert all(set(stage)==set(tasks) for stage in scores)
        reference_ids = {}
        count = 0
        for stage, entries in enumerate(scores):
            for task, values in entries.items():
                prediction_path = directory / 'predictions' / f'stage_{stage:02d}' / (task+'.jsonl')
                records = read_jsonl(prediction_path)
                assert len(records) == cfg['test_count'] == values['count']
                ids = [r['instance_id'] for r in records]
                assert len(set(ids)) == len(ids)
                signatures = [(r['instance_id'], r['references'], r['prompt_token_ids']) for r in records]
                if task in reference_ids:assert signatures == reference_ids[task]
                else:reference_ids[task] = signatures
                for key in SCORES:
                    close(sum(r['scores'][key] for r in records)/len(records), values[key])
                row = {'method':method,'stage':stage,'learned_task':tasks[stage-1] if stage else 'base',
                    'domain':TASK_NAMES[task][0], 'evaluated_task':task, 'task_name':TASK_NAMES[task][1],
                    **{k:values[k] for k in SCORES}, 'count':len(records)}
                rows.append(row);count += len(records)
                for key, info in values.get('groups',{}).items():
                    groups.append({'method':method,'stage':stage,'task':task,'group':key,**{k:info[k] for k in SCORES},'count':info['count']})
                if stage in [0,tasks.index(task)+1,len(tasks)]:
                    for record in records:
                        for example in selected[task].get(record['instance_id'],[]):
                            example['responses'].setdefault(method,{})[str(stage)] = record
        total_records += count
        metrics_path = directory / 'continual_metrics.json'
        metrics = json.loads(metrics_path.read_text()) if completed else {}
        if completed:
            matrix = [[s[t]['rougeL'] for t in tasks] for s in scores]
            assert matrix == metrics['matrix']
            for key, value in continual_metrics(matrix).items():close(value,metrics[key])
        overview.append({'method':method,'status':status['status'],**{k:metrics.get(k,'') for k in ['AP','F.Rate','BWT','FWT']}})
        checks.append({'method':method,'status':status['status'],'verified_test_records':count,
                       'verified_stage_task_files':sum(len(s) for s in scores),'metrics_match_predictions':True})
        if scores and len(scores[-1]) == len(tasks):
            for domain in dict.fromkeys(TASK_NAMES[t][0] for t in tasks):
                group = [t for t in tasks if TASK_NAMES[t][0]==domain]
                domain_rows.append({'method':method,'domain':domain,'task_count':len(group),
                    **{k:sum(scores[-1][t][k] for t in group)/len(group) for k in SCORES}})
    publish_samples(examples)
    lines = build_report(run, ROOT, examples, overview, domain_rows, rows, total_records, config)
    (output/'README.md').write_text(add_captions('\n'.join(lines)+'\n'))
    save_csv(output/'results.csv',overview,['method','status','AP','F.Rate','BWT','FWT'])
    save_csv(output/'stage_scores.csv',rows,['method','stage','learned_task','domain','evaluated_task','task_name',*SCORES,'count'])
    save_csv(output/'domain_scores.csv',domain_rows,['method','domain','task_count',*SCORES])
    save_csv(output/'category_scores.csv',groups,['method','stage','task','group',*SCORES,'count'])
    write_json(output/'examples.json',examples)
    ex_lines = ['# 各任务与图像答案类别的真实示例', '',
        '每个任务取固定测试顺序的第一条；图像答案类别再各取第一条。选择不依赖模型得分。回答来自保存的预测文件，未人工改写。', '',
        '“刚学完”指该任务训练完成后的阶段；“最终”指九任务全部学习后的阶段。图像问答保留真实图片链接；文本保持原始英文。', '']
    for index, example in enumerate(examples,1):
        record = example['source'];task = example['task_id']
        ex_lines += [f"## {index}. {example['category']}：{example['title']}", '',
            f"任务：`{task}`；实例：`{example['instance_id']}`。", '']
        if record.get('image'):
            ex_lines += [f"![{example['title']}](../{example['display_image']})", '',
                f"图片：[`{record['image']}`](../{example['display_image']})；视觉 token：{record['image_tokens']}。", '']
        ex_lines += ['输入：', '', fence(record['prompt']), '参考答案（去除完全重复的文字）：', '']
        for reference in dict.fromkeys(record['references']):ex_lines.append(fence(reference))
        if record.get('prompt_truncated'):
            ex_lines += ['该输入在模型中已截断；实际输入 token 和 chat prompt 完整保存在 `examples.json`。', '']
        for method in METHODS:
            ex_lines += [f"### {NAMES[method]}", '']
            for stage,label in dict.fromkeys([(0,'基线'),(tasks.index(task)+1,'刚学完'),(len(tasks),'最终')]).items():
                result = example['responses'].get(method,{}).get(str(stage))
                if result is None:continue
                scores = result['scores']
                ex_lines += [f"**{label}（阶段 {stage}）**：ROUGE-L {scores['rougeL']:.3f}，EM {scores['exact_match']:.3f}，F1 {scores['token_f1']:.3f}。", '',
                             fence(result['prediction'])]
    (output/'examples.md').write_text(add_captions('\n'.join(ex_lines)+'\n'))
    write_json(output/'verification.json',{'run':str(run),'checks':checks,'total_test_records':total_records,
        'task_examples':sum(e['kind']=='task' for e in examples),'answer_category_examples':sum(e['kind']=='answer_type' for e in examples)})
    write_json(output/'latest.json',{'run':str(run),'model':'Qwen/Qwen2-VL-2B-Instruct','tasks':tasks})
    print('\n'.join(lines[:14]))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path)
    args=parser.parse_args()
    run=args.run.resolve() if args.run else Path(json.loads((ROOT/'exp/CV_result/latest.json').read_text())['run'])
    collect(run, ROOT/'summary_cv')
