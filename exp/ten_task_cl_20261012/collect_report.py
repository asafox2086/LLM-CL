"""Collect actual completed matrices, metrics and compute counters; never invent scores."""
import csv
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'summary/ten_task_cl_20261012'


def write_csv(name,fields,rows):
    path=REPORT/name
    temporary=path.with_suffix('.csv.tmp')
    with temporary.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def collect():
    status_path=REPORT/'run_status.json'
    if not status_path.is_file():
        return
    status=json.loads(status_path.read_text())
    stages,metrics,compute=[],[],[]
    for method,job in status['jobs'].items():
        if not job.get('run_dir'):
            continue
        run=ROOT/job['run_dir']
        matrix_path=run/'score_matrix.json'
        if matrix_path.is_file():
            matrix=json.loads(matrix_path.read_text())
            for stage,scores in enumerate(matrix):
                for task,score in enumerate(scores,1):
                    stages.append({'method':method,'seed':42,'stage':stage,'task':task,'rougeL':score})
            if method=='SeqLoRA' and matrix:
                # Frozen control is the unchanged zero-adapter initial model.
                for stage in range(11):
                    for task,score in enumerate(matrix[0],1):
                        stages.append({'method':'Frozen','seed':42,'stage':stage,'task':task,'rougeL':score})
                metrics.append({'method':'Frozen','seed':42,'AP':sum(matrix[0])/10,'ForgetRate':0,'FWT':0,'BWT':0})
        metric_path=run/'metrics.json'
        if metric_path.is_file():
            metrics.append({'method':method,'seed':42,**json.loads(metric_path.read_text())})
        event_path=run/'logs/events.jsonl'
        if event_path.is_file():
            rows=[]
            for line in event_path.read_text().splitlines():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break
            for stage in sorted({row['stage'] for row in rows}):
                selected=[row for row in rows if row['stage']==stage]
                compute.append({'method':method,'seed':42,'stage':stage,
                    'logged_qa_exposures':sum(16 if row['qa_seen']%16==0 else row['qa_seen']%16 for row in selected),
                    'optimizer_updates':len(selected),'train_seconds':sum(row['seconds'] for row in selected),
                    'prompt_tokens':sum(row['prompt_tokens'] for row in selected),
                    'target_tokens':sum(row['target_tokens'] for row in selected),
                    'peak_allocated_mib':max(row['peak_allocated_mib'] for row in selected)})
    write_csv('stage_scores.csv',['method','seed','stage','task','rougeL'],stages)
    write_csv('metrics.csv',['method','seed','AP','ForgetRate','FWT','BWT'],metrics)
    write_csv('compute.csv',['method','seed','stage','logged_qa_exposures','optimizer_updates','train_seconds',
                             'prompt_tokens','target_tokens','peak_allocated_mib'],compute)


if __name__=='__main__':
    collect()
