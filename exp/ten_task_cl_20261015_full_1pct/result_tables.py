"""Immutable predictions -> task/dataset score tables and CL matrices."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

SCORES=['exact_match','token_f1','rougeL']


def metrics(matrix):
    if len(matrix)!=11 or any(len(row)!=10 for row in matrix):
        raise ValueError('Final metrics require a complete 11 x 10 matrix')
    if not all(math.isfinite(value) for row in matrix for value in row):
        raise ValueError('Nonfinite matrix')
    return {'AP':sum(matrix[10])/10,
        'ForgetRate':sum(max(matrix[k][j-1] for k in range(j,10))-matrix[10][j-1] for j in range(1,10))/9,
        'BWT':sum(matrix[10][j-1]-matrix[j][j-1] for j in range(1,10))/9,
        'FWT':sum(matrix[j-1][j-1]-matrix[0][j-1] for j in range(2,11))/9}


def csv_file(path,rows,fields):
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def summarize(run):
    run=Path(run)
    config=json.loads((run/'config.json').read_text())
    task_rows,dataset_rows=[],[]
    cells=defaultdict(dict)
    for path in sorted((run/'evaluations').glob('stage_*/*_predictions.jsonl')):
        stage=int(path.parent.name.split('_')[1])
        task=int(path.name[1:3])
        split=path.name.split('_')[1]
        rows=[json.loads(line) for line in path.open()]
        expected=config.get('counts_per_task',{'test':200,'dev':100})[split]
        if len(rows)!=expected or len({row['id'] for row in rows})!=expected:
            raise ValueError('Incomplete or duplicate prediction rows: '+str(path))
        common={'method':config['method'],'seed':config['seed'],'stage':stage,'task':task,'split':split}
        scores={key:sum(row['scores'][key] for row in rows)/len(rows) for key in SCORES}
        task_rows.append(dict(common,count=len(rows),em_correct=sum(row['scores']['exact_match']==100 for row in rows),**scores))
        if split=='test':
            cells[stage][task]=scores
        grouped=defaultdict(list)
        for row in rows:
            grouped[row.get('dataset',row['id'].split(':',1)[0])].append(row)
        for name,group in grouped.items():
            dataset_rows.append(dict(common,dataset=name,count=len(group),em_correct=sum(row['scores']['exact_match']==100 for row in group),
                **{key:sum(row['scores'][key] for row in group)/len(group) for key in SCORES}))
    fields=['method','seed','stage','task','split','count','em_correct']+SCORES
    csv_file(run/'task_metrics.csv',task_rows,fields)
    csv_file(run/'dataset_metrics.csv',dataset_rows,fields[:5]+['dataset']+fields[5:])
    matrices={key:[] for key in SCORES}
    for stage in range(11):
        if len(cells[stage])!=10:
            break
        for key in SCORES:
            matrices[key].append([cells[stage][task][key] for task in range(1,11)])
    (run/'score_matrices.json').write_text(json.dumps(matrices,indent=2)+'\n')
    complete=len(matrices['rougeL'])==11
    if complete:
        (run/'metrics_by_score.json').write_text(json.dumps({key:metrics(value) for key,value in matrices.items()},indent=2)+'\n')
    return task_rows,dataset_rows,matrices
