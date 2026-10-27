"""Sparse required cells only. Never impute absent evaluations as zeros."""
import csv
import json


def required_tasks(stage, n):
    return list(range(1,n+1)) if stage in (0,n) else [stage,stage+1]


def aggregate(run, manifest, write):
    n=len(manifest['tasks']); metrics={}; matrices={s:[[None]*n for _ in range(n+1)] for s in ['exact_match','token_f1','rougeL']}
    results={}
    for stage in range(n+1):
        for tid in required_tasks(stage,n):
            p=run/'evaluations'/f'stage_{stage:02d}'/f'T{tid:02d}_test_predictions.scores.json'
            result=json.loads(p.read_text()); assert result['count']==manifest['tasks'][tid-1]['files']['test']['count']
            results[stage,tid]=result
            for s in matrices: matrices[s][stage][tid-1]=result[s]
    domains=list(dict.fromkeys(t['domain'] for t in manifest['tasks']))
    domain_results={}
    for score,a in matrices.items():
        metrics[score]={'AP':sum(a[n])/n,
            'BWT':sum(a[n][j-1]-a[j][j-1] for j in range(1,n))/(n-1),
            'FWT':sum(a[j-1][j-1]-a[0][j-1] for j in range(2,n+1))/(n-1),
            'aggregation':'equal-weight native tasks; points on 0-100 scale'}
        domain_results[score]={}
        for d in domains:
            ts=[t for t in manifest['tasks'] if t['domain']==d]
            total=sum(t['files']['test']['count'] for t in ts)
            value=sum(a[n][t['task']-1]*t['files']['test']['count'] for t in ts)/total
            domain_results[score][d]={'final':value,'test_count':total,'native_tasks':len(ts)}
        metrics[score]['domain_macro_final']=sum(v['final'] for v in domain_results[score].values())/len(domains)
        with (run/f'{score}_sparse.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['stage']+[t['task_id'] for t in manifest['tasks']]);w.writerows([[i]+row for i,row in enumerate(a)])
    write(run/'score_matrices.json',matrices)
    write(run/'metrics_by_score.json',metrics)
    write(run/'domain_results.json',domain_results)
    write(run/'evaluation_coverage.json',{'tasks':n,'required_groups':4*n-2,'missing_cells':'intentionally not evaluated; null, never zero',
        'ForgetRate':'not measured; sparse cells cannot recover historical peak'})
