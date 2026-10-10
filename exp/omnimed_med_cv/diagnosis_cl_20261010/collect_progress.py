import csv,json,time
from pathlib import Path
import numpy as np
exp=Path(__file__).resolve().parent;root=exp.parents[2]
dest=root/'summary/omnimed_med_cv/diagnosis_cl_20261010';dest.mkdir(parents=True,exist_ok=True)
methods=['seq_lora','migu_lora','olora','sapt_lora','ewc_lora','kd_lora','medqwen','moe_lora','seq_capacity','ra_ldl','own_inc_readout']
names=['ISIC2019','Retinal OCT-C8','Fitzpatrick 17k']
progress=[];table=[];stage_table=[]
for method in methods:
    complete=[];parameter_counts=[]
    planned=[42] if method=='own_inc_readout' else [42,43,44]
    for seed in planned:
        out=exp/'runs'/f'{method}_seed{seed}';status_path=out/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else {'status':'queued'}
        progress.append({**status,'method':method,'seed':seed})
        scores=json.loads((out/'scores.json').read_text()) if (out/'scores.json').exists() else []
        matrix={}
        for stage in scores:
            for row in stage:
                if row['condition']=='image':matrix[(row['stage'],row['task'])]=row;stage_table.append(dict(method=method,seed=seed,**row))
        if status['status']=='completed':
            final=[matrix[(3,name)]['accuracy'] for name in names]
            ba=[matrix[(3,name)]['balanced_accuracy'] for name in names]
            diag=[matrix[(i+1,name)]['accuracy'] for i,name in enumerate(names)]
            forgetting=[max(matrix[(s,names[i])]['accuracy'] for s in range(i+1,4))-final[i] for i in range(2)]
            complete.append(dict(macro_accuracy=np.mean(final),balanced_accuracy=np.mean(ba),
                                 acquisition=np.mean(diag),backward_transfer=np.mean(np.array(final[:2])-diag[:2]),forgetting=np.mean(forgetting)))
        param=out/'selection_stage3.json'
        if param.exists():parameter_counts.append(json.loads(param.read_text()))
    row=dict(method=method,completed_seeds=len(complete),planned_seeds=len(planned),status='completed' if len(complete)==len(planned) else 'in_progress',
             role='本研究：增量线性读出（确定性）' if method=='own_inc_readout' else '本研究诊断监督的顺序LoRA / CL基准' if method=='seq_lora' else '论文方法基线或容量对照')
    for key in ['macro_accuracy','balanced_accuracy','acquisition','backward_transfer','forgetting']:
        vals=[r[key] for r in complete]
        row[key+'_mean']=float(np.mean(vals)) if vals else ''
        row[key+'_seed_sd']=float(np.std(vals,ddof=1)) if len(vals)>1 else ''
    table.append(row)
def savecsv(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
savecsv(dest/'baseline_comparison.csv',table);savecsv(dest/'stage_scores.csv',stage_table)
(dest/'run_status.json').write_text(json.dumps(dict(snapshot_unix_time=time.time(),runs=progress),indent=2))
print(json.dumps([dict(method=r['method'],completed_seeds=r['completed_seeds'],status=r['status']) for r in table],indent=2))
