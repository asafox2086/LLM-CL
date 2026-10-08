"""Read-only experiment observer; writes separate medical-extension summary tables."""
import argparse
import csv
import json
import os
import signal
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULT=ROOT/'exp/result/medical_generation2_after_superni7_v1'
METHODS=['seq_lora','migu_lora','olora','sapt_lora']

def load(path,default=None):
    if not path.exists(): return {} if default is None else default
    return json.loads(path.read_text())

def atomic_text(path,text):
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(text);temporary.replace(path)

def collect():
    rows=[];task_rows=[]
    for method in METHODS:
        runs=sorted(RESULT.glob(f'{method}_t5large_after7_medical2_seed42_epoch1/*/config.json'))
        if not runs: continue
        run=runs[-1].parent;status=load(run/'status.json');metrics=load(run/'metrics.json')
        supervisor_error=load(run/'supervisor_error.json')
        state=status.get('status','preparing')
        if supervisor_error and state!='completed': state='failed'
        row={'method':method,'status':state,'stage':status.get('stage',9 if state=='completed' else ''),
             'phase':status.get('phase',''),'task':status.get('task',''),
             'medical_stages_completed':metrics.get('medical_stages_completed',0),
             'old_task_AP_before_medical':metrics.get('old_task_AP_before_medical'),
             'old_task_AP':metrics.get('old_task_AP'),'old_task_AP_change':metrics.get('old_task_AP_change'),
             'medical_AP_before_medical':metrics.get('medical_AP_before_medical'),
             'medical_AP':metrics.get('medical_AP'),'medical_AP_gain':metrics.get('medical_AP_gain'),
             'run':str(run.relative_to(ROOT))}
        rows.append(row)
        matrix=load(run/'score_matrix.json');config=load(run/'config.json')
        if matrix:
            for i,task in enumerate(matrix['tasks']):
                scores=[r[i] for r in matrix['rows']]+[None]*(3-len(matrix['rows']))
                task_rows.append({'method':method,'task':task,'group':'old' if task in config['old_tasks'] else 'medical',
                                  'after7':scores[0],'after8':scores[1],'after9':scores[2]})
    def csv_text(items):
        import io
        buf=io.StringIO()
        if items:
            writer=csv.DictWriter(buf,fieldnames=list(items[0]),lineterminator='\n');writer.writeheader();writer.writerows(items)
        return buf.getvalue()
    atomic_text(ROOT/'summary/medical_continuation.csv',csv_text(rows))
    atomic_text(ROOT/'summary/medical_continuation_tasks.csv',csv_text(task_rows))
    fmt=lambda v:'—' if v is None else f'{v:.3f}'
    lines=['# 医学续训进度与结果','',f'更新时间：{time.strftime("%Y-%m-%d %H:%M:%S %Z")}。',
           '', '从各自七任务 checkpoint 继续：任务8 MTS-Dialog → 任务9 IU X-ray。',
           '每新增任务 train/dev/test = 1000/100/200；旧任务保持原500条测试。分数为 ROUGE-L，变化单位为分。',
           '运行中仅展示已完成整阶段的汇总，不把未完成结果当最终结果。','',
           '| 方法 | 状态/阶段 | 医学前旧任务均分 | 当前旧任务均分 | 旧任务变化 | 医学前均分 | 当前医学均分 | 医学变化 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| '+r['method']+f' | {r["status"]} / {r["stage"]} {r["phase"]} | '+' | '.join(fmt(r[k]) for k in ['old_task_AP_before_medical','old_task_AP','old_task_AP_change','medical_AP_before_medical','medical_AP','medical_AP_gain'])+' |')
    lines+=['','## 运行目录','']
    for r in rows:
        lines.append(f'- [{r["method"]}](../{r["run"]}/)：{r["task"] or r["status"]}；[阶段对比表](../{r["run"]}/comparison.md)。')
    lines+=['','详细设置：[协议](../rules/003_medical_continuation.md)；[论文目录](../paper/medical_papers.md)。']
    atomic_text(ROOT/'summary/medical_continuation.md','\n'.join(lines)+'\n')
    return len(rows)==len(METHODS) and all(r['status'] in ['completed','failed','interrupted'] for r in rows)

if __name__=='__main__':
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--interval',type=int,default=60);a=p.parse_args()
    while True:
        done=collect()
        if done or not a.watch: break
        time.sleep(a.interval)
