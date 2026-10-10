"""Collect report snapshots as this experiment's worker processes finish."""
import json,os,subprocess,sys,time
from pathlib import Path
exp=Path(__file__).resolve().parent
pids=[int(p.read_text()) for p in exp.glob('worker_gpu*.pid')]
last=None
while True:
    states={str(p.relative_to(exp)):p.read_text() for p in (exp/'runs').glob('*/status.json')}
    completed=tuple(sorted((name,json.loads(value).get('status')) for name,value in states.items()))
    alive=[]
    for pid in pids:
        try:os.kill(pid,0);alive.append(pid)
        except ProcessLookupError:pass
    if completed!=last or not alive:
        subprocess.run([sys.executable,str(exp/'collect_progress.py')],check=True)
        last=completed
    if not alive:break
    time.sleep(20)
print('All registered worker processes exited; report snapshot collected.',flush=True)
