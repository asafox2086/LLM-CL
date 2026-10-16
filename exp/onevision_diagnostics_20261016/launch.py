"""Persistent diagnostic launcher; only idle new-server GPU1, no baseline mutation."""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent;REPORT=ROOT/'summary'/HERE.name
REPORT.mkdir(parents=True,exist_ok=True)
lock=(HERE/'.launcher.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while not (ROOT/'model/LLaVA-OneVision-Qwen2-0.5B-OV/download_integrity.json').exists():
    (REPORT/'launch_status.json').write_text(json.dumps({'phase':'waiting_verified_model','proxy':False})+'\n');time.sleep(10)
while True:
    used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
    pids=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if used<500 and util<10 and not pids:break
    (REPORT/'launch_status.json').write_text(json.dumps({'phase':'waiting_idle_gpu1','memory_used_mib':used})+'\n');time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6')
with (REPORT/'worker.log').open('a') as log:
    worker=subprocess.Popen([str(ROOT/'.vision-env/bin/python'),'-u',str(HERE/'run.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (REPORT/'launch_status.json').write_text(json.dumps({'phase':'running','pid':worker.pid,'gpu':1})+'\n')
    code=worker.wait()
if code:
    (REPORT/'launch_status.json').write_text(json.dumps({'phase':'failed','exit_code':code,'log':'worker.log'})+'\n')
    raise SystemExit(code)
(REPORT/'launch_status.json').write_text(json.dumps({'phase':'controls_complete','adaptive_capacity_claimed':False})+'\n')
