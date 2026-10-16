"""Small CPU-only coordinator: independent cap workers, then fixed confirmation.

Waits for the deterministic native gate/reference and idle GPU1. No active
mechanism/native queue source is edited or restarted by this coordinator.
"""
import fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_diagnostics_20261016/capacity'
PLAN=[('knowledge_sanity','onevision_knowledge_encoding_probe.py',[]),
      ('cap1','onevision_capacity_probe.py',['--cap','1']),('cap2','onevision_capacity_probe.py',['--cap','2']),
      ('cap3','onevision_capacity_probe.py',['--cap','3']),('cap5','onevision_capacity_probe.py',['--cap','5']),
      ('cap1_full_confirm','onevision_capacity_probe.py',['--cap','1','--confirm-full']),
      ('cap3_full_confirm','onevision_capacity_probe.py',['--cap','3','--confirm-full'])]


def write(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'pid':os.getpid(),'phase':'waiting_native_reference','jobs':{n:{'status':'waiting'} for n,_,_ in PLAN}}
    path=REPORT/'queue_state.json';write(path,state)
    ref=ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/complete.json'
    attention=ROOT/'summary/onevision_diagnostics_20261016/deterministic_native/needs_attention.json'
    while not ref.exists():
        if attention.exists():
            state['phase']='native_reference_needs_attention';write(path,state);return
        time.sleep(30)
    for name,script,args in PLAN:
        complete=REPORT/(name+'_complete.json')
        while not complete.exists():
            used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            owners=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if used<500 and util<10 and not owners:break
            state.update(phase='waiting_idle_gpu1',active_job=name,owner_pids=owners.splitlines());write(path,state);time.sleep(15)
        if (REPORT/(name+'_needs_attention.json')).exists():
            if name=='knowledge_sanity':
                state['jobs'][name]['status']='needs_attention';write(path,state);continue
            state.update(phase='needs_attention',active_job=name);write(path,state);return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1')
        with (REPORT/(name+'_nohup.log')).open('a') as log:
            worker=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u',str(ROOT/'analysis'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            state['jobs'][name]={'status':'running','pid':worker.pid}
            state.update(phase='running',active_job=name);state.pop('owner_pids',None);write(path,state)
            code=worker.wait()
        if code or not complete.exists() or (REPORT/(name+'_needs_attention.json')).exists():
            state['jobs'][name].update(status='needs_attention',exit_code=code)
            if name=='knowledge_sanity':write(path,state);continue
            state['phase']='needs_attention';write(path,state);return
        state['jobs'][name]['status']='completed';write(path,state)
    state['phase']='capacity_study_complete_with_auxiliary_attention' if state['jobs']['knowledge_sanity']['status']=='needs_attention' else 'capacity_study_complete';write(path,state)


if __name__=='__main__':main()
