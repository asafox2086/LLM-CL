"""Serial GPU1 study queue after the live sequential research process completes."""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/onevision_mechanisms_20261016';REPORT.mkdir(parents=True,exist_ok=True)
PLAN=[('format','onevision_knowledge_intervention.py',['--mode','format']),
      ('task_domain','onevision_task_domain_intervention.py',[]),
      ('knowledge','onevision_knowledge_intervention.py',['--mode','knowledge'])]

def write(path,value):
    temporary=path.with_suffix('.json.partial');temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temporary.replace(path)

def current_boot():return Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def main():
    lock=(REPORT/'queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=REPORT/'queue_state.json';state=json.loads(path.read_text()) if path.exists() else {'jobs':{},'phase':'waiting_sequential'}
    for name,script,args in PLAN:
        state['jobs'].setdefault(name,{'status':'waiting','script':script,'args':args,'attempts':0})
    write(path,state)
    while True:
        research=ROOT/'summary/onevision_diagnostics_20261016/status.json'
        status=json.loads(research.read_text()) if research.exists() else {}
        if status.get('phase')=='sequential_recovery_complete':break
        # A stale status is never treated as completion; inspect a concrete service PID.
        unit=subprocess.run(['systemctl','--user','show','llmcl-onevision-research.service','-p','MainPID','--value'],capture_output=True,text=True)
        pid=int(unit.stdout.strip() or '0');cmd=Path('/proc',str(pid),'cmdline')
        live=pid>0 and cmd.exists() and b'onevision_research_worker.py' in cmd.read_bytes()
        state.update(phase='waiting_live_sequential' if live else 'waiting_research_recovery',research_pid=pid,research_phase=status.get('phase'))
        write(path,state);time.sleep(30)
    for name,script,args in PLAN:
        job=state['jobs'][name]
        complete=REPORT/(name+'_complete.json')
        if complete.exists() and json.loads(complete.read_text()).get('complete'):
            proof=json.loads(complete.read_text())
            for key,expected in proof['source_sha256'].items():
                source=ROOT/'data/OneVisionMechanisms/20261016/knowledge16.json' if key=='knowledge16.json' else ROOT/key
                assert hashlib.sha256(source.read_bytes()).hexdigest()==expected,'Completed study source changed; explicit new attempt required'
            job['status']='completed';write(path,state);continue
        if job.get('status')=='needs_attention':continue
        # Old running state after a boot is not proof the old PID is still ours.
        if job.get('status')=='running':
            old=Path('/proc',str(job['pid']),'cmdline')
            if job.get('boot_id')==current_boot() and old.exists() and str(ROOT/'analysis'/script).encode() in old.read_bytes():
                raise RuntimeError('Existing live study worker: do not duplicate it')
            job['interrupted_previous_pid']=job.get('pid')
        while True:
            used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(','))
            owners=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
            if used<500 and util<10 and not owners:break
            state['phase']='waiting_idle_gpu1';write(path,state);time.sleep(15)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1')
        with (REPORT/(name+'_nohup.log')).open('a') as log:
            worker=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u',str(ROOT/'analysis'/script),*args],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            job.update(status='running',pid=worker.pid,boot_id=current_boot(),attempts=job['attempts']+1)
            state.update(phase='running',active_job=name);write(path,state)
            code=worker.wait()
        if code or not complete.exists():
            job.update(status='needs_attention',exit_code=code,log=str((REPORT/(name+'_nohup.log')).relative_to(ROOT)))
        else:job['status']='completed'
        write(path,state)
    state['phase']='studies_finished_or_need_attention';write(path,state)

if __name__=='__main__':main()
