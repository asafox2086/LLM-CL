"""One isolated carrier worker; resource waiting is not a model attempt."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,HERE,REPORT,WORK,BUDGET,ready,identity,write,sha,verify

def inventory():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    cards={}
    for line in raw.splitlines():
        if not line.strip():continue
        i,u,m,z=map(str.strip,line.split(','));cards[int(i)]=dict(uuid=u,memory_MiB=int(m),utilization=int(z),owners=[])
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        if not line.strip():continue
        p,u=map(str.strip,line.split(','));who=identity(int(p))
        if who is not None:
            card=[x for x in cards.values() if x['uuid']==u];assert len(card)==1;card[0]['owners'].append(who)
    return cards

def available():
    return [g for g,c in sorted(inventory().items()) if not c['owners'] and c['memory_MiB']<500 and c['utilization']<10]

def reclaim(proc,expected):
    actual=identity(proc.pid)
    if actual is None:return
    assert actual==expected and actual['uid']==os.getuid() and actual['pgid']==actual['pid']
    os.killpg(actual['pgid'],signal.SIGTERM)
    try:proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        assert identity(proc.pid)==expected
        os.killpg(expected['pgid'],signal.SIGKILL);proc.wait(timeout=15)

def dispatch():
    r=ready();REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not any((REPORT/n).exists() for n in ('STARTED.json','COMPLETE.json','FAILURE.json','CONTROLLER_FAILURE.json','VERIFIED.json'))
    while True:
        cards=available();gpu=next((g for g in cards if g in r['allowed_GPUs']),None)
        if gpu is None:
            write(REPORT/'queue.json',dict(phase='wait',pid=os.getpid(),allowed_GPUs=r['allowed_GPUs'],one_condition=True,TEST_QA=0))
            time.sleep(60);continue
        with (REPORT/'worker.log').open('ab') as log:
            proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            who=None
            try:
                for _ in range(50):
                    who=identity(proc.pid)
                    if who is not None:break
                    if proc.poll() is not None:break
                    time.sleep(.02)
                if who is None:raise RuntimeError('Worker exited before identity binding')
                assert who['uid']==os.getuid() and who['pgid']==who['pid']
                write(REPORT/'queue.json',dict(phase='running',pid=os.getpid(),worker=who,GPU=gpu,TEST_QA=0))
                first=None;startup=time.monotonic()
                while proc.poll() is None:
                    marker=REPORT/'STARTED.json'
                    if marker.exists():
                        first=json.loads(marker.read_bytes());assert all(first[k]==who[k] for k in who)
                        if time.monotonic()>=first['started_monotonic']+BUDGET:
                            reclaim(proc,who);raise TimeoutError('Independent fixed model deadline reached')
                    elif time.monotonic()-startup>60:
                        reclaim(proc,who);raise TimeoutError('Worker did not resolve startup admission')
                    remaining=(first['started_monotonic']+BUDGET-time.monotonic()) if first is not None else .2
                    time.sleep(max(.001,min(.2,remaining)))
                code=proc.wait();ended=time.monotonic()
            finally:
                if proc.poll() is None:
                    if who is not None and identity(proc.pid)==who and who['uid']==os.getuid() and who['pgid']==who['pid']:
                        reclaim(proc,who)
                    else:
                        write(REPORT/'NEEDS_ATTENTION.json',dict(pid=proc.pid,expected=who,actual=identity(proc.pid),unverified_identity_not_signalled=True))
        if code==75 and not (REPORT/'STARTED.json').exists():
            write(REPORT/'queue.json',dict(phase='wait',pid=os.getpid(),resource_change_before_attempt=True,TEST_QA=0))
            time.sleep(60);continue
        assert first is not None or (REPORT/'STARTED.json').exists()
        first=first or json.loads((REPORT/'STARTED.json').read_bytes())
        elapsed=ended-first['started_monotonic']
        write(REPORT/'ACTUAL_EXIT.json',dict(worker=who,returncode=code,observed_elapsed_seconds=elapsed,identity_after=identity(proc.pid)))
        if code!=0 or elapsed>BUDGET or not (REPORT/'COMPLETE.json').exists():raise RuntimeError('Fixed carrier failed or incomplete; no retry')
        verify()
        write(REPORT/'queue.json',dict(phase='complete_verified',worker=None,controller=None,TEST_QA=0));return

if __name__=='__main__':
    try:dispatch()
    except Exception:
        write(REPORT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True))
        write(REPORT/'queue.json',dict(phase='closed_failure',worker=None,controller=None,TEST_QA=0));raise
