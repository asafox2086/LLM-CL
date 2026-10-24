"""One free OLD card, three isolated serial phases, one final verifier."""
import fcntl,json,os,signal,subprocess,sys,time,traceback
from pathlib import Path
from common import ROOT,HERE,REPORT,WORK,CASES,BUDGETS,identity,write,ready,verify,sha


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
    who=identity(proc.pid)
    if who is None:return
    assert who==expected and who['uid']==os.getuid() and who['pgid']==who['pid']
    os.killpg(who['pgid'],signal.SIGTERM)
    try:proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        assert identity(proc.pid)==expected;os.killpg(who['pgid'],signal.SIGKILL);proc.wait(timeout=15)


def dispatch():
    ready();REPORT.mkdir(parents=True,exist_ok=True)
    admission=json.loads((REPORT/'ADMISSION_REVIEW.json').read_bytes())
    assert admission['model_admission'] is True and admission['ready_sha256']==sha(REPORT/'READY.json')
    lock=(REPORT/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not any((REPORT/(c+'_STARTED.json')).exists() for c in CASES) and not (REPORT/'VERIFIED.json').exists()
    selected=None
    for case in CASES:
        while True:
            free=available()
            if selected is not None:free=[g for g in free if g==selected]
            if not free:
                write(REPORT/'queue.json',dict(phase='wait',case=case,controller=identity(os.getpid()),TEST_QA=0));time.sleep(60);continue
            gpu=free[0]
            with (REPORT/(case+'.log')).open('ab') as log:
                proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),case,str(gpu)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                who=None;first=None;startup=time.monotonic()
                try:
                    for _ in range(50):
                        who=identity(proc.pid)
                        if who is not None or proc.poll() is not None:break
                        time.sleep(.02)
                    assert who is not None and who['uid']==os.getuid() and who['pgid']==who['pid']
                    write(REPORT/'queue.json',dict(phase='running',case=case,controller=identity(os.getpid()),worker=who,GPU=gpu,TEST_QA=0))
                    while proc.poll() is None:
                        p=REPORT/(case+'_STARTED.json')
                        if p.exists():
                            first=json.loads(p.read_bytes());assert all(first[k]==who[k] for k in who)
                            if time.monotonic()>=first['started_monotonic']+BUDGETS[case]:reclaim(proc,who);raise TimeoutError('Independent model deadline')
                        elif time.monotonic()-startup>60:reclaim(proc,who);raise TimeoutError('Unresolved startup')
                        remaining=first['started_monotonic']+BUDGETS[case]-time.monotonic() if first is not None else .2
                        time.sleep(max(.001,min(.2,remaining)))
                    code=proc.wait();ended=time.monotonic()
                finally:
                    if proc.poll() is None:reclaim(proc,who)
            if code==75 and first is None and not (REPORT/(case+'_STARTED.json')).exists():time.sleep(60);continue
            first=first or json.loads((REPORT/(case+'_STARTED.json')).read_bytes());elapsed=ended-first['started_monotonic']
            write(REPORT/(case+'_EXIT.json'),dict(worker=who,returncode=code,observed_elapsed_seconds=elapsed,identity_after=identity(proc.pid)))
            assert code==0 and elapsed<=BUDGETS[case] and (REPORT/(case+'_COMPLETE.json')).exists(),'Fixed phase failed; no retry'
            selected=gpu;break
    verify();write(REPORT/'queue.json',dict(phase='complete_verified',controller=None,worker=None,TEST_QA=0))


if __name__=='__main__':
    try:dispatch()
    except Exception:
        write(REPORT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True));write(REPORT/'queue.json',dict(phase='closed_failure'));raise
