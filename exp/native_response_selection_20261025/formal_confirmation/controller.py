"""One TRAIN/resource then four fixed shards; stop only owned peers on failure."""
import os,signal,subprocess,sys,time
from common import RUN,read,sha,write,LIMITS

def identity(pid):
    try:
        root=__import__('pathlib').Path('/proc')/str(pid);stat=(root/'stat').read_text().rsplit(') ',1)[1].split()
        uid=int(next(x.split()[1] for x in (root/'status').read_text().splitlines() if x.startswith('Uid:')))
        return dict(pid=pid,startticks=stat[19],uid=uid,pgid=int(stat[2]))
    except FileNotFoundError:return None

def launch(phase,gpu):
    assert not (RUN/(phase+'_STARTED.json')).exists()
    log=(RUN/(phase+'.log')).open('xb')
    try:worker=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),phase],env=dict(os.environ,LLMCL_GPU=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    except BaseException:log.close();raise
    return worker,log,identity(worker.pid)

def finish(phase,worker,log,who):
    try:code=worker.wait()
    finally:log.close()
    write(phase+'_EXIT.json',dict(pid=worker.pid,returncode=code,unix=time.time()))
    return code==0 and (RUN/(phase+'_COMPLETE.json')).exists()

def stop_owned(children,trigger):
    stopped=[]
    for phase,(worker,log,who) in children.items():
        if worker.poll() is not None:continue
        current=identity(worker.pid)
        if current is None and worker.poll() is not None:continue
        assert who is not None and current==who and who['uid']==os.getuid() and who['pgid']==who['pid']
        try:os.killpg(who['pgid'],signal.SIGTERM)
        except ProcessLookupError:
            assert worker.poll() is not None;continue
        stopped.append(dict(phase=phase,identity=who))
    write('PEER_FAILURE_CANCELLATION.json',dict(trigger=trigger,terminated_owned_groups=stopped,unix=time.time(),no_restarts=True))
    until=time.monotonic()+15
    while any(x[0].poll() is None for x in children.values()) and time.monotonic()<until:time.sleep(.2)
    for phase,(worker,log,who) in children.items():
        if worker.poll() is None:
            assert identity(worker.pid)==who and who['uid']==os.getuid() and who['pgid']==who['pid']
            try:os.killpg(who['pgid'],signal.SIGKILL)
            except ProcessLookupError:assert worker.poll() is not None

if __name__=='__main__':
    ready=read(RUN/'READY.json');tick=time.monotonic()
    for phase in ('train','resource'):
        child=launch(phase,ready['phase_GPUs'][phase]);assert finish(phase,*child),phase
    children={}
    try:
        for i in range(4):children[f'test_{i}']=launch(f'test_{i}',ready['phase_GPUs'][f'test_{i}'])
        while True:
            bad=[phase for phase,(p,log,who) in children.items() if p.poll() is not None and (p.returncode!=0 or not (RUN/(phase+'_COMPLETE.json')).exists())]
            if bad:stop_owned(children,bad);break
            if all(p.poll() is not None for p,log,who in children.values()):break
            time.sleep(.5)
    except BaseException:
        if children:stop_owned(children,['controller_exception'])
        raise
    finally:
        outcomes={phase:finish(phase,*child) for phase,child in children.items()}
    assert len(outcomes)==4 and all(outcomes.values()),outcomes
    write('ALL_EXIT.json',dict(complete=True,ready_sha256=sha(RUN/'READY.json'),controller_seconds=time.monotonic()-tick,
        source_COMPLETE_sha256={p:sha(RUN/(p+'_COMPLETE.json')) for p in LIMITS},new_generations=21459,new_scores=0,TRAIN_backwards=1313,optimizer_updates=93,TEST_QA=21454))
