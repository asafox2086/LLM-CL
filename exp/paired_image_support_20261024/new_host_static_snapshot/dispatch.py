"""One serial diagnostic stream; only pre-STARTED resource changes can wait."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from common import ROOT,HERE,REPORT,WORK,STATES,BUDGET,TOTAL_BUDGET,available,check_ready,identity,sha,write

def stop_own(proc,expected):
    for sig,timeout in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
        if proc.poll() is not None:return
        assert identity(proc.pid)==expected and expected['uid']==os.getuid() and expected['pgid']==proc.pid
        os.killpg(proc.pid,sig)
        try:proc.wait(timeout=timeout);return
        except subprocess.TimeoutExpired:pass
    proc.wait()

def main():
    ready=check_ready();review=json.loads((REPORT/'MODEL_ADMISSION_REVIEW.json').read_bytes())
    assert review['decision']=='accept_single_four_state_diagnostic_activation'
    assert review['ready_sha256']==sha(REPORT/'MODEL_READY.json') and review['sources']==ready['sources']
    assert not (REPORT/'MODEL_OWNER.json').exists() and not WORK.exists()
    assert not any((REPORT/(s+'_STARTED.json')).exists() for s in STATES)
    lock=(REPORT/'model_controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    owner=identity(os.getpid());assert owner['uid']==os.getuid()
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    write(REPORT/'MODEL_OWNER.json',dict(**owner,boot_id=boot,ready_sha256=sha(REPORT/'MODEL_READY.json')))
    total=0
    for state in STATES:
        attempts=0;marker=REPORT/(state+'_STARTED.json')
        while True:
            while not available()[0]:
                write(REPORT/'model_queue/state.json',dict(phase='wait',pid=os.getpid(),state=state,only_GPU=1,training_updates=0,TEST_QA=0));time.sleep(60)
            assert not marker.exists();attempts+=1
            with (REPORT/(state+f'_activation_{attempts:03d}.log')).open('xb') as stream:
                proc=subprocess.Popen([sys.executable,str(HERE/'worker.py'),state],cwd=ROOT,
                    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                expected=None;launch=time.monotonic();started=None
                try:
                    expected=identity(proc.pid);assert expected and expected['uid']==os.getuid() and expected['pgid']==proc.pid
                    write(REPORT/'model_queue/state.json',dict(phase='pre_STARTED_validation',pid=os.getpid(),state=state,worker=expected,only_GPU=1))
                    while proc.poll() is None:
                        if started is None and marker.exists():
                            started=json.loads(marker.read_bytes())
                            assert all(started[k]==v for k,v in expected.items()) and started['boot_id']==boot
                            assert started['ready_sha256']==sha(REPORT/'MODEL_READY.json') and started['budget_seconds']==BUDGET
                            write(REPORT/'model_queue/state.json',dict(phase='running',pid=os.getpid(),state=state,worker=expected,only_GPU=1))
                        now=time.monotonic()
                        if (started is not None and now>=started['started_monotonic']+BUDGET) or (started is None and now-launch>=300):
                            stop_own(proc,expected);raise TimeoutError('Independent own-worker diagnostic deadline')
                        remaining=started['started_monotonic']+BUDGET-now if started is not None else 300-(now-launch)
                        time.sleep(min(1,max(.001,remaining)))
                except BaseException:
                    if expected and expected['uid']==os.getuid() and expected['pgid']==proc.pid:
                        stop_own(proc,expected)
                    elif proc.poll() is None:
                        write(REPORT/(state+'_OWNERSHIP_NEEDS_ATTENTION.json'),dict(pid=proc.pid,expected=expected,
                            reason='Child identity not confirmed; no blind signal was sent.'))
                    raise
                code=proc.wait()
            if code==75 and not marker.exists():
                assert not (WORK/state).exists() and not (REPORT/(state+'_FAILURE.json')).exists();continue
            assert code==0 and (REPORT/(state+'_COMPLETE.json')).exists() and not (REPORT/(state+'_FAILURE.json')).exists(),'Fixed diagnostic failed; no repeat'
            started=json.loads(marker.read_bytes());assert all(started[k]==v for k,v in expected.items())
            assert started['boot_id']==boot and started['ready_sha256']==sha(REPORT/'MODEL_READY.json')
            bound=time.monotonic()-started['started_monotonic'];total+=bound
            write(REPORT/(state+'_ACTUAL_EXIT.json'),dict(identity=expected,code=code,observed_elapsed_bound_seconds=bound))
            c=json.loads((REPORT/(state+'_COMPLETE.json')).read_bytes());assert c['model_RUN_seconds']<=bound<=BUDGET
            assert total<=TOTAL_BUDGET
            break
    subprocess.run([sys.executable,str(HERE/'common.py'),'verify'],cwd=ROOT,check=True)
    write(REPORT/'model_queue/state.json',dict(phase='complete_verified',pid=os.getpid(),only_GPU=1,no_automatic_next_model=True))

if __name__=='__main__':
    try:main()
    except BaseException:
        write(REPORT/'MODEL_CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry=True,not_quality_negative=True))
        write(REPORT/'model_queue/state.json',dict(phase='closed_failure',pid=os.getpid(),only_GPU=1));raise
