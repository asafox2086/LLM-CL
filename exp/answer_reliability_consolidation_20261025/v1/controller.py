"""Fixed blind sequence: gate, original, shuffle; then all three evaluations."""
import os
import signal
import subprocess
import sys
import time
from common import RUN,ARMS,LIMITS,read,sha,write

OUTER_SECONDS=18300


def launch(phase,gpu,deadline):
    assert not (RUN/(phase+'_STARTED.json')).exists()
    with (RUN/(phase+'.log')).open('xb') as log:
        worker=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),phase],
            env=dict(os.environ,LLMCL_GPU=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            code=worker.wait(timeout=min(LIMITS[phase]+30,max(0.001,deadline-time.monotonic())))
        except BaseException:
            # Only the isolated group this controller just created is eligible.
            if worker.poll() is None:
                assert os.getpgid(worker.pid)==worker.pid
                os.killpg(worker.pid,signal.SIGTERM)
                try:worker.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid,signal.SIGKILL);worker.wait(timeout=10)
            write(phase+'_EXIT.json',dict(pid=worker.pid,returncode=worker.returncode,unix=time.time(),
                controller_interrupted=True))
            raise
    write(phase+'_EXIT.json',dict(pid=worker.pid,returncode=code,unix=time.time()))
    assert code==0 and (RUN/(phase+'_COMPLETE.json')).exists()


if __name__=='__main__':
    tick=time.monotonic();deadline=tick+OUTER_SECONDS
    try:
        ready=read(RUN/'READY.json')
        phases=['train_'+a for a in ARMS]+['eval_'+a for a in ARMS]
        assert sum(LIMITS.values())==18000
        assert ready['phase_GPUs']=={p:(2 if p.startswith('train_') else 1) for p in phases}
        for phase in phases:launch(phase,ready['phase_GPUs'][phase],deadline)
        seconds=time.monotonic()-tick;assert seconds<=OUTER_SECONDS
        write('ALL_EXIT.json',dict(complete=True,ready_sha256=sha(RUN/'READY.json'),controller_seconds=seconds,
            source_COMPLETE_sha256={p:sha(RUN/(p+'_COMPLETE.json')) for p in phases},
            new_generations=315,new_scores=0,FIT_backwards=3639,current_QA=3639,replay_QA=2658,
            optimizer_updates=261,TEST_QA=0,naive_training_repeated=False))
    except BaseException:
        write('CONTROLLER_FAILURE.json',dict(unix=time.time(),controller_seconds=time.monotonic()-tick,
            traceback=__import__('traceback').format_exc(),no_quality_claim=True))
        raise
