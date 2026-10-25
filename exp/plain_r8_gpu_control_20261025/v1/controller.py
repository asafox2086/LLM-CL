import os,subprocess,sys,time
from common import RUN,read,sha,write,LIMITS

def launch(phase,gpu):
    assert not (RUN/(phase+'_STARTED.json')).exists()
    with (RUN/(phase+'.log')).open('xb') as log:
        worker=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),phase],env=dict(os.environ,LLMCL_GPU=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=worker.wait()
    write(phase+'_EXIT.json',dict(pid=worker.pid,returncode=code,unix=time.time()))
    assert code==0 and (RUN/(phase+'_COMPLETE.json')).exists()

if __name__=='__main__':
    ready=read(RUN/'READY.json');tick=time.monotonic()
    for phase in ['train_qv','eval_qv']:launch(phase,ready['phase_GPUs'][phase])
    write('ALL_EXIT.json',dict(complete=True,ready_sha256=sha(RUN/'READY.json'),controller_seconds=time.monotonic()-tick,
        source_COMPLETE_sha256={p:sha(RUN/(p+'_COMPLETE.json')) for p in LIMITS},new_generations=105,new_scores=0,FIT_backwards=1213,optimizer_updates=87,TEST_QA=0))
