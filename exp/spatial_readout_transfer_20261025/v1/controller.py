from pathlib import Path
import json,os,subprocess,sys,time
from common import RUN,read,sha,write,LIMITS

def launch(phase,gpu):
    assert not (RUN/(phase+'_STARTED.json')).exists()
    log=(RUN/(phase+'.log')).open('xb')
    worker=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py'),phase],env=dict(os.environ,LLMCL_GPU=str(gpu)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return worker,log

def finish(phase,handle):
    worker,log=handle;code=worker.wait();log.close()
    write(phase+'_EXIT.json',dict(pid=worker.pid,returncode=code,unix=time.time()))
    return code==0 and (RUN/(phase+'_COMPLETE.json')).exists()

def main():
    ready=read(RUN/'READY.json');tick=time.monotonic();gpus=ready['phase_GPUs']
    assert finish('fit',launch('fit',gpus['fit']))
    first=launch('train_native',gpus['train_native']);second=launch('train_permuted',gpus['train_permuted'])
    a=finish('train_native',first);b=finish('train_permuted',second);assert a and b
    for phase in ['eval_native','eval_permuted','parent_resource']:assert finish(phase,launch(phase,gpus[phase]))
    write('ALL_EXIT.json',dict(complete=True,ready_sha256=sha(RUN/'READY.json'),controller_seconds=time.monotonic()-tick,
        source_COMPLETE_sha256={p:sha(RUN/(p+'_COMPLETE.json')) for p in LIMITS},new_generations=415,new_scores=0,FIT_backwards=2426,optimizer_updates=174,TEST_QA=0))

if __name__=='__main__':main()
