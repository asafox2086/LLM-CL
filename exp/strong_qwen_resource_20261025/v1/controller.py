"""Run one original-Qwen resource worker and its metadata verifier once."""
from pathlib import Path
import json,os,signal,subprocess,sys,time,traceback

RUN=Path(os.environ['LLMCL_RUN'])

def write(name,value):
    with (RUN/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')

if __name__=='__main__':
    tick=time.monotonic()
    try:
        assert not any((RUN/n).exists() for n in ('STARTED.json','EXIT.json','VERIFIED.json','CONTROLLER_FAILURE.json'))
        with (RUN/'worker.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py')],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:code=child.wait(timeout=630)
            except BaseException:
                if child.poll() is None:
                    assert os.getpgid(child.pid)==child.pid
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=5)
                write('EXIT.json',dict(pid=child.pid,returncode=child.returncode,unix=time.time(),controller_interrupted=True))
                raise
        write('EXIT.json',dict(pid=child.pid,returncode=code,unix=time.time()))
        assert code==0 and (RUN/'COMPLETE.json').exists()
        p=subprocess.run([sys.executable,'-u',str(RUN/'verify.py')],capture_output=True,text=True,timeout=30)
        write('VERIFIER_EXIT.json',dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr,unix=time.time()))
        assert p.returncode==0 and (RUN/'VERIFIED.json').exists(),p.stderr
        write('ALL_EXIT.json',dict(complete=True,controller_seconds=time.monotonic()-tick,new_scores=0,new_generations=5))
    except BaseException:
        write('CONTROLLER_FAILURE.json',dict(unix=time.time(),controller_seconds=time.monotonic()-tick,traceback=traceback.format_exc()))
        raise
