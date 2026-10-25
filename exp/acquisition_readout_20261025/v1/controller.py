"""Run one GPU reconstruction/readout, then one metadata-only80-score verifier."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time,traceback

RUN=Path(os.environ['LLMCL_RUN'])

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((RUN/name).read_bytes())
def write(name,value):
    with (RUN/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def main():
    tick=time.monotonic();ready=read('READY.json')
    for name,h in ready['sources'].items():assert sha(RUN/name)==h,name
    assert not (RUN/'acquisition_readout_STARTED.json').exists()
    with (RUN/'worker.log').open('x') as log:
        p=subprocess.Popen([sys.executable,'-u',str(RUN/'worker.py')],stdout=log,stderr=subprocess.STDOUT,
                           cwd=RUN,env=os.environ.copy(),start_new_session=True)
        try:code=p.wait(timeout=1850)
        except subprocess.TimeoutExpired:
            # Only this newly-created, isolated child and its own process group.
            os.killpg(p.pid,__import__('signal').SIGTERM);p.wait(timeout=10);raise
    write('acquisition_readout_EXIT.json',dict(unix=time.time(),pid=p.pid,returncode=code))
    assert code==0 and (RUN/'acquisition_readout_COMPLETE.json').exists()
    assert time.monotonic()-tick<2100
    assert not (RUN/'VERIFY_STARTED.json').exists() and not (RUN/'VERIFIED.json').exists()
    p=subprocess.run([sys.executable,'-u',str(RUN/'verify.py')],cwd=RUN,capture_output=True,text=True,
                     timeout=min(180,max(.001,2100-(time.monotonic()-tick))))
    write('VERIFIER_EXIT.json',dict(unix=time.time(),returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
    assert p.returncode==0 and (RUN/'VERIFIED.json').exists(),p.stderr
    assert read('VERIFIED.json')['complete']
    assert time.monotonic()-tick<=2100
    write('CONTROLLER_COMPLETE.json',dict(unix=time.time(),seconds=time.monotonic()-tick,
        VERIFIED_sha256=sha(RUN/'VERIFIED.json'),worker_EXIT_sha256=sha(RUN/'acquisition_readout_EXIT.json'),
        new_model_training=0,worker_runs=1,scoring_runs=1))

if __name__=='__main__':
    try:main()
    except BaseException:
        write('CONTROLLER_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),no_automatic_retry=True));raise
