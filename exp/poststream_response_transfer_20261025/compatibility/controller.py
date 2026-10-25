"""One fixed GPU pair, with no outcome-dependent phase or retry."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
run=Path(os.environ['LLMCL_RUN'])
ready=json.loads((run/'READY.json').read_bytes())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['sources']['controller.py']
for case in ['CE_train','KD_train','CE_eval','KD_eval']:
    assert not (run/(case+'_STARTED.json')).exists()
    with (run/(case+'.log')).open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u',str(run/'worker.py'),case],env=os.environ,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=process.wait()
    with (run/(case+'_EXIT.json')).open('x') as f:
        json.dump(dict(pid=process.pid,returncode=code,unix=time.time()),f,indent=2);f.write('\n')
    if code:sys.exit(code)
    assert (run/(case+'_COMPLETE.json')).exists()
with (run/'ALL_EXIT.json').open('x') as f:
    json.dump(dict(complete=True,unix=time.time(),no_quality_consumed=True),f,indent=2);f.write('\n')
