from pathlib import Path
import json,os,subprocess,sys,time
run=Path(os.environ['LLMCL_RUN'])
assert not (run/'STARTED.json').exists()
with (run/'worker.log').open('xb') as log:
    worker=subprocess.Popen([sys.executable,'-u',str(run/'worker.py')],env=os.environ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    code=worker.wait()
with (run/'EXIT.json').open('x') as f:json.dump(dict(pid=worker.pid,returncode=code,unix=time.time()),f,indent=2);f.write('\n')
if code:sys.exit(code)
assert (run/'COMPLETE.json').exists()
