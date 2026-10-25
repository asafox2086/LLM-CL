from pathlib import Path
import json,os,subprocess,sys,time
run=Path(os.environ['LLMCL_RUN'])
assert not (run/'STARTED.json').exists() and not (run/'EXIT.json').exists()
with (run/'worker.log').open('xb') as log:
    worker=subprocess.Popen([sys.executable,'-u',str(run/'worker.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    code=worker.wait()
with (run/'EXIT.json').open('x') as stream:
    json.dump(dict(pid=worker.pid,returncode=code,unix=time.time()),stream,indent=2);stream.write('\n')
sys.exit(code)
