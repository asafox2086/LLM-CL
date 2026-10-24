"""Record the one worker's actual exit; never retry."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
run=Path(os.environ['LLMCL_RUN'])
ready=json.loads((run/'READY.json').read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['controller_sha256']
assert not (run/'STARTED.json').exists() and not (run/'EXIT.json').exists()
with (run/'worker.log').open('xb') as f:
 p=subprocess.Popen([sys.executable,'-u',str(run/'worker.py')],stdout=f,stderr=subprocess.STDOUT,env=os.environ,start_new_session=True)
 code=p.wait()
with (run/'EXIT.json').open('x') as f:json.dump(dict(pid=p.pid,returncode=code,unix=time.time()),f,indent=2);f.write('\n')
sys.exit(code)
