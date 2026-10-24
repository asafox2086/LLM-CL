"""Run each frozen-graph intervention once, preserve real exits, no restart."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
run=Path(os.environ['LLMCL_RUN']);ready=json.loads((run/'READY.json').read_bytes())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['sources']['controller.py']
for case in ('parent','candidate'):
 assert not (run/(case+'_STARTED.json')).exists()
 with (run/(case+'.log')).open('xb') as f:
  p=subprocess.Popen([sys.executable,'-u',str(run/'worker.py'),case],env=os.environ,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  code=p.wait()
 with (run/(case+'_EXIT.json')).open('x') as f:json.dump(dict(pid=p.pid,returncode=code,unix=time.time()),f,indent=2);f.write('\n')
 if code:sys.exit(code)
 assert (run/(case+'_COMPLETE.json')).exists()
with (run/'ALL_EXIT.json').open('x') as f:json.dump(dict(unix=time.time(),complete=True),f);f.write('\n')
