from pathlib import Path
import json,os,subprocess,sys,time

run=Path(os.environ['LLMCL_RUN']);log=(run/'diagnostic.log').open('xb')
worker=subprocess.Popen([sys.executable,'-u',str(run/'diagnostic.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
code=worker.wait();log.close()
with (run/'EXIT.json').open('x') as f:json.dump(dict(pid=worker.pid,returncode=code,unix=time.time()),f,indent=2);f.write('\n')
raise SystemExit(code)
