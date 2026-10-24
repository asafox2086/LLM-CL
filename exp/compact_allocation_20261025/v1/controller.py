"""Execute one preregistered allocation study without quality-dependent branches."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

run=Path(os.environ['LLMCL_RUN'])
ready=json.loads((run/'READY.json').read_bytes())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==ready['sources']['controller.py']
cases=['anchor','train','compact_eval','natural1','medical11','geography15','finance16']
for case in cases:
    assert not (run/(case+'_STARTED.json')).exists()
    worker='anchor_worker.py' if case=='anchor' else 'compact_worker.py'
    with (run/(case+'.log')).open('xb') as log:
        process=subprocess.Popen([sys.executable,'-u',str(run/worker),case],env=os.environ,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=process.wait()
    with (run/(case+'_EXIT.json')).open('x') as output:
        json.dump(dict(pid=process.pid,returncode=code,unix=time.time()),output,indent=2);output.write('\n')
    if code:sys.exit(code)
    assert (run/(case+'_COMPLETE.json')).exists()
with (run/'ALL_EXIT.json').open('x') as output:
    json.dump(dict(complete=True,unix=time.time(),no_quality_consumed=True),output,indent=2);output.write('\n')
