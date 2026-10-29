from pathlib import Path
import json,hashlib,subprocess,os,time,signal,traceback
ROOT=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028')
REPORT=ROOT/'summary/task_gradient_four_domain_final_20261029'
WORKER=ROOT/'exp/task_gradient_four_domain_final_20261029/worker.py'
BINDING=REPORT/'BINDING.json'
OUT=ROOT/'exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/final_s16_v1'
PYTHON='/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def ident(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if s[0]=='Z' else dict(pid=pid,startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
 except (FileNotFoundError,ProcessLookupError):return None
b=read(BINDING);assert sha(WORKER)==b['worker_sha256']
assert read(REPORT/'READY.json')['binding_sha256']==sha(BINDING)
assert not OUT.exists() and not (REPORT/'SUPERVISOR_STARTED.json').exists()
assert not subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=15).strip()
owner=ident(os.getpid());assert owner and owner['pid']==owner['pgid'] and owner['uid']==1084
command=[PYTHON,'-u',str(WORKER),'--binding',str(BINDING),'--output',str(OUT),'--timeout-seconds','172800','--execute']
env=dict(os.environ,LLMCL_ROOT=str(ROOT),LLMCL_HOST='old2080',LLMCL_BACKBONE='onevision',CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
child=None;who=None
write(REPORT/'SUPERVISOR_STARTED.json',dict(unix=time.time(),identity=owner,binding_sha256=sha(BINDING),worker_sha256=sha(WORKER),supervisor_sha256=sha(Path(__file__)),command=command,GPU=1))
try:
 with (REPORT/'worker.log').open('x') as log:
  child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  who=ident(child.pid);assert who and who['uid']==1084 and who['pgid']==who['pid']
  write(REPORT/'WORKER_STARTED.json',dict(unix=time.time(),identity=who,supervisor=owner,command=command,GPU=1))
  code=child.wait(timeout=172800)
 write(REPORT/'WORKER_EXIT.json',dict(unix=time.time(),identity=who,returncode=code,identity_absent=ident(child.pid)!=who))
 assert code==0 and ident(child.pid)!=who and (OUT/'COMPLETE.json').is_file() and not (OUT/'FAILURE.json').exists()
 write(REPORT/'SUPERVISOR_COMPLETE.json',dict(unix=time.time(),worker=who,supervisor=owner,returncode=0,complete_sha256=sha(OUT/'COMPLETE.json'),quality_not_independently_verified=True,resources_pending=True))
except BaseException:
 failure=traceback.format_exc();cleanup=[]
 try:
  if child is not None:
   if child.poll() is None:
    # Signal only the exact child session created above, never another PID owner.
    assert who and who['pid']==child.pid and who['pgid']==child.pid and who['uid']==owner['uid']
    assert ident(child.pid)==who,'Child identity changed; refusing signal'
    os.killpg(who['pgid'],signal.SIGTERM);cleanup.append('SIGTERM_own_child_pgid')
    try:child.wait(timeout=30)
    except subprocess.TimeoutExpired:
     assert ident(child.pid)==who,'Child identity changed; refusing SIGKILL'
     os.killpg(who['pgid'],signal.SIGKILL);cleanup.append('SIGKILL_own_child_pgid')
     child.wait(timeout=30)
   code=child.wait(timeout=0)
   if not (REPORT/'WORKER_EXIT.json').exists():
    write(REPORT/'WORKER_EXIT.json',dict(unix=time.time(),identity=who,returncode=code,identity_absent=ident(child.pid)!=who,cleanup=cleanup))
 except BaseException:
  cleanup.append(dict(cleanup_failure=traceback.format_exc()))
 write(REPORT/'SUPERVISOR_FAILURE.json',dict(unix=time.time(),worker=who,supervisor=owner,traceback=failure,cleanup=cleanup,actual_returncode=child.returncode if child else None,actual_worker=ident(child.pid) if child else None))
 raise
