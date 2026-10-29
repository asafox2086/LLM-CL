from pathlib import Path
import subprocess,json,os,time,hashlib,signal
R=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028')
D=R/'summary/seq_s16_four_domain_final_20261029'
OUT=Path('/data2/liyapeng_grp/program/LLMCL/exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/seq_s16_v1')
B=D/'BINDING.json';W=R/'exp/task_gradient_four_domain_final_20261029/seq_s16_worker.py'
PY='/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python'
def ident(pid):
 p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
 return dict(pid=pid,startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
assert not OUT.exists() and not (D/'WORKER_STARTED.json').exists()
env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8',PYTHONUNBUFFERED='1')
with (D/'worker.log').open('x') as log:
 p=subprocess.Popen([PY,'-u',str(W),'--binding',str(B),'--output',str(OUT),'--timeout-seconds','172800','--execute'],cwd=R,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 who=ident(p.pid);assert who['pid']==who['pgid'] and who['uid']==1084
 with (D/'WORKER_STARTED.json').open('x') as f:json.dump(dict(unix=time.time(),identity=who,GPU=1,binding_sha256=hashlib.sha256(B.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(W.read_bytes()).hexdigest(),output=str(OUT)),f,indent=2)
 try:code=p.wait(timeout=172860)
 except subprocess.TimeoutExpired:
  assert ident(p.pid)==who
  os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=30)
  except subprocess.TimeoutExpired:
   assert ident(p.pid)==who
   os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
 with (D/'EXIT.json').open('x') as f:json.dump(dict(unix=time.time(),worker=who,GPU=1,returncode=code,binding_sha256=hashlib.sha256(B.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(W.read_bytes()).hexdigest()),f,indent=2)
 if code:raise SystemExit(code)
