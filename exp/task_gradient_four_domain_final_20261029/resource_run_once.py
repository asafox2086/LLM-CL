from pathlib import Path
import subprocess,json,os,time,hashlib,signal
R=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028')
D=R/'summary/task_gradient_four_domain_final_20261029'
OUT=R/'exp/task_gradient_four_domain_final_20261029/runs/onevision/pct01/resource_s16_v1'
B=D/'RESOURCE_BINDING.json';W=R/'exp/task_gradient_four_domain_final_20261029/resource_worker.py'
PY='/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python'
assert not OUT.exists()
with (D/'resource_worker.log').open('x') as log:
 p=subprocess.Popen([PY,'-u',str(W),'--root',str(R),'--output',str(OUT),'--binding',str(B),'--gpu','1'],cwd=R,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 try:code=p.wait(timeout=660)
 except subprocess.TimeoutExpired:
  assert os.getpgid(p.pid)==p.pid
  os.killpg(p.pid,signal.SIGTERM)
  try:code=p.wait(timeout=30)
  except subprocess.TimeoutExpired:
   assert os.getpgid(p.pid)==p.pid
   os.killpg(p.pid,signal.SIGKILL);code=p.wait(timeout=30)
 start=json.loads((OUT/'STARTED.json').read_text())
 with (OUT/'EXIT.json').open('x') as f:json.dump(dict(unix=time.time(),worker=start['identity'],GPU=1,returncode=code,binding_sha256=hashlib.sha256(B.read_bytes()).hexdigest(),worker_sha256=hashlib.sha256(W.read_bytes()).hexdigest()),f,indent=2)
 if code:raise SystemExit(code)
