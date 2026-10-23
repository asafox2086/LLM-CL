"""Single forward-only baseline controller, no automatic model retry."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from baseline import ROOT,REPORT,WORK,BUDGET,check_ready,identity,available,sha,write

def stop_own(proc,expected):
 for sig,timeout in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
  if proc.poll() is not None:return
  actual=identity(proc.pid)
  assert actual==expected and actual['uid']==os.getuid() and actual['pgid']==proc.pid
  os.killpg(proc.pid,sig)
  try:proc.wait(timeout=timeout);return
  except subprocess.TimeoutExpired:pass
 proc.wait()

def main():
 ready=check_ready();review=json.loads((REPORT/'ADMISSION_REVIEW.json').read_text())
 assert review['decision']=='accept_single_fixed_unadapted_baseline_activation'
 assert review['ready_sha256']==sha(REPORT/'READY.json') and review['sources']==ready['sources']
 assert not (REPORT/'OWNER.json').exists() and not (REPORT/'STARTED.json').exists() and not WORK.exists()
 lock=(REPORT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 owner=identity(os.getpid());assert owner['uid']==os.getuid()
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 write(REPORT/'OWNER.json',dict(**owner,boot_id=boot,ready_sha256=sha(REPORT/'READY.json')))
 attempt=0
 while True:
  while not available():
   write(REPORT/'queue/state.json',dict(phase='wait',pid=os.getpid(),only_GPU=1,TEST_QA=0,unadapted_baseline=True));time.sleep(60)
  assert not (REPORT/'STARTED.json').exists();attempt+=1
  with (REPORT/f'activation_{attempt:03d}.log').open('xb') as stream:
   proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('baseline.py')),'worker'],cwd=ROOT,
    stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   expected=identity(proc.pid);assert expected and expected['uid']==os.getuid() and expected['pgid']==proc.pid
   launch=time.monotonic();started=None
   write(REPORT/'queue/state.json',dict(phase='pre_STARTED_validation',pid=os.getpid(),worker=expected,only_GPU=1,TEST_QA=0))
   try:
    while proc.poll() is None:
     if started is None and (REPORT/'STARTED.json').exists():
      started=json.loads((REPORT/'STARTED.json').read_text())
      assert all(started[k]==expected[k] for k in expected)
      assert started['boot_id']==boot and started['ready_sha256']==sha(REPORT/'READY.json') and started['budget_seconds']==BUDGET
      write(REPORT/'queue/state.json',dict(phase='running',pid=os.getpid(),worker=expected,only_GPU=1,TEST_QA=0))
     now=time.monotonic()
     if (started is not None and now>=started['started_monotonic']+BUDGET) or (started is None and now-launch>=300):
      stop_own(proc,expected);raise TimeoutError('Independent deadline; close baseline without retry')
     time.sleep(1)
   except BaseException:
    stop_own(proc,expected);raise
   code=proc.wait()
  if code==75 and not (REPORT/'STARTED.json').exists():
   assert not WORK.exists() and not (REPORT/'FAILURE.json').exists();continue
  assert code==0 and (REPORT/'COMPLETE.json').exists() and not (REPORT/'FAILURE.json').exists(),'Single baseline failed; no retry'
  started=json.loads((REPORT/'STARTED.json').read_text());assert all(started[k]==expected[k] for k in expected)
  assert started['boot_id']==boot and started['ready_sha256']==sha(REPORT/'READY.json')
  bound=time.monotonic()-started['started_monotonic']
  write(REPORT/'ACTUAL_EXIT.json',dict(identity=expected,code=code,observed_elapsed_bound_seconds=bound,
   note='Conservative exit observation bound, includes up to one controller sampling delay.'))
  complete=json.loads((REPORT/'COMPLETE.json').read_text())
  assert complete['model_RUN_seconds']<=bound<=BUDGET
  break
 subprocess.run([sys.executable,str(Path(__file__).with_name('baseline.py')),'verify'],cwd=ROOT,check=True)
 write(REPORT/'queue/state.json',dict(phase='complete_verified',pid=os.getpid(),only_GPU=1,TEST_QA=0,no_automatic_training=True))

if __name__=='__main__':
 try:main()
 except BaseException:
  write(REPORT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),no_repeat=True,not_quality_negative=True))
  write(REPORT/'queue/state.json',dict(phase='closed_failure',pid=os.getpid(),only_GPU=1,TEST_QA=0));raise
