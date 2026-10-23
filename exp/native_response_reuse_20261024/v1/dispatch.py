"""One frozen serial pilot; conditional R, independent own-worker deadlines."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from core import ROOT,REPORT,WORK,CASES,BUDGET,sha,write
from common import check_ready

def identity(pid):
 try:
  p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
  uid=int(next(x.split()[1] for x in (p/'status').read_text().splitlines() if x.startswith('Uid:')))
  return dict(pid=pid,startticks=s[19],uid=uid,pgid=int(s[2]))
 except FileNotFoundError:return None

def stop_own(proc,expected):
 """Only a newly spawned isolated process group, verified before each signal."""
 for sig,timeout in ((signal.SIGTERM,10),(signal.SIGKILL,10)):
  if proc.poll() is not None:return
  actual=identity(proc.pid)
  assert actual==expected and actual['uid']==os.getuid() and actual['pgid']==proc.pid
  os.killpg(proc.pid,sig)
  try:proc.wait(timeout=timeout);return
  except subprocess.TimeoutExpired:pass
 # No new case can start before the killed child is reaped.
 proc.wait()

def available():
 sys.path.insert(0,str(ROOT/'analysis'));import onevision_research_device_entry as device
 cards=device.inventory();ours={g for g,c in cards.items() if any(p['uid']==os.getuid() for p in c['owners'])}
 c=cards[1]
 return not c['owners'] and c['memory_MiB']<500 and c['utilization']<10 and not (ours&{0,1}) and len(ours|{1})<=3

def main():
 ready=check_ready(False)
 approval=json.loads((REPORT/'ADMISSION_REVIEW.json').read_text())
 assert approval['decision']=='accept_READY_for_single_pilot_activation' and approval['ready_sha256']==sha(REPORT/'READY.json')
 assert approval['sources']==ready['sources']
 assert Path(sys.executable).resolve()==Path(ready['interpreter']).resolve()
 REPORT.mkdir(parents=True,exist_ok=True)
 lock=(REPORT/'controller.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (REPORT/'OWNER.json').exists(),'No automatic controller resurrection'
 assert not any((REPORT/f'cases/{c}/STARTED.json').exists() for c in CASES)
 assert not WORK.exists()
 owner=identity(os.getpid());assert owner and owner['uid']==os.getuid()
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 write(REPORT/'OWNER.json',dict(**owner,boot_id=boot,ready_sha256=sha(REPORT/'READY.json')))
 total_seconds=0
 for case in CASES:
  if case=='R':
   decision=json.loads((REPORT/'GATE_DECISION.json').read_text())
   if not decision['run_R']:
    write(REPORT/'cases/R/NOT_RUN.json',dict(reason='E_did_not_exceed_Q_point_estimate',not_quality_negative=True));break
  report=REPORT/'cases'/case;attempt=0
  while True:
   assert not (report/'STARTED.json').exists()
   while not available():
    write(REPORT/'queue/state.json',dict(phase='wait',case=case,pid=os.getpid(),only_GPU=1,TEST_QA=0))
    time.sleep(60)
   attempt+=1;report.mkdir(parents=True,exist_ok=True)
   log=report/f'activation_{attempt:03d}.log'
   with log.open('xb') as stream:
    proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('run.py')),'--case',case],cwd=ROOT,
     stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    expected=identity(proc.pid);assert expected and expected['uid']==os.getuid() and expected['pgid']==proc.pid
    launch=time.monotonic();started=None
    write(REPORT/'queue/state.json',dict(phase='pre_STARTED_validation',case=case,pid=os.getpid(),worker=expected,only_GPU=1,TEST_QA=0))
    try:
     while proc.poll() is None:
      marker=report/'STARTED.json'
      if started is None and marker.exists():
       started=json.loads(marker.read_text());assert all(started[k]==expected[k] for k in expected)
       assert started['boot_id']==boot and started['ready_sha256']==sha(REPORT/'READY.json') and started['budget_seconds']==BUDGET[case]
       write(REPORT/'queue/state.json',dict(phase='running',case=case,pid=os.getpid(),worker=expected,only_GPU=1,TEST_QA=0))
      now=time.monotonic()
      if (started is not None and now>=started['started_monotonic']+BUDGET[case]) or (started is None and now-launch>=300):
       stop_own(proc,expected);raise TimeoutError('Independent controller deadline; no extra model time or retry')
      time.sleep(1)
    except BaseException:
     stop_own(proc,expected);raise
    code=proc.wait()
   if code==75 and not (report/'STARTED.json').exists():
    assert not (WORK/case).exists() and not (report/'FAILURE.json').exists();continue
   assert code==0 and (report/'COMPLETE.json').exists() and not (report/'FAILURE.json').exists(),f'case {case} failed; stop whole pilot'
   marker=json.loads((report/'STARTED.json').read_text());assert all(marker[k]==expected[k] for k in expected)
   assert marker['ready_sha256']==sha(REPORT/'READY.json') and marker['boot_id']==boot
   c=json.loads((report/'COMPLETE.json').read_text());bound=time.monotonic()-marker['started_monotonic'];total_seconds+=bound
   write(report/'ACTUAL_EXIT.json',dict(case=case,identity=expected,code=code,observed_elapsed_bound_seconds=bound,
    note='Conservative elapsed bound includes controller observation lag; not exact active compute time'))
   assert c['model_RUN_seconds']<=bound<=BUDGET[case] and total_seconds<=36000
   break
  # Exactly one verifier invocation; a failed invocation stops, never retries.
  subprocess.run([sys.executable,str(Path(__file__).with_name('common.py')),'case','--case',case],cwd=ROOT,check=True)
  if case=='Q':subprocess.run([sys.executable,str(Path(__file__).with_name('common.py')),'gate'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(Path(__file__).with_name('common.py')),'aggregate'],cwd=ROOT,check=True)
 write(REPORT/'queue/state.json',dict(phase='complete_verified',pid=os.getpid(),model_seconds=total_seconds,only_GPU=1,TEST_QA=0))

if __name__=='__main__':
 try:main()
 except BaseException:
  write(REPORT/'CONTROLLER_FAILURE.json',dict(traceback=traceback.format_exc(),no_retry_resume_or_v2=True))
  write(REPORT/'queue/state.json',dict(phase='closed_failure',pid=os.getpid(),only_GPU=1,TEST_QA=0));raise
