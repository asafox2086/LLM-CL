from pathlib import Path
import argparse,json,os,hashlib,signal,time,subprocess
R=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028');OLD=R/'exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1';REPORT=R/'summary/runtime_recovery_20261029';OUT=REPORT/'USER_SCOPE_TERMINATION_20261029'
EXPECTED=dict(pid=278958,startticks='794865544',uid=1084,pgid=278958)
def ident(pid):
 try:
  p=Path('/proc')/str(pid);f=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if f[0]=='Z' else dict(pid=pid,startticks=f[19],uid=p.stat().st_uid,pgid=int(f[2]))
 except (FileNotFoundError,ProcessLookupError):return None
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def protected():
 paths=[OLD/'checkpoints/stage_16.pt',OLD/'checkpoints/stage_17.pt',OLD/'checkpoints/stage_16.json',OLD/'checkpoints/stage_17.json',R/'run_v1/FORMAL_FAILURE.json']
 return {str(p):dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino) for p in paths}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');ap.add_argument('--ready',type=Path,required=True);ap.add_argument('--ready-sha',required=True);a=ap.parse_args()
 assert os.getuid()==1084 and not OUT.exists() and a.ready.is_absolute() and sha(a.ready)==a.ready_sha
 started=read(REPORT/'STARTED.json');assert started['worker']==EXPECTED and started['GPU']==1
 assert ident(EXPECTED['pid'])==EXPECTED
 assert started['command'][-1]=='--resume' and 'training_recovery_S17_20261029' in started['command'][-2]
 m=read(OLD/'checkpoints/stage_16.json');assert m['stage']==16 and m['bytes']==1928263780 and m['sha256']=='7a34ff44b996dbdff200d1125b13f4ef6b44328025e4df83f43ec7f30dc56acb'
 assert (OLD/m['path']).stat().st_size==m['bytes']
 assert read(OLD/'stages/stage_16.json')['complete'] and read(OLD/'stages/stage_16.json')['weights_saved']
 assert not (REPORT/'EXIT.json').exists()
 prior=protected();others={str(pid):ident(pid) for pid in [284532,284533,288261]}
 if not a.execute:print(json.dumps(dict(inspect_only=True,worker=EXPECTED,S16=m,protected=prior,other_workers=others)));return
 # Preserve the four-domain source full checkpoint before ending out-of-scope traffic work.
 assert sha(OLD/m['path'])==m['sha256']
 assert ident(EXPECTED['pid'])==EXPECTED and sha(a.ready)==a.ready_sha
 OUT.mkdir();write(OUT/'INTENT.json',dict(unix=time.time(),reason='Human scope change: remove traffic; use committed S16 for four-domain study',classification='USER_SCOPE_CANCELLATION_NOT_METHOD_FAILURE',worker=EXPECTED,ready_path=str(a.ready),ready_sha256=a.ready_sha,S16=m,protected=prior,other_workers=others,no_file_removal=True,no_scientific_source_change=True))
 assert ident(EXPECTED['pid'])==EXPECTED
 os.kill(EXPECTED['pid'],signal.SIGTERM)
 deadline=time.monotonic()+30
 while ident(EXPECTED['pid'])==EXPECTED and time.monotonic()<deadline:time.sleep(1)
 gone=ident(EXPECTED['pid'])!=EXPECTED
 current={pid:ident(int(pid)) for pid in others}
 receipt=dict(unix=time.time(),worker=EXPECTED,worker_absent=gone,signal='SIGTERM to exact worker PID only',protected_unchanged=protected()==prior,other_workers_before=others,other_workers_after=current,classification='USER_SCOPE_CANCELLATION_NOT_METHOD_FAILURE',original_supervisor_failure_retained=True,no_retry=True,no_models_started=True)
 write(OUT/'USER_SCOPE_TERMINATION.json',receipt)
 assert gone and receipt['protected_unchanged']
 print(json.dumps(receipt))
if __name__=='__main__':main()
