from pathlib import Path
import os,json,hashlib,fcntl,subprocess,importlib.util,time,argparse,traceback
ROOT=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_');FRAME=ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027';OUT=ROOT/'summary/runtime_recovery_20261029/TWO_BASELINE_SUPERVISOR'
JOBS=[dict(method='PaRSP',condition='pct10',completed_stage=6,gpu=0,run_dir='exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/PaRSP/recovery_io_20261029_S06',sha='8c11f1d6b180bd96b636ff7a59ddb676605593380a221cc29731748649f16439'),dict(method='OSFT',condition='pct10',completed_stage=0,gpu=2,run_dir='exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/OSFT/training_20261027_142732',sha='48697fbe06cc7cb081f33e16b5d4f2473f66680913259e27d9d86306a2f29678')]
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def ident(pid):
 try:
  p=Path('/proc')/str(pid);x=(p/'stat').read_text().rsplit(') ',1)[1].split()
  return None if x[0]=='Z' else dict(pid=pid,startticks=x[19],uid=p.stat().st_uid)
 except FileNotFoundError:return None
def check():
 assert os.getuid()==1084 and not OUT.exists()
 sources={p.name:sha(p) for p in FRAME.glob('*.py')}
 for j in JOBS:
  u=ROOT/j['run_dir'];c=read(u/'config.json');m=read(u/'checkpoints/latest.json')
  assert c['execution_source_sha256']==sources and c['method']==j['method'] and c['condition']=='pct10'
  assert m['stage']==j['completed_stage'] and m['sha256']==j['sha']
  assert (u/m['path']).stat().st_size==m['bytes']
  for pp in Path('/proc').glob('[0-9]*'):
   try:
    if pp.stat().st_uid==1084:assert str(u).encode() not in (pp/'cmdline').read_bytes().split(b'\0')
   except (FileNotFoundError,PermissionError,ProcessLookupError):pass
  assert not subprocess.check_output(['nvidia-smi',f"--id={j['gpu']}",'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  if j['method']=='PaRSP':assert read(u/'IO_RECOVERY_SOURCE.json')['CP_SHA_verified']
  else:
   st=read(u/'run_status.json');assert st['status']=='paused_checkpoint'
   assert sha(u/st['pause_receipt'])==st['pause_receipt_sha256']=='b2f949b2b89777df04b59ca839c33cdd03c8105bd9679a5ddbd63b2e8d641b1c'
   assert not (u/'RESUME_REQUEST.json').exists()
 return sources
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args();sources=check()
 if not a.execute:print(json.dumps(dict(inspect=True,jobs=JOBS)));return
 handles=[]
 for p in [ROOT/'summary/two_backbone_pct10_20261025/dispatch_old.lock',FRAME/'.pipeline_old_pct10_onevision.lock']:
  h=p.open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);handles.append(h)
 check()
 for j in JOBS:
  u=ROOT/j['run_dir'];assert sha(u/read(u/'checkpoints/latest.json')['path'])==j['sha']
 OUT.mkdir();write(OUT/'STARTED.json',dict(unix=time.time(),identity=ident(os.getpid()),jobs=JOBS,sources=sources,scientific_changes=False,allocator=False))
 os.environ.update(LLMCL_ROOT=str(ROOT),LLMCL_HOST='old_pct10_onevision',LLMCL_BACKBONE='onevision')
 sp=importlib.util.spec_from_file_location('original_pipeline',FRAME/'pipeline.py');pipe=importlib.util.module_from_spec(sp);sp.loader.exec_module(pipe)
 assert pipe.HOST=='old_pct10_onevision' and pipe.HERE==FRAME
 children=[]
 try:
  for j in JOBS:
   u=ROOT/j['run_dir']
   if j['method']=='OSFT':write(u/'RESUME_REQUEST.json',dict(pause_receipt_sha256=read(u/'run_status.json')['pause_receipt_sha256'],reason='User authorized recovery 2026-10-29'))
   child=pipe.launch(j,'training',j['gpu'],resume=True);children.append((j,child));write(OUT/(j['method']+'_STARTED.json'),dict(identity=ident(child.pid),job=j))
 except BaseException:
  write(OUT/'LAUNCH_FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),started=[dict(method=j['method'],pid=p.pid) for j,p in children],no_auto_kill=True))
 pending=list(children)
 while pending:
  for j,p in list(pending):
   code=p.poll()
   if code is not None:
    write(OUT/(j['method']+'_EXIT.json'),dict(unix=time.time(),returncode=code,identity_absent=ident(p.pid) is None,run_dir=j['run_dir'],phase='training',worker_pid=p.pid));pending.remove((j,p))
  if pending:time.sleep(30)
 write(OUT/'COMPLETE.json',dict(unix=time.time(),all_children_exited=True,not_quality_verification=True))
if __name__=='__main__':main()
