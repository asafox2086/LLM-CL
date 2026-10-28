from pathlib import Path
import os,json,hashlib,shutil,time,sys
ROOT=Path('/data1/liyapeng/program/LLMCL')
RUN=ROOT/'exp/two_backbone_native_checkpoint_20261027/runs/qwen/pct01/OSFT/recovery_io_20261028_S02'
OUT=ROOT/'summary/storage_recovery_20261028/OSFT_S03_S06_RETENTION.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def stat(p):
 s=p.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
def check_fds(targets):
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=os.getuid():continue
   for fd in (proc/'fd').iterdir():
    try:assert os.readlink(fd) not in targets,'Target open by live process'
    except FileNotFoundError:pass
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert os.getuid()==1004 and RUN.resolve()==RUN
assert read(RUN/'checkpoints/latest.json')['stage']>=8 and not OUT.exists()
protected={}
for k in [2,7,8]:
 p=RUN/f'checkpoints/stage_{k:02d}.pt';protected[str(p)]=stat(p)
metadata={};items=[]
for k in range(3,7):
 p=RUN/f'checkpoints/stage_{k:02d}.pt';m=read(p.with_suffix('.json'));stage=read(RUN/f'stages/stage_{k:02d}.json')
 assert p.resolve()==p and not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_uid==1004
 assert m['stage']==k and m['method']=='OSFT' and m['full_architecture_and_runtime'] and m['bytes']==p.stat().st_size
 assert stage['complete'] and stage['weights_saved'] and stage['stage']==k
 for q in [p.with_suffix('.json'),RUN/f'stages/stage_{k:02d}.json',RUN/f'validity/stage_{k:02d}.json']:metadata[str(q)]=sha(q)
 items.append({'path':str(p),'stat':stat(p),'metadata_sha256':sha(p.with_suffix('.json'))})
check_fds({i['path'] for i in items});print(json.dumps({'targets':items,'free':shutil.disk_usage(ROOT).free}),flush=True)
if '--execute' in sys.argv:
 with OUT.with_name('OSFT_S03_S06_RETENTION_STARTED.json').open('x') as f:json.dump({'items':items,'protected':protected,'metadata':metadata},f);f.flush();os.fsync(f.fileno())
 before=shutil.disk_usage(ROOT).free
 for i in items:
  assert read(RUN/'checkpoints/latest.json')['stage']>=8
  assert all(stat(Path(p))==v for p,v in protected.items())
  check_fds({i['path']});p=Path(i['path']);assert stat(p)==i['stat'] and p.stat().st_nlink==1
  p.unlink()
 assert all(sha(Path(p))==v for p,v in metadata.items())
 assert all(stat(Path(p))==v for p,v in protected.items())
 with OUT.open('x') as f:json.dump({'unix':time.time(),'deleted':items,'bytes_retired':sum(i['stat'][2] for i in items),'free_before':before,'free_after':shutil.disk_usage(ROOT).free,'protected_S02_S07_S08_stat_unchanged':True,'all_metadata_retained':True,'later_and_final_CP_untouched':True,'early_backups_claimed':False,'GPU_signals':0},f,indent=2);f.flush();os.fsync(f.fileno())
 print('COMPLETE',flush=True)
