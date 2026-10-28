from pathlib import Path
import os,json,hashlib,time,shutil,sys
ROOT=Path('/data1/liyapeng/program/LLMCL')
RUN=ROOT/'exp/five_domain_native_cl_20261016_sparse/runs/pct01/SeqFT_LoRA/training_20261016_160137'
OUT=ROOT/'summary/storage_recovery_20261028'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def guard():
 assert json.loads((RUN/'run_status.json').read_bytes())['status']=='completed'
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   if p.stat().st_uid==os.getuid():assert str(RUN).encode() not in (p/'cmdline').read_bytes()
  except (FileNotFoundError,PermissionError):pass
latest=json.loads((RUN/'checkpoints/latest.json').read_bytes());assert latest['stage']==19
final=RUN/latest['path'];assert final.resolve()==final and not final.is_symlink()
assert final.stat().st_size==latest['bytes'] and sha(final)==latest['sha256']
protected={str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.suffix!='.pt'}
finalstat=(final.stat().st_ino,final.stat().st_size,final.stat().st_mtime_ns)
targets=[RUN/'checkpoints'/('stage_%02d.pt'%i) for i in range(9,19)]
items=[]
for p in targets:
 assert p.resolve()==p and p.is_relative_to(ROOT/'exp') and not p.is_symlink()
 st=p.stat();assert st.st_uid==os.getuid() and st.st_nlink==1 and st.st_size==4429795510
 items.append(dict(path=str(p),inode=st.st_ino,size=st.st_size,nlink=st.st_nlink))
guard();print(json.dumps(dict(candidates=items,protected_final_sha=latest['sha256'],free=shutil.disk_usage(ROOT).free)),flush=True)
if '--execute' in sys.argv:
 receipt=OUT/'CLEANUP_REMAINDER.json';assert not receipt.exists()
 with (OUT/'CLEANUP_REMAINDER_STARTED.json').open('x') as f:json.dump(dict(unix=time.time(),items=items,protected=protected,final_sha=latest['sha256']),f)
 before=shutil.disk_usage(ROOT).free
 for p,it in zip(targets,items):
  guard();st=p.stat();assert (st.st_ino,st.st_nlink,st.st_size)==(it['inode'],1,it['size'])
  assert (final.stat().st_ino,final.stat().st_size,final.stat().st_mtime_ns)==finalstat
  p.unlink()
 assert {str(p.relative_to(RUN)):sha(p) for p in RUN.rglob('*') if p.is_file() and p.suffix!='.pt'}==protected
 with receipt.open('x') as f:json.dump(dict(unix=time.time(),deleted=items,free_before=before,free_after=shutil.disk_usage(ROOT).free,final_sha=latest['sha256'],final_stat_unchanged=True,nonweight_files_unchanged=True,early_backups_claimed=False),f,indent=2);f.flush();os.fsync(f.fileno())
 print('CLEANUP_COMPLETE',flush=True)
