from pathlib import Path
import os,json,hashlib,shutil,time
r=Path('/dev/shm/llmcl-task-gradient-formal-1084-20261028')
u=r/'exp/task_gradient_formal_20261028/runs/onevision/pct01/TaskGradReuse/training_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def guard():
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid==1084:
    c=(p/'cmdline').read_bytes().decode(errors='replace')
    assert str(u) not in c, ('live_run',p.name)
  except (FileNotFoundError,ProcessLookupError,PermissionError):pass
 assert json.loads((r/'summary/task_gradient_formal_20261028/SLOT.json').read_text())['released'] is True
 assert json.loads((u/'checkpoints/latest.json').read_text())['stage']==17
 assert (u/'checkpoints/stage_17.pt').stat().st_size==1928308132
 assert not (u/'checkpoints/stage_19.pt').exists()
guard()
assert sha(u/'checkpoints/stage_17.pt')=='939b7081ad009f387cc142b9baffcbfb6ad1e5e031bc3b1d47463771e7c74ed4'
paths=[u/f'checkpoints/stage_{i:02d}.pt' for i in range(11,16)]
for p in paths:
 assert p.resolve().parent==(u/'checkpoints').resolve() and not p.is_symlink()
 assert p.stat().st_uid==1084 and p.stat().st_nlink==1
 m=json.loads(p.with_suffix('.json').read_text());assert p.stat().st_size==m['bytes']
protected={str(p.relative_to(u)):(p.stat().st_size,p.stat().st_mtime_ns) for p in u.rglob('*') if p.is_file() and p not in paths}
receipt={'unix':time.time(),'free_before':shutil.disk_usage('/dev/shm').free,'latest_sha256':'939b7081ad009f387cc142b9baffcbfb6ad1e5e031bc3b1d47463771e7c74ed4','targets':[{'path':str(p),'bytes':p.stat().st_size,'inode':p.stat().st_ino} for p in paths]}
print(json.dumps(dict(mode='inspect',**receipt)),flush=True)
if os.environ.get('LLMCL_DELETE_EARLY')=='1':
 for p in paths:guard();p.unlink()
 assert protected=={str(p.relative_to(u)):(p.stat().st_size,p.stat().st_mtime_ns) for p in u.rglob('*') if p.is_file()}
 receipt.update(free_after=shutil.disk_usage('/dev/shm').free,removed_bytes=sum(x['bytes'] for x in receipt['targets']),protected_unchanged=True)
 dest=r/'EARLY_S11_S15_RETENTION_20261029.json'
 with dest.open('x') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
 print(json.dumps(dict(mode='completed',**receipt)),flush=True)
