"""Fixed OLD OSFT early-payload retention. Default inspect; --execute unlinks only S01..04.
Run under OLD UID1084. No model, QA, queue, signal or scientific source access.
Receipts are exclusive on persistent data2, never on the full source tmpfs.
INTENT and each completed unlink are also flushed to stdout for the caller.
"""
from pathlib import Path
import argparse,fcntl,hashlib,json,os,shutil,stat,time,traceback
RUN=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_/exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/OSFT/training_20261027_142732')
CP=RUN/'checkpoints'
LATEST_SHA='c137198504159ad1b7b147cdbc5be3e655156a6f2a31b83ea246f8d2021a8018'
LATEST_BYTES=3030859789
TARGETS=tuple(CP/f'stage_{i:02d}.pt' for i in range(1,5))
PROTECTED=tuple(CP/f'stage_{i:02d}.pt' for i in (0,5,6))
OUT=Path('/data2/liyapeng_grp/program/LLMCL/summary/runtime_recovery_20261029/OSFT_S01_S04_retention_20261029')
RECEIPT=OUT/'RECEIPT.json'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_bytes())
def stamp(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1084 and not p.is_symlink()
 return dict(bytes=s.st_size,device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns)
def emit(v):print(json.dumps(v,ensure_ascii=False),flush=True)
def no_live_run():
 for p in Path('/proc').glob('[0-9]*'):
  try:
   if p.stat().st_uid!=1084:continue
   c=(p/'cmdline').read_bytes().decode(errors='replace')
   assert str(RUN) not in c,('Run still referenced by live process',p.name)
  except (FileNotFoundError,ProcessLookupError):pass
def guard():
 assert os.getuid()==1084 and RUN.resolve()==RUN and CP.resolve()==CP
 assert RUN.stat().st_uid==CP.stat().st_uid==1084
 no_live_run()
 assert not RECEIPT.exists(),'One-shot receipt already exists'
 config=read(RUN/'config.json');assert config['method']=='OSFT' and config['condition']=='pct10' and config['backbone']=='onevision'
 latest=read(CP/'latest.json');m=read(CP/'stage_06.json')
 assert latest==m and m['stage']==6 and m['method']=='OSFT' and m['path']=='checkpoints/stage_06.pt'
 assert m['sha256']==LATEST_SHA and m['bytes']==LATEST_BYTES and m['full_architecture_and_runtime']
 assert stamp(CP/'stage_06.pt')['bytes']==LATEST_BYTES
 for i,p in zip((0,5,6),PROTECTED):
  v=read(p.with_suffix('.json'));assert v['stage']==i and v['method']=='OSFT' and v['path']==f'checkpoints/stage_{i:02d}.pt'
  assert stamp(p)['bytes']==v['bytes'] and v['full_architecture_and_runtime']
def write_proof(name,v):
 with (OUT/name).open('x',encoding='utf-8') as f:
  json.dump(v,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 parent_fd=os.open(OUT,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(parent_fd)
 finally:os.close(parent_fd)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 # Lock the existing directory inode; no lock file/space is needed on full tmpfs.
 directory_fd=os.open(CP,os.O_RDONLY|os.O_DIRECTORY)
 fcntl.flock(directory_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 guard()
 targets=[]
 for i,p in zip(range(1,5),TARGETS):
  assert p.parent.resolve()==CP and p.resolve()==p
  s=stamp(p);m=read(p.with_suffix('.json'))
  assert s['nlink']==1 and s['bytes']==m['bytes'] and m['stage']==i and m['method']=='OSFT'
  assert m['path']==f'checkpoints/stage_{i:02d}.pt' and m['full_architecture_and_runtime']
  targets.append(dict(path=str(p),metadata_sha256=sha(p.with_suffix('.json')),producer_checkpoint_sha256=m['sha256'],**s))
 protected={str(p):stamp(p) for p in PROTECTED}
 metadata={str(p):sha(p) for p in [RUN/'config.json',CP/'latest.json']+[CP/f'stage_{i:02d}.json' for i in range(7)]}
 plan=dict(unix=time.time(),run=str(RUN),targets=targets,protected=protected,metadata_sha256=metadata,
  candidate_bytes=sum(v['bytes'] for v in targets),free_before=shutil.disk_usage('/dev/shm').free,latest_checkpoint_sha256=LATEST_SHA)
 emit(dict(mode='inspect',payload_hashes_read=False,**plan))
 if a.execute:
  removed=[]
  try:
   assert sha(CP/'stage_06.pt')==LATEST_SHA
   assert protected=={str(p):stamp(p) for p in PROTECTED}
   assert OUT.resolve()==OUT
   OUT.mkdir(parents=True,exist_ok=False)
   write_proof('INTENT.json',plan)
   emit(dict(mode='INTENT',**plan))
   for p,s in zip(TARGETS,targets):
    guard()
    assert protected=={str(q):stamp(q) for q in PROTECTED}
    assert metadata=={q:sha(Path(q)) for q in metadata}
    assert stamp(p)=={k:s[k] for k in ('bytes','device','inode','nlink','mtime_ns')}
    p.unlink();os.fsync(directory_fd);removed.append(s);emit(dict(mode='UNLINKED',**s))
   assert sha(CP/'stage_06.pt')==LATEST_SHA
   assert protected=={str(p):stamp(p) for p in PROTECTED}
   assert metadata=={q:sha(Path(q)) for q in metadata}
   assert all(not p.exists() for p in TARGETS)
   result=dict(complete=True,**plan,removed=removed,removed_bytes=sum(v['bytes'] for v in removed),
    free_after=shutil.disk_usage('/dev/shm').free,latest_full_SHA_before_and_after=True,
    all_stage_JSON_config_and_protected_payloads_unchanged=True,new_models=0,new_scores=0)
   write_proof('RECEIPT.json',result);emit(dict(mode='COMPLETE',receipt=str(RECEIPT),receipt_sha256=sha(RECEIPT),removed_bytes=result['removed_bytes']))
  except BaseException:
   failure=dict(complete=False,unix=time.time(),removed=removed,traceback=traceback.format_exc(),no_automatic_retry=True)
   emit(dict(mode='FAILURE',**failure))
   if OUT.exists() and not RECEIPT.exists():write_proof('RECEIPT.json',failure)
   raise
