"""Add original resource logs to unstarted IO recovery branches only.
Default inspect; --execute copies opaque bytes and writes exclusive supplements.
Never change PREPARED/INTENT/IO_RECOVERY_SOURCE, original RUN or any model/queue.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time
ROOT=Path('/data1/liyapeng/program/LLMCL')
REPORT=ROOT/'summary/storage_recovery_20261028'
PREPARED=REPORT/'BRANCH_PREPARATION/PREPARED.json'
PREPARED_SHA='9f356ac3fba318b05c063988571c88ae5ff98d72ae1e8271d35b76e19eec5fe4'
LOADED='29bb1d41c2ff1c8d1de659867eb8a3914f8911dc70bfad3abb12f7c7d9c5050b'
CURRENT='59489ac9a7369f46a0a6c46d3bad92c500e34c5fe543284358a53e43d42a845a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def key(p):
 s=p.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def safe(p):
 assert p.is_absolute() and p.resolve()==p and p.is_relative_to(ROOT) and p.stat().st_uid==1004
 q=p
 while q!=ROOT:assert not q.is_symlink();q=q.parent
 return p
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert os.getuid()==1004 and sha(PREPARED)==PREPARED_SHA
 assert sha(REPORT/'prepare_recovery_loaded_29bb1d41.py')==LOADED and sha(REPORT/'prepare_recovery.py')==CURRENT
 rows=read(PREPARED)['prepared'];assert len(rows)==2
 protected={str(PREPARED):sha(PREPARED)};items=[]
 for row in rows:
  dst=safe(Path(row['branch']));proof=dst/'IO_RECOVERY_SOURCE.json';assert sha(proof)==row['proof_sha256']
  v=read(proof);original=safe(Path(v['original_run']));state=dst/'run_status.json'
  assert read(state)['status']=='io_recovery_prepared'
  src=safe(original/'resources.jsonl');assert src.is_file()
  assert b'"training_before"' in src.read_bytes(),'Native ResourceMonitor.finish requires its original before record'
  target=dst/'resources.jsonl';assert not target.exists()
  for p in (proof,state):protected[str(p)]=sha(p)
  items.append(dict(source=str(src),target=str(target),sha256=sha(src),source_stat=key(src),bytes=src.stat().st_size))
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   if proc.stat().st_uid!=1004:continue
   argv=(proc/'cmdline').read_bytes().split(b'\0')
   assert not any(str(Path(row['branch'])).encode() in argv for row in rows),'Branch already started'
  except (FileNotFoundError,ProcessLookupError):continue
 note=dict(schema='IO-recovery-prepare-source-version-note-v1',
  loaded_prepare_source_sha256=LOADED,loaded_source_basis='Operator reported already loaded source in session91353 before subsequent edit; exact old source independently reconstructed and retained.',
  current_on_disk_prepare_source_sha256=CURRENT,
  discrepancy='prepare_recovery.py was edited while original execution already held the old code. sha(__file__) in the running code could report the newer disk file. No existing receipt is rewritten.',
  PREPARED_sha256=PREPARED_SHA,INTENT_sha256=sha(REPORT/'BRANCH_PREPARATION/INTENT.json'),
  original_receipts_untouched=True,resource_log_scope='Historical multiple-process log for native finish compatibility; not a consistent single-process memory growth measurement.',
  new_scores=0,new_models=0,queue_writes=0,GPU_launches=0)
 out=REPORT/'RESOURCE_SUPPLEMENT.json';assert not out.exists()
 if not a.execute:print(json.dumps(dict(inspection_only=True,items=items,source_version_note=note,writes=0)));return
 for item,row in zip(items,rows):
  src=Path(item['source']);dst=Path(item['target']);assert key(src)==tuple(item['source_stat']) and sha(src)==item['sha256']
  with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g,8*1024**2);g.flush();os.fsync(g.fileno())
  assert key(src)==tuple(item['source_stat']) and sha(dst)==item['sha256']
  write(dst.parent/'IO_RECOVERY_RESOURCE_SUPPLEMENT.json',dict(item,**note))
 assert all(sha(Path(p))==h for p,h in protected.items()) and sha(REPORT/'BRANCH_PREPARATION/INTENT.json')==note['INTENT_sha256']
 write(REPORT/'SOURCE_VERSION_NOTE.json',note)
 write(out,dict(unix=time.time(),script_sha256=sha(__file__),items=items,source_version_note=note))
 print(json.dumps(dict(supplement=str(out),sha256=sha(out),models=0,GPU_launches=0)))
if __name__=='__main__':main()
