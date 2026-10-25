"""Copy pct10 records and each run's latest closed checkpoint to NEW storage.

Metadata copies are live snapshots, not complete-result verification. OLD's
checkpoint retention is untouched. Only this collector's older verified slot
may be replaced, after a new checkpoint is copied and its existing SHA checked.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess,time

ROOT=Path('/data1/liyapeng/program/LLMCL')
OLD='/dev/shm/llmcl-pct10-1084-zu7e3hb_'
DEST=ROOT/'exp/two_backbone_pct10_20261025/private_old_archive'
KEY=ROOT/'.deployment/old_pct01_id_ed25519'
SSH=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=10','-i',str(KEY),'-p','8666']
REMOTE='liyapeng_grp@10.254.29.8'
FOLDERS=['exp/two_backbone_pct10_native_20261025','exp/two_backbone_pct10_forever_20261025']

INVENTORY='''from pathlib import Path
import json
r=Path(__OLD__);folders=__FOLDERS__;checkpoints=[];metadata=[]
for folder in folders:
 base=r/folder/'runs'
 if not base.exists():continue
 metadata.append(str(base.relative_to(r)))
 for run in base.glob('*/pct10/*/training_*'):
  pointer=run/'checkpoints'/('committed_latest.json' if 'forever' in folder else 'latest.json')
  try:
   meta=json.loads(pointer.read_bytes());stage=meta['stage']
   record=json.loads((run/'stages'/f'stage_{stage:02d}.json').read_bytes())
   config=json.loads((run/'config.json').read_bytes())
   weight=run/meta['path']
   assert record['complete'] and record['weights_saved'] and record['stage']==stage
   assert config['condition']=='pct10' and config['manifest_sha256']==meta['manifest_sha256']
   assert weight.resolve().is_relative_to(run.resolve()) and weight.stat().st_size==meta['bytes']
   checkpoints.append(dict(run=str(run.relative_to(r)),meta=meta))
  except (OSError,KeyError,ValueError,AssertionError):continue
metadata.append('summary/two_backbone_pct10_20261025')
print(json.dumps(dict(metadata=metadata,checkpoints=checkpoints)))
'''.replace('__OLD__',repr(OLD)).replace('__FOLDERS__',repr(FOLDERS))

def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:f.write(json.dumps(v,indent=2)+'\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p);sync_directory(p.parent)

def sync_directory(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)

def rsync(relative,target,metadata=False):
 assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
 target.parent.mkdir(parents=True,exist_ok=True)
 command=['rsync','-a','--timeout=90','-e',' '.join(SSH)]
 if metadata:command+=['--exclude=*.pt','--exclude=*.partial','--exclude=*.tmp','--exclude=*.lock']
 source=REMOTE+':'+OLD+'/'+relative+('/' if metadata else '')
 if metadata:target.mkdir(parents=True,exist_ok=True)
 return subprocess.run(command+[source,str(target)+('/' if metadata else '')],capture_output=True,text=True,timeout=600)

def cycle():
 started=time.monotonic()
 p=subprocess.run(SSH+[REMOTE,'/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python','-'],input=INVENTORY,capture_output=True,text=True,timeout=60)
 assert p.returncode==0,p.stderr
 inventory=json.loads(p.stdout);copied=[];skipped=[];metadata=[]
 for relative in inventory['metadata']:
  result=rsync(relative,DEST/relative,True)
  metadata.append(dict(path=relative,returncode=result.returncode,stderr=result.stderr[-1000:]))
 for entry in inventory['checkpoints']:
  run=entry['run'];meta=entry['meta'];slot=DEST/run/'durable_latest.json'
  previous=json.loads(slot.read_bytes()) if slot.exists() else None
  if previous and previous['sha256']==meta['sha256']:continue
  if shutil.disk_usage(DEST).free<meta['bytes']+20*1024**3:
   skipped.append(dict(run=run,reason='NEW free-space reserve',bytes=meta['bytes']));continue
  target=DEST/run/meta['path'];temporary=target.with_suffix('.archive_transfer')
  result=rsync(run+'/'+meta['path'],temporary)
  if result.returncode:
   assert temporary.resolve().is_relative_to((DEST/run/'checkpoints').resolve())
   if temporary.exists():temporary.unlink()
   skipped.append(dict(run=run,reason='source may have rotated before transfer',returncode=result.returncode,stderr=result.stderr[-1200:]));continue
  h=hashlib.sha256()
  with temporary.open('rb') as stream:
   for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
   os.fsync(stream.fileno())
  assert temporary.stat().st_size==meta['bytes'] and h.hexdigest()==meta['sha256'],run
  temporary.replace(target);sync_directory(target.parent);write(slot,meta)
  if previous and previous['path']!=meta['path']:
   old=DEST/run/previous['path'];assert old.resolve().is_relative_to((DEST/run/'checkpoints').resolve())
   if old.exists():old.unlink()
  copied.append(dict(run=run,stage=meta['stage'],bytes=meta['bytes'],sha256=meta['sha256']))
 value=dict(unix=time.time(),seconds=time.monotonic()-started,copied=copied,skipped=skipped,
  metadata_snapshots=metadata,free_bytes=shutil.disk_usage(DEST).free,
  old_checkpoint_retention_untouched=True,only_latest_checkpoint_per_run_is_durable=True,
  metadata_is_live_snapshot_not_result_verification=True)
 write(DEST/'COLLECTOR_LATEST.json',value)
 with (DEST/'COLLECTOR_HISTORY.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')

if __name__=='__main__':
 assert os.getuid()==1004
 DEST.mkdir(parents=True,exist_ok=True)
 while True:
  try:cycle()
  except Exception as exc:
   write(DEST/'COLLECTOR_ERROR.json',dict(unix=time.time(),error=repr(exc)))
  time.sleep(120)
