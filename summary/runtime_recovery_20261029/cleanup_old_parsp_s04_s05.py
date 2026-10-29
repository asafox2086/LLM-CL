"""Fixed OLD PaRSP early CP cleanup; default inspect, --execute deletes two pt.
Run only after source review. Never touches active recovery branch files,
queues, GPU processes, prediction/score/source/metadata or latest S06 payload.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time
ROOT=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_')
BASE=ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/PaRSP'
OLD=BASE/'training_20261027_113623'
BRANCH=BASE/'recovery_io_20261029_S06'
S6_SHA='8c11f1d6b180bd96b636ff7a59ddb676605593380a221cc29731748649f16439'
S6_BYTES=5807396814
REPORT=ROOT/'summary/runtime_recovery_20261029/PARSP_S04_S05_RETENTION'

def read(p):return json.loads(p.read_bytes())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def key(p):
 s=p.lstat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_uid,s.st_nlink]
def safe(p):
 assert p.is_absolute() and p.resolve()==p and p.is_relative_to(OLD)
 q=p
 while q!=ROOT:
  assert not q.is_symlink(),str(q);q=q.parent
 assert p.is_file() and p.stat().st_uid==1084
 return p

def guard():
 assert os.name=='posix' and os.getuid()==1084 and ROOT.resolve()==ROOT
 for proc in Path('/proc').glob('[0-9]*'):
  try:
   if proc.stat().st_uid!=1084:continue
   args=(proc/'cmdline').read_bytes().split(b'\0')
   assert not any(x==str(OLD).encode() or x.startswith((str(OLD)+'/').encode()) for x in args),('live original RUN reference',proc.name)
  except (FileNotFoundError,ProcessLookupError):pass
 original=read(OLD/'checkpoints/latest.json');anchor=read(OLD/'checkpoints/stage_06.json')
 assert original==anchor and anchor['stage']==6 and anchor['method']=='PaRSP'
 assert anchor['bytes']==S6_BYTES and anchor['sha256']==S6_SHA and anchor['path']=='checkpoints/stage_06.pt'
 assert anchor['full_architecture_and_runtime']
 lineage=read(BRANCH/'IO_RECOVERY_SOURCE.json')
 assert lineage['original_run']==str(OLD) and lineage['branch']==str(BRANCH)
 assert lineage['checkpoint']==anchor and lineage['CP_SHA_verified'] and not lineage['cooperative_pause']
 assert lineage['scientific_settings_changed'] is False and lineage['original_RUN_untouched']
 assert read(BRANCH/'checkpoints/stage_06.json')==anchor
 source=safe(OLD/anchor['path']);target=BRANCH/anchor['path']
 assert target.is_file() and not target.is_symlink()
 a,b=source.stat(),target.stat()
 assert (a.st_dev,a.st_ino,a.st_size)==(b.st_dev,b.st_ino,S6_BYTES) and a.st_nlink==b.st_nlink==2
 return source,target

def inventory(excluded):
 return {str(p.relative_to(OLD)):key(p) for p in OLD.rglob('*') if p.is_file() and p not in excluded}
def digest_map(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,v):
 with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');args=ap.parse_args()
 cp,linked=guard();before_cp=key(cp);assert sha(cp)==S6_SHA and key(cp)==before_cp
 paths=[safe(OLD/f'checkpoints/stage_{s:02d}.pt') for s in (4,5)]
 targets=[]
 for stage,path in zip((4,5),paths):
  meta=read(path.with_suffix('.json'));st=key(path)
  assert st[-1]==1 and meta['stage']==stage and meta['method']=='PaRSP'
  assert meta['path']==str(path.relative_to(OLD)) and st[2]==meta['bytes']
  targets.append(dict(path=str(path),stat=st,metadata_sha256=sha(path.with_suffix('.json'))))
 protected=inventory(paths);protected_sha=digest_map(protected)
 receipt=dict(unix=time.time(),script_sha256=sha(Path(__file__)),original_run=str(OLD),active_branch=str(BRANCH),
  source_S6_full_sha256=S6_SHA,source_S6_bytes=S6_BYTES,source_and_branch_S6_stat=before_cp,
  S6_same_inode_two_links=True,targets=targets,protected_files=len(protected),protected_stat_sha256=protected_sha,
  free_before=shutil.disk_usage(ROOT).free,branch_dynamic_files_never_enumerated_or_modified=True)
 print(json.dumps(dict(mode='inspect',**receipt)),flush=True)
 if not args.execute:return
 REPORT.mkdir(parents=True,exist_ok=False)
 write(REPORT/'INTENT.json',receipt)
 for path,record in zip(paths,targets):
  guard();safe(path);assert key(path)==record['stat'] and path.stat().st_nlink==1
  assert sha(path.with_suffix('.json'))==record['metadata_sha256']
  path.unlink()
 cp2,linked2=guard()
 assert key(cp2)==before_cp and key(linked2)==before_cp
 after=inventory([]);assert protected==after and digest_map(after)==protected_sha
 receipt.update(completed=True,removed_bytes=sum(x['stat'][2] for x in targets),removed_payloads=2,
  protected_stat_unchanged=True,S6_preserved=True,free_after=shutil.disk_usage(ROOT).free,
  no_model_or_GPU_signals=True,no_queue_or_control_changes=True)
 write(REPORT/'COMPLETE.json',receipt);print(json.dumps(dict(mode='completed',**receipt)),flush=True)

if __name__=='__main__':main()
