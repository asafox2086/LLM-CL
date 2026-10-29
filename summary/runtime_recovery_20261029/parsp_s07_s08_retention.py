"""OLD fixed-run retirement of ONLY PaRSP S07/S08 payloads, default inspect.
Run from a saved real file; --execute is explicit. No model, signal, queue,
source, JSON, QA, current/latest/final or S06 anchor deletion. Original failed
or partial receipts are retained; no automatic retry. Active logs may grow.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,time,traceback
ROOT=Path('/dev/shm/llmcl-pct10-1084-zu7e3hb_')
RUN=ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027/runs/onevision/pct10/PaRSP/recovery_io_20261029_S06'
CP=RUN/'checkpoints';OUT=RUN/'checkpoint_cleanup_history/S07_S08_retention_20261029'
STAGES=(7,8);PROTECT=(6,9,10)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def statkey(p):
 s=p.stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,nlink=s.st_nlink)
def regular(p):
 assert p.is_absolute() and p.resolve()==p and p.is_relative_to(RUN)
 q=p
 while q!=ROOT:
  assert not q.is_symlink(),str(q);q=q.parent
 assert p.is_file() and p.stat().st_uid==1084
 return p
def metadata(stage):
 p=regular(CP/f'stage_{stage:02d}.json');m=read(p)
 assert m['stage']==stage and m['method']=='PaRSP' and m['full_architecture_and_runtime']
 assert m['path']==f'checkpoints/stage_{stage:02d}.pt' and m['manifest_sha256']=='02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
 stageproof=regular(RUN/f'stages/stage_{stage:02d}.json');s=read(stageproof)
 assert s['stage']==stage and s['complete'] and s['weights_saved']
 return m
def guard():
 assert os.name=='posix' and os.getuid()==1084 and ROOT.resolve()==ROOT and RUN.resolve()==RUN
 cfg=read(regular(RUN/'config.json'))
 assert cfg['method']=='PaRSP' and cfg['condition']=='pct10' and cfg['backbone']=='onevision'
 assert cfg['checkpoint_all_stages'] and not cfg['preflight_only'] and cfg['save_final_parameters']
 latest=read(regular(CP/'latest.json'));assert latest['stage']>=10
 assert latest['method']=='PaRSP' and latest['path']==f"checkpoints/stage_{latest['stage']:02d}.pt"
 # latest may legitimately advance while this script runs. Never freeze its
 # mtime, active logs, run_status, resources, new stages or control requests.
 kept={}
 for stage in PROTECT:
  m=metadata(stage);p=regular(CP/f'stage_{stage:02d}.pt');assert p.stat().st_size==m['bytes']
  kept[str(stage)]=dict(stat=statkey(p),metadata_sha256=sha(CP/f'stage_{stage:02d}.json'),stageproof_sha256=sha(RUN/f'stages/stage_{stage:02d}.json'))
 for stage in STAGES:metadata(stage)
 return dict(config_sha256=sha(RUN/'config.json'),preserved=kept,
  target_metadata_sha256={str(stage):sha(CP/f'stage_{stage:02d}.json') for stage in STAGES},
  target_stageproof_sha256={str(stage):sha(RUN/f'stages/stage_{stage:02d}.json') for stage in STAGES})
def write(p,v):
 with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--execute',action='store_true');a=ap.parse_args()
 assert not OUT.exists(),'Preserve old intent/failure; no automatic retry'
 before=guard();targets=[]
 for stage in STAGES:
  p=regular(CP/f'stage_{stage:02d}.pt');st=statkey(p);m=metadata(stage)
  assert st['nlink']==1 and st['bytes']==m['bytes']
  assert st['inode'] not in {v['stat']['inode'] for v in before['preserved'].values()}
  targets.append(dict(path=str(p),stage=stage,stat=st,metadata_sha256=sha(p.with_suffix('.json'))))
 assert len({x['stat']['inode'] for x in targets})==2
 s10=regular(CP/'stage_10.pt');expected=metadata(10)['sha256'];assert sha(s10)==expected and guard()==before
 summary=dict(unix=time.time(),script_sha256=sha(__file__),run=str(RUN),targets=targets,
  candidate_bytes=sum(x['stat']['bytes'] for x in targets),protected=before,protected_S10_full_sha256=expected,
  free_before=shutil.disk_usage(ROOT).free,early_CP_backup_not_required_or_claimed=True,
  active_logs_not_compared=True,models=0,signals=0,queue_writes=0,new_scores=0)
 if not a.execute:print(json.dumps(dict(inspection_only=True,**summary)));return
 OUT.mkdir(parents=True,exist_ok=False);removed=[]
 try:
  write(OUT/'INTENT.json',summary);syncdir(OUT)
  for x in targets:
   assert guard()==before
   p=regular(Path(x['path']));assert p.parent==CP and p.name==f"stage_{x['stage']:02d}.pt" and x['stage'] in STAGES
   assert statkey(p)==x['stat'] and sha(p.with_suffix('.json'))==x['metadata_sha256']
   p.unlink();removed.append(x);syncdir(CP)
   write(OUT/('UNLINK_%02d.json'%x['stage']),dict(unix=time.time(),removed=x))
  assert all(not Path(x['path']).exists() for x in targets)
  assert guard()==before and sha(s10)==expected and guard()==before
  result=dict(summary,completed_unix=time.time(),removed=removed,retired_bytes=sum(x['stat']['bytes'] for x in removed),free_after=shutil.disk_usage(ROOT).free,
   S06_S09_S10_and_all_other_payloads_untouched=True,all_QA_scores_JSON_source_controls_untouched=True)
  write(OUT/'RECEIPT.json',result);syncdir(OUT);print(json.dumps(dict(complete=True,receipt=str(OUT/'RECEIPT.json'),removed_bytes=result['retired_bytes'],free_after=result['free_after'])))
 except BaseException:
  write(OUT/'FAILURE.json',dict(unix=time.time(),traceback=traceback.format_exc(),removed=removed,no_automatic_retry=True));syncdir(OUT);raise
if __name__=='__main__':main()
