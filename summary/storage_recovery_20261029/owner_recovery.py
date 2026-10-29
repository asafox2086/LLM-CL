"""Original single capacity owner plus an explicit two-branch IO resume permit.
No new allocator, GPU launch, service start or source mutation here. Original
FullFT v2 idle offer, pct10 queue, reserved GPU0 and other-user guards remain.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,os,sys
ROOT=Path('/data1/liyapeng/program/LLMCL')
FULL=ROOT/'analysis/full_parameter_preflight_capacity_v2_20261028.py'
FULL_SHA='03fe7f0028ab4b8df479bc997aab7011f413eee6e47c4023fed26dd4442c9d56'
REPORT=ROOT/'summary/storage_recovery_20261029'
FRAME=ROOT/'exp/two_backbone_native_checkpoint_20261027'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def validate(path,expected):
 assert path.is_absolute() and sha(path)==expected
 b=read(path);assert b['schema']=='NEW-three-baseline-IO-recovery-20261029'
 assert b['owner_sha256']==sha(__file__) and sha(FULL)==b['fullft_outer_sha256']==FULL_SHA
 assert sha(Path(b['prepared']['path']))==b['prepared']['sha256']
 assert len(b['resume_branches'])==2
 for v in b['resume_branches']:
  run=Path(v['run']);assert run.resolve()==run and run.is_relative_to(FRAME/'runs')
  assert sha(run/'IO_RECOVERY_SOURCE.json')==v['proof_sha256']
  proof=read(run/'IO_RECOVERY_SOURCE.json');assert sha(run/'config.json')==proof['config_sha256']
  assert read(run/'config.json')['execution_source_sha256']==proof['source_sha256']=={p.name:sha(p) for p in FRAME.glob('*.py')}
 return b

def io_resume_authorized(run,branch):
 run=Path(run).resolve();proof=read(run/'IO_RECOVERY_SOURCE.json');status=read(run/'run_status.json')
 assert status['status']=='io_recovery_prepared' and status['completed_stage']==14
 assert proof['CP_SHA_verified'] and proof['cooperative_pause'] is False and proof['checkpoint']['stage']==14
 assert sha(run/'IO_RECOVERY_SOURCE.json')==branch['proof_sha256']
 assert sha(run/'config.json')==proof['config_sha256']
 meta=read(run/'checkpoints/latest.json');assert meta==proof['checkpoint']
 assert read(run/'checkpoints/stage_14.json')==meta and read(run/'stages/stage_14.json')['complete']
 cp=(run/meta['path']).resolve();assert cp.is_relative_to(run/'checkpoints') and cp.stat().st_size==meta['bytes']
 assert not (run/'PAUSE_REQUEST.json').exists() and not (run/'RESUME_REQUEST.json').exists()
 return True # Original checkpoints.load performs the full-file SHA on actual launch.

def main():
 parser=argparse.ArgumentParser(add_help=False)
 parser.add_argument('--recovery-admission',type=Path,required=True)
 parser.add_argument('--recovery-admission-sha256',required=True)
 parser.add_argument('--fullft-admission',type=Path,required=True)
 args,remaining=parser.parse_known_args();b=validate(args.recovery_admission,args.recovery_admission_sha256)
 assert sha(args.fullft_admission)==b['fullft_admission']['sha256'] and args.fullft_admission==Path(b['fullft_admission']['path'])
 full=load('original_fullft_v2_IO_recovery_owner',FULL)
 entry=load('original_checkpoint_capacity_IO_recovery_owner',full.ENTRY)
 slot=full.Slot(args.fullft_admission,'--once' in remaining);factory=entry.addition_class
 branches={str(Path(v['run']).resolve()):v for v in b['resume_branches']}
 def wrapped(*a,**kw):
  base=factory(*a,**kw)
  class Addition(base):
   def patch(self):
    super().patch();original_load=self.cap.load_pipeline
    def load_pipeline():
     pipe=original_load();original_resume=pipe.resume_requested
     def requested(run):
      path=Path(run).resolve();branch=branches.get(str(path))
      if branch and read(path/'run_status.json').get('status')=='io_recovery_prepared':return io_resume_authorized(path,branch)
      return original_resume(run)
     pipe.resume_requested=requested;return pipe
    self.cap.load_pipeline=load_pipeline
   def reconcile(self):super().reconcile();slot.reconcile()
   def offer(self,gpu,minimum,original_idle):
    result=slot.offer(gpu,minimum,original_idle,super().held(),self.pipes['onevision'].PYTHON)
    return super().offer(gpu,minimum,original_idle) if result is None else result
  return Addition
 entry.addition_class=wrapped
 sys.argv=[str(full.ENTRY),*remaining];entry.main()
if __name__=='__main__':main()
