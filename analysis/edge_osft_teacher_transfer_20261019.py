"""Copy one sealed teacher to the GPU1 host without entering formal run discovery."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'exp/edge_teacher_transfer_20261019/onevision_osft_final'
REPORT=ROOT/'summary/edge_deployment_20261019/teacher_transfer'
REMOTE='/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_cl_20261017_v2/runs/onevision/pct01/OSFT/training_20261018_044103'
EXPECTED='f93481daa528bdca4cd6bdb68ca6ec522698dfaf4703cb277cf5548b19aba541'


def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()


def main():
 REPORT.mkdir(parents=True,exist_ok=True)
 assert DEST.resolve().is_relative_to((ROOT/'exp/edge_teacher_transfer_20261019').resolve())
 assert not (REPORT/'READY.json').exists(),'A verified teacher is immutable; do not recopy'
 DEST.mkdir(parents=True,exist_ok=True);(DEST/'checkpoints').mkdir(exist_ok=True)
 ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-i',str(ROOT/'.deployment/old_pct01_id_ed25519'),'-p','8666']
 def copy(relative,target):
  subprocess.run(['rsync','-a','--partial','--append-verify','-e',shlex.join(ssh),
                  'liyapeng_grp@10.254.29.8:'+REMOTE+'/'+relative,str(target)],check=True)
 copy('config.json',DEST/'config.json')
 copy('runner_source/',str(DEST/'runner_source')+'/')
 copy('checkpoints/stage_19.json',DEST/'checkpoints/stage_19.json')
 meta=json.loads((DEST/'checkpoints/stage_19.json').read_text());assert meta['sha256']==EXPECTED
 copy('checkpoints/stage_19.pt',DEST/'checkpoints/stage_19.pt')
 checkpoint=DEST/'checkpoints/stage_19.pt'
 assert checkpoint.stat().st_size==meta['bytes'] and sha(checkpoint)==EXPECTED
 config=json.loads((DEST/'config.json').read_text())
 assert config['method']=='OSFT' and config['backbone']=='onevision' and config['condition']=='pct01'
 for rel,expected in config['execution_source_sha256'].items():
  p=DEST/'runner_source'/rel;assert p.is_file() and sha(p)==expected,rel
 ready=dict(complete=True,source_host='old2080',source_run=REMOTE,
    destination=str(DEST.relative_to(ROOT)),checkpoint_sha256=EXPECTED,bytes=meta['bytes'],
    copied_scientific_sources=len(config['execution_source_sha256']),source_sha256=sha(__file__),
    not_a_new_formal_result=True,outside_formal_discovery=True,
    no_GPU_started=True,no_training_or_predictions_transferred=True)
 (REPORT/'READY.json').write_text(json.dumps(ready,indent=2)+'\n')
 print(json.dumps(ready),flush=True)


if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  REPORT.mkdir(parents=True,exist_ok=True)
  (REPORT/'FAILURE.json').write_text(json.dumps(dict(traceback=traceback.format_exc(),no_blind_retry=True),indent=2)+'\n')
  raise
