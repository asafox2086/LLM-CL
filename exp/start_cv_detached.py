"""Launch/resume four isolated nohup-equivalent sessions without a terminal."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from common import ROOT, write_json

METHODS=['seq_lora','migu_lora','olora','sapt_lora']
p=argparse.ArgumentParser();p.add_argument('--resume',type=Path);args=p.parse_args()
root=ROOT/'exp/CV_result';root.mkdir(parents=True,exist_ok=True)
lock=(root/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
for method in METHODS:
    assert (root/'validation'/method/'two_updates/passed.json').exists(),f'Preflight missing: {method}'
    assert (root/'validation/recovery'/(method+'.json')).exists(),f'Recovery test missing: {method}'
assert (root/'validation/sapt_lora/two_updates/reflection_passed.json').exists()
model=ROOT/'model/Qwen2-VL-2B-Instruct'
for entry in json.loads((model/'download_manifest.json').read_text())['files']:
    h=hashlib.sha256()
    with (model/entry['Path']).open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):h.update(chunk)
    assert h.hexdigest()==entry['Sha256'],entry['Path']
run=args.resume.resolve() if args.resume else root/'qwen2vl2b_language5_medical2_vision2'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
run.mkdir(parents=True,exist_ok=True)
if not args.resume:
    # Refuse accidental concurrent duplicate formal workers.
    active=subprocess.check_output(['pgrep','-af','exp/cv_run.py'],text=True) if subprocess.run(['pgrep','-f','exp/cv_run.py'],stdout=subprocess.DEVNULL).returncode==0 else ''
    if active:raise RuntimeError('A CV worker is already running: '+active)
jobs=[]
for gpu,method in enumerate(METHODS):
    output=run/method;output.mkdir(parents=True,exist_ok=True)
    if args.resume and (output/'status.json').exists():
        status=json.loads((output/'status.json').read_text())
        if status['status']=='completed':continue
        try:
            os.kill(status['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError(f'Worker still alive: {method}, PID {status["pid"]}')
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
    cmd=[str(ROOT/'.vision-env/bin/python'),'-u',str(ROOT/'exp/cv_run.py'),'--config',str(ROOT/f'exp/configs/{method}_qwen2vl.json'),'--output',str(output)]
    if args.resume:cmd.append('--resume')
    with (output/'worker.log').open('ab') as log:
        child=subprocess.Popen(['nohup',*cmd],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    jobs.append({'method':method,'gpu':gpu,'pid':child.pid,'output':str(output),'command':cmd})
write_json(run/'launch.json',{'run':str(run),'jobs':jobs,'detached':True})
write_json(root/'latest.json',{'run':str(run),'jobs':jobs})
print(json.dumps({'run':str(run),'jobs':jobs},ensure_ascii=False,indent=2))
