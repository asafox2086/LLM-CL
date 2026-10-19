"""One-shot nohup formal FOREVER launcher; no blind training retries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/'exp/two_backbone_forever_20261019_v3'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def identity(pid):
    try:
        p=Path(f'/proc/{pid}');s=(p/'stat').read_text()
        return {'pid':pid,'startticks':int(s[s.rfind(')')+2:].split()[19]),
                'cmdline':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:return None


def available(backbone):
    # Existing Qwen and native-BF16 lanes retain priority. GPU3 belongs to
    # those established lanes; this additional method may take free GPU2 only
    # after the original dispatcher has no eligible pending pct01 job.
    candidates=[0] if backbone=='onevision' else [2]
    if backbone=='qwen':
        registry=ROOT/'summary/two_backbone_native_cl_20261017_v2/new3090_run_status.json'
        state=json.loads(registry.read_text())
        jobs=[j for j in state['jobs'].values()
              if j.get('host')=='new3090' and j['condition']=='pct01']
        if any(j['status'] in ['waiting','ready_reload','ready_training','ready_resume'] for j in jobs):
            return None
        if any(j['status'] in ['training','preflight','reload'] and j.get('gpu')==2 for j in jobs):
            return None
    minimum=8500 if backbone=='onevision' else 16000
    raw=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in proc.splitlines() if line.strip()}
    for line in raw.splitlines():
        number,uuid,free,used,util=[x.strip() for x in line.split(',')]
        if (int(number) in candidates and uuid not in busy and int(free)>=minimum
                and int(used)<100 and int(util)==0):return int(number)
    return None


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--backbone',choices=['onevision','qwen'],required=True)
    args=parser.parse_args()
    if args.backbone=='onevision':assert str(ROOT).startswith('/data2/liyapeng_grp/')
    else:assert str(ROOT).startswith('/data1/liyapeng/')
    admission=json.loads((REPORT/'FORMAL_ADMISSION.json').read_text())
    assert admission['both_backbone_GPU_and_reload_passed']
    assert admission['rolling_latest_and_branch_recovery_CPU_passed']
    def sources():
        for name,digest in admission['training_source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    sources()
    folder=REPORT/'formal_queue'/args.backbone
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    # A dead CPU launcher does not authorize replacing its surviving GPU child.
    previous=folder/'state.json'
    assert not previous.exists(),'Inspect existing queue/child identity; never auto-restart'
    write(folder/'OWNER.json',identity(os.getpid()))
    while True:
        gpu=available(args.backbone)
        if gpu is not None:break
        write(folder/'state.json',{'phase':'healthy_wait_for_free_baseline_slot',
                                  'backbone':args.backbone,'worker':None})
        time.sleep(15)
    sources()
    run=FRAME/'runs'/args.backbone/'pct01/FOREVER'/time.strftime('training_%Y%m%d_%H%M%S')
    assert not run.exists()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),
         'LLMCL_HOST':'old2080' if args.backbone=='onevision' else 'new3090',
         'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    command=['nohup',str(ROOT/'.vision-env/bin/python'),'-u',str(FRAME/'run.py'),
             '--method','FOREVER','--condition','pct01','--run-dir',str(run)]
    log=folder/'training_nohup.log';assert not log.exists()
    with log.open('xb') as stream:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                               stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.2)
        write(folder/'state.json',{'phase':'running','backbone':args.backbone,
              'GPU':gpu,'worker':identity(child.pid),'run':str(run),
              'formal_admission_sha256':hashlib.sha256((REPORT/'FORMAL_ADMISSION.json').read_bytes()).hexdigest()})
        code=child.wait()
    if code:
        write(folder/'NEEDS_ATTENTION.json',{'exit_code':code,'run':str(run),
                  'no_auto_retry':True,'latest_committed_stage_can_be_inspected':True})
        raise RuntimeError(f'Formal FOREVER exited {code}; preserve failure evidence')
    status=json.loads((run/'run_status.json').read_text())
    assert status['status']=='completed' and status['completed_stage']==19
    write(folder/'TRAINING_PROCESS_COMPLETE.json',{'run':str(run),
              'requires_independent_19_74_21454_verification_before_publication':True,
              'not_yet_verified_result':True})


if __name__=='__main__':main()
