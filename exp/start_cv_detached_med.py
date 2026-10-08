"""Start/resume medical-only four-method CL; independent process sessions and reports."""
import argparse
import datetime
import fcntl
import json
import os
import subprocess
from pathlib import Path
from common import ROOT, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',type=Path);args=p.parse_args()
    root=ROOT/'exp/CV_result_med';root.mkdir(parents=True,exist_ok=True)
    lock=(root/'launch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    latest=root/'latest.json'
    if not args.resume and latest.exists():
        previous=json.loads(latest.read_text())
        for job in previous['jobs']:
            try:os.kill(job['pid'],0)
            except ProcessLookupError:continue
            raise RuntimeError(f'Medical supervisor already alive: {job["method"]}, PID {job["pid"]}')
    assert (root/'shared/manifest.json').exists(), 'Run exp/cv_data_med.py first'
    for m in ['seq_lora','migu_lora','olora','sapt_lora']:
        assert json.loads((root/f'validation/recovery/{m}.json').read_text())['passed']
    run=args.resume.resolve() if args.resume else root/'qwen2vl2b_pure_med'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run.mkdir(parents=True,exist_ok=True);jobs=[]
    for gpu,method in enumerate(['seq_lora','migu_lora','olora','sapt_lora']):
        output=run/method;output.mkdir(parents=True,exist_ok=True)
        if args.resume and (output/'queue.json').exists():
            state=json.loads((output/'queue.json').read_text())
            if state['state']=='completed':continue
            try:os.kill(state['pid'],0)
            except ProcessLookupError:pass
            else:raise RuntimeError(f'Supervisor still alive: {state["pid"]}')
        env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='4')
        cmd=[str(ROOT/'.vision-env/bin/python'),'-u',str(ROOT/'exp/med_worker.py'),'--method',method,'--gpu',str(gpu),'--output',str(output)]
        if args.resume:cmd.append('--resume')
        with (output/'supervisor.log').open('ab') as log:
            child=subprocess.Popen(['nohup',*cmd],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        jobs.append({'method':method,'gpu':gpu,'pid':child.pid,'output':str(output),'command':cmd})
    info={'run':str(run),'jobs':jobs,'detached':True,'initialization':'fresh base model'}
    write_json(run/'launch.json',info);write_json(latest,info)
    with (run/'reporter.log').open('ab') as log:
        child=subprocess.Popen(['nohup',str(ROOT/'.plot-env/bin/python'),'-u',str(ROOT/'summary_cv_med/collect_med.py'),'--watch'],
                               cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    write_json(run/'reporter.json',{'pid':child.pid})
    print(json.dumps(info,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
