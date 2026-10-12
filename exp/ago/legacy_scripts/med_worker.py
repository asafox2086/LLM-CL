"""Detached GPU memory admission, preflight and checkpoint-aware OOM restart."""
import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from common import ROOT, write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',required=True);p.add_argument('--gpu',type=int,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    signal.signal(signal.SIGHUP,signal.SIG_IGN)
    python=str(ROOT/'.vision-env/bin/python');a.output.mkdir(parents=True,exist_ok=True)
    def status(state,**info):
        write_json(a.output/'queue.json',{'state':state,'pid':os.getpid(),'gpu':a.gpu,'time':time.time(),**info})
    # Original four-method nine-stage peak allocated memory was <=6177 MiB.
    # Require 7680 MiB currently free, leaving >=1500 MiB beyond that peak.
    retries=0
    while True:
        free=int(subprocess.check_output(['nvidia-smi',f'--id={a.gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
        if free<7680:
            status('waiting_for_memory',free_mib=free,required_mib=7680,retries=retries);time.sleep(30);continue
        marker=ROOT/f'exp/CV_result_med/validation/{a.method}/two_updates/passed.json'
        if not marker.exists():
            status('preflight',free_mib=free)
            with (a.output/'preflight.log').open('ab') as log:
                code=subprocess.call([python,'-u',str(ROOT/'exp/tests/cv_preflight_med.py'),'--method',a.method],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            if code:
                status('failed',phase='preflight',exit_code=code);raise SystemExit(code)
        command=[python,'-u',str(ROOT/'exp/cv_run_med.py'),'--config',str(ROOT/f'exp/configs/{a.method}_qwen2vl_med.json'),'--output',str(a.output)]
        if a.resume or (a.output/'config.json').exists():command.append('--resume')
        with (a.output/'worker.log').open('ab') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
            status('running',worker_pid=child.pid,command=command,retries=retries)
            code=child.wait()
        if code==0:
            status('completed');return
        state=json.loads((a.output/'status.json').read_text()) if (a.output/'status.json').exists() else {}
        if 'OutOfMemory' not in state.get('error','') and 'out of memory' not in state.get('error','').lower():
            status('failed',exit_code=code,error=state.get('error'));raise SystemExit(code)
        retries+=1
        if retries>=5:
            status('failed',error='five GPU OOM failures; inspect worker.log');raise SystemExit(code)
        status('retrying_after_oom',retries=retries);time.sleep(45)

if __name__=='__main__':main()
