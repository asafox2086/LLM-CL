"""Wait for the preserved CPU model to finish, then run all fresh v2 views."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'summary/edge_deployment_20261019/cpu_precision/v2'


def write(name,value):
    (REPORT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def identity(pid):
    try:
        fields=(Path('/proc')/str(pid)/'stat').read_text().split(') ',1)[1].split()
        return int(fields[19]) if fields[0]!='Z' else None
    except (FileNotFoundError,ProcessLookupError):return None


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    assert not (REPORT/'RESUME_LAUNCH.json').exists()
    write('RESUME_LAUNCH.json',dict(pid=os.getpid(),startticks=identity(os.getpid()),CPU_only=True))
    previous=json.loads((REPORT/'RETIREMENT.json').read_text())
    pid,ticks=previous['preserved_CPU_worker'],previous['worker_startticks']
    while pid and identity(pid)==ticks:
        write('controller_state.json',dict(phase='waiting_original_CPU_worker_natural_finish',pid=pid,startticks=ticks))
        time.sleep(15)
    v1=ROOT/'summary/edge_deployment_20261019/cpu_precision/v1'
    assert (v1/'fp32_dense_COMPLETE.json').exists(), 'Original CPU model failed; retain evidence, no blind retry'
    with (REPORT/'nohup.log').open('ab') as log:
        child=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u',
            str(ROOT/'analysis/edge_cpu_precision_v2_20261019.py'),'--all'],cwd=ROOT,env=os.environ,
            stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        write('controller_state.json',dict(phase='running_all_v2',worker_pid=child.pid,startticks=identity(child.pid)))
        code=child.wait()
    assert code==0 and (REPORT/'ALL_COMPLETE.json').exists(), 'v2 failure; no blind retry'
    write('controller_state.json',dict(phase='complete',all_views_retained=True))


if __name__=='__main__':
    try:main()
    except Exception:
        write('RESUME_FAILURE.json',dict(traceback=traceback.format_exc(),no_blind_retry=True));raise
