"""Retry the failed SMoLoRA preflight without repeating completed methods."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from launch import ROOT,HERE,REPORT,TRAIN_PYTHON,env,gpu_inventory,write_status


def worker():
    with (HERE/'.launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        state=json.loads((REPORT/'run_status.json').read_text())
        job=state['jobs']['SMoLoRA']
        if job['status']=='completed':
            return
        if job['status'] not in {'failed','waiting_gpu'}:
            raise ValueError('SMoLoRA is not eligible for this retry')
        run=ROOT/job['run_dir']
        (run/'logs').mkdir(exist_ok=True)
        state['launcher_pid']=os.getpid()
        state['launcher_log']=str((run/'logs/retry_supervisor.log').relative_to(ROOT))
        (HERE/'launcher.pid').write_text(str(os.getpid())+'\n')
        job.update(status='waiting_gpu')
        job.pop('exit_code',None)
        state['phase']='waiting_for_idle_gpus'
        write_status(state)
        idle_history={}
        while True:
            inventory=gpu_inventory()
            ready=[]
            for gpu in inventory:
                index=gpu['index']
                idle=gpu['used_mib']<1024 and gpu['utilization']<10 and gpu['total_mib']-gpu['used_mib']>=9500
                idle_history[index]=idle_history.get(index,0)+1 if idle else 0
                if idle_history[index]>=2:
                    ready.append(index)
            state['gpu_snapshot']=inventory
            write_status(state)
            if ready:
                gpu=ready[0]
                break
            time.sleep(30)
        for phase,log_name,extra in [('gpu_preflight','gpu_preflight_retry.log',['--preflight-only']),
                                     ('running','train.log',[])]:
            with (run/'logs'/log_name).open('a') as log:
                process=subprocess.Popen([TRAIN_PYTHON,str(HERE/'run.py'),'--method','SMoLoRA','--seed','42',
                            '--run-dir',str(run)]+extra,cwd=ROOT,env=env(gpu),stdout=log,stderr=subprocess.STDOUT)
                job.update(status=phase,gpu=gpu,pid=process.pid)
                state['phase']='running_or_waiting_gpu'
                write_status(state)
                while process.poll() is None:
                    progress=run/'run_status.json'
                    if progress.is_file():
                        job['completed_stage']=json.loads(progress.read_text()).get('completed_stage',0)
                    write_status(state)
                    time.sleep(30)
                if process.returncode:
                    job.update(status='failed',exit_code=process.returncode)
                    state['phase']='run_failed'
                    write_status(state)
                    return
            if phase=='gpu_preflight':
                check=json.loads((run/'gpu_preflight.json').read_text())
                if not check.get('passed'):
                    raise ValueError('preflight did not pass')
        progress=json.loads((run/'run_status.json').read_text())
        job.update(status=progress['status'],completed_stage=progress['completed_stage'])
        state['phase']='available_adapters_finished' if job['status']=='completed' else 'run_failed'
        write_status(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker()
    else:
        state=json.loads((REPORT/'run_status.json').read_text())
        run=ROOT/state['jobs']['SMoLoRA']['run_dir']
        with (run/'logs/retry_supervisor.log').open('a') as log:
            process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker'],cwd=ROOT,
                env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('SMoLoRA retry supervisor PID:',process.pid)
