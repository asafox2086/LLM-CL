"""Additional comparisons: validated adapters wait for idle GPUs and preflight."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from launch import ROOT,HERE,REPORT,TRAIN_PYTHON,env,gpu_inventory

METHODS=['SMoLoRA','PASs_MoE','Hyper_LLaVA','PaRSP','OSFT','SpaRTA','FV_guided_Training','ALER']
READY=['SMoLoRA','SeqLoRA','PASs_MoE']


class AdoptedProcess:
    def __init__(self,pid,run,phase):
        self.pid,self.run,self.phase=pid,run,phase
    def poll(self):
        from pathlib import Path
        process=Path('/proc')/str(self.pid)
        try:
            if process.exists() and (process/'stat').read_text().split()[2]!='Z':
                command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
                if str(self.run) in command:
                    return None
        except (FileNotFoundError,ProcessLookupError):
            pass
        path=self.run/('gpu_preflight.json' if self.phase=='gpu_preflight' else 'run_status.json')
        value=json.loads(path.read_text()) if path.exists() else {}
        return 0 if value.get('passed') or value.get('status')=='completed' else 1


def report(jobs):
    main=json.loads((REPORT/'run_status.json').read_text())
    combined={name:main['jobs'].get(name,{'status':'adapter_pending','completed_stage':0}) for name in METHODS}
    combined.update({name:job for name,job in jobs.items() if name in METHODS})
    state={'updated_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),'seed':42,
        'task_counts':{'train':100,'dev':50,'test':50},'paper_methods':combined,
        'controls':{name:jobs.get(name,{'status':'derived_from_stage_00'}) for name in ['SeqLoRA','Frozen']},
        'historical_qa_replay':False,'supervisor_pid':os.getpid()}
    path=REPORT/'comparison_status.json'
    temp=path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)
    core={'experiment_id':'ten_task_cl_20261012','experiment_budget_id':'100_50_50_seed42',
        'updated_at':state['updated_at'],'phase':'running_or_waiting_gpu',
        'launcher_pid':os.getpid(),'launcher_log':'exp/ten_task_cl_20261012/pipeline_logs/comparison_queue.log',
        'seed':42,'task_counts':state['task_counts'],'save_checkpoints':False,
        'training_started':any(job['status']=='running' for job in jobs.values()),
        'jobs':dict(combined,**state['controls'])}
    core_path=REPORT/'run_status.json'
    core_temp=core_path.with_suffix('.json.tmp')
    core_temp.write_text(json.dumps(core,ensure_ascii=False,indent=2)+'\n')
    core_temp.replace(core_path)
    lines=['# 八种非回放论文方法的对比进度','',state['updated_at'],'',
        '小模型、小数据、轻量化；五领域十任务；每任务 100/50/50；seed 42；不保存 checkpoint。Frozen/SeqLoRA 不计入八种论文方法。','',
        '| 方法 | 状态 | 完成阶段 | GPU |','|---|---|---:|---|']
    for name,job in combined.items():
        lines.append(f'| {name} | {job["status"]} | {job.get("completed_stage",0)} | {job.get("gpu","—")} |')
    lines+=['','adapter_pending 表示适配未完成，不能算成已排队或已启动。',
        'DGG 作者流程含历史样本回放，本轮不参加；论文二维表和源码保留。','']
    (REPORT/'comparison_report.md').write_text('\n'.join(lines))
    (REPORT/'report.md').write_text('\n'.join(lines))
    from collect_report import collect
    collect()


def worker(adopt=False):
    lock=(HERE/'.comparison_queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run_id='small_100_50_50_'+datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    jobs={name:{'status':'waiting_gpu','completed_stage':0,'seed':42} for name in READY}
    if adopt:
        old=json.loads((REPORT/'run_status.json').read_text())
        if old.get('experiment_budget_id')!='100_50_50_seed42':
            raise ValueError('Can only adopt current 100-example budget')
        jobs={name:old['jobs'][name] for name in READY}
    # Store this large comparison's checkpoints on the root partition rather
    # than exhausting the user's 700 GiB /data2 quota. Keep the usual repo path.
    from pathlib import Path
    storage=HERE/'runs'
    storage.mkdir(mode=0o700,exist_ok=True)
    if storage.stat().st_uid!=os.getuid():
        raise ValueError('checkpoint storage is not owned by current user')
    if shutil.disk_usage(storage).free<2*1024**3:
        raise ValueError('Comparison needs 2 GiB free for result files')
    active={}
    if adopt:
        for name,job in jobs.items():
            if job['status'] not in ['running','gpu_preflight']:
                continue
            process=Path('/proc')/str(job['pid'])
            command=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            if process.stat().st_uid!=os.getuid() or 'run.py --method '+name not in command:
                raise ValueError('Unexpected adopted worker identity')
            run=ROOT/job['run_dir']
            phase='gpu_preflight' if job['status']=='gpu_preflight' else 'training'
            handle=(run/'logs/supervisor_adoption.log').open('a')
            active[name]=(AdoptedProcess(job['pid'],run,phase),handle,run,job['gpu'],phase)
    idle_history={}
    while True:
        for name,(process,handle,run,gpu,phase) in list(active.items()):
            progress=run/'run_status.json'
            if progress.is_file():
                jobs[name]['completed_stage']=json.loads(progress.read_text()).get('completed_stage',0)
            code=process.poll()
            if code is None:
                continue
            handle.close()
            if code:
                jobs[name].update(status='failed',exit_code=code)
                active.pop(name)
            elif phase=='gpu_preflight':
                if not json.loads((run/'gpu_preflight.json').read_text()).get('passed'):
                    raise ValueError('GPU preflight has not passed')
                handle=(run/'logs/train.log').open('a')
                child=subprocess.Popen([TRAIN_PYTHON,str(HERE/'run.py'),'--method',name,'--seed','42',
                    '--run-dir',str(run)],cwd=ROOT,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
                jobs[name].update(status='running',pid=child.pid)
                active[name]=(child,handle,run,gpu,'training')
            else:
                result=json.loads(progress.read_text()) if progress.is_file() else {}
                jobs[name]['status']='completed' if result.get('status')=='completed' else 'failed'
                if jobs[name]['status']=='completed':
                    with (run/'logs/figures.log').open('a') as graph_log:
                        graph=subprocess.run(['/tmp/llmcl-assets-env/bin/python',str(HERE/'plot_results.py'),
                            '--run-dir',str(run)],cwd=ROOT,env=env(),stdout=graph_log,stderr=subprocess.STDOUT)
                    jobs[name]['figures_status']='completed' if graph.returncode==0 else 'failed'
                active.pop(name)
        inventory=gpu_inventory()
        for item in inventory:
            gpu=item['index']
            idle=item['used_mib']<1024 and item['utilization']<10 and item['total_mib']-item['used_mib']>=9500
            idle_history[gpu]=idle_history.get(gpu,0)+1 if idle else 0
            name=next((key for key in READY if jobs[key]['status']=='waiting_gpu'),None)
            if name is None or idle_history[gpu]<2 or gpu in {entry[3] for entry in active.values()}:
                continue
            run=storage/name/'seed_42'/run_id
            (run/'logs').mkdir(parents=True,exist_ok=True)
            link=HERE/'runs'/name/'seed_42'/run_id
            link.parent.mkdir(parents=True,exist_ok=True)
            if not link.exists():
                link.symlink_to(run,target_is_directory=True)
            handle=(run/'logs/gpu_preflight.log').open('a')
            child=subprocess.Popen([TRAIN_PYTHON,str(HERE/'run.py'),'--method',name,'--seed','42',
                '--run-dir',str(run),'--preflight-only'],cwd=ROOT,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
            jobs[name].update(status='gpu_preflight',gpu=gpu,pid=child.pid,run_dir=str(link.relative_to(ROOT)),
                physical_run_dir=str(run))
            active[name]=(child,handle,run,gpu,'gpu_preflight')
        report(jobs)
        # Keep reporting other ongoing methods, even after this batch finishes.
        time.sleep(30)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--adopt-current',action='store_true')
    args=parser.parse_args()
    if args.worker:
        worker(args.adopt_current)
    else:
        with (HERE/'.comparison_queue.lock').open('a') as lock:
            try:
                fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('Additional comparison queue already active')
                sys.exit(0)
        with (HERE/'pipeline_logs/comparison_queue.log').open('a') as log:
            child=subprocess.Popen([sys.executable,str(__import__('pathlib').Path(__file__).resolve()),'--worker']+
                (['--adopt-current'] if args.adopt_current else []),
                cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('Additional comparison queue PID:',child.pid)
