"""Detached preparation and multi-GPU queue for the single-seed experiment."""
import argparse
import csv
import fcntl
import io
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'summary/ten_task_cl_20261012'
ASSET_PYTHON='/tmp/llmcl-assets-env/bin/python'
TRAIN_PYTHON=str(ROOT/'.vision-env/bin/python')
METHODS=['SeqLoRA','SMoLoRA']
PENDING=['DGG','PaRSP','PASs_MoE','Hyper_LLaVA']


def env(gpu=None):
    result=os.environ.copy()
    for key in list(result):
        if key.lower() in {'http_proxy','https_proxy','all_proxy'}:
            result.pop(key)
    result.update({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2',
                   'MKL_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'})
    if gpu is not None:
        result['CUDA_VISIBLE_DEVICES']=str(gpu)
    return result


def write_status(status):
    status['updated_at']=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    REPORT.mkdir(parents=True,exist_ok=True)
    path=REPORT/'run_status.json'
    temporary=path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)
    lines=['# 本轮运行进度','',f'更新时间：{status["updated_at"]}','',f'当前阶段：**{status["phase"]}**。',
           '', 'seed 42；每任务 train/dev/test = 2000/100/200；十任务一轮。','',
           '| 方法 | 状态 | GPU | 已完成训练阶段 |','|---|---|---|---|']
    for name,job in status['jobs'].items():
        lines.append(f'| {name} | {job["status"]} | {job.get("gpu","—")} | {job.get("completed_stage",0)} |')
    lines+=['', '待适配方法：DGG、PaRSP、PASs-MoE、Hyper-LLaVA；不以简化实现冒充作者方法。',
            '',f'总日志：`{status["launcher_log"]}`。',
            '运行日志与阶段检查点：`exp/ten_task_cl_20261012/runs/<method>/seed_42/<run_id>/`。',
            '',f'正式训练已开始：{status.get("training_started",False)}。',
            '仅在 GPU 空闲时分配不同方法，不停止其他用户进程。', '']
    if status.get('error'):
        lines+=['错误：`'+status['error']+'`。','']
    (REPORT/'report.md').write_text('\n'.join(lines))
    # Only read actually recorded output files; no missing-stage interpolation.
    from collect_report import collect
    collect()


def command(python,script,log,args=None):
    with log.open('a') as stream:
        process=subprocess.run([python,str(HERE/script)]+(args or []),cwd=ROOT,env=env(),
                                stdout=stream,stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f'{script} exited {process.returncode}; see {log.relative_to(ROOT)}')


def gpu_inventory():
    result=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'],text=True)
    return [{'index':int(row[0]),'total_mib':int(row[1]),'used_mib':int(row[2]),'utilization':int(row[3])}
            for row in csv.reader(io.StringIO(result)) if row]


def worker(run_id):
    lock=(HERE/'.launcher.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    (HERE/'launcher.pid').write_text(str(os.getpid())+'\n')
    log_dir=HERE/'pipeline_logs'/run_id
    log_dir.mkdir(parents=True,exist_ok=True)
    jobs={name:{'status':'waiting_preparation','seed':42,'completed_stage':0} for name in METHODS}
    jobs.update({name:{'status':'adapter_pending','seed':42,'completed_stage':0} for name in PENDING})
    status={'experiment_id':'ten_task_cl_20261012','run_id':run_id,'phase':'preparing',
            'launcher_pid':os.getpid(),'launcher_log':str((log_dir/'launcher.log').relative_to(ROOT)),
            'seed':42,'task_counts':{'train':2000,'dev':100,'test':200},'training_started':False,'jobs':jobs}
    write_status(status)
    try:
        selection=ROOT/'data/TenTaskCL/final_task_manifest.json'
        if not selection.is_file():
            command(ASSET_PYTHON,'prepare_data.py',log_dir/'prepare_data.log')
        manifest=json.loads(selection.read_text())
        if manifest['counts_per_task']!={'train':2000,'dev':100,'test':200}:
            raise ValueError('unexpected sample budget; explicit manifest regeneration needed')
        command(TRAIN_PYTHON,'verify_adapters.py',log_dir/'adapter_cpu_test.log')
        if not manifest.get('formal_ready'):
            status['phase']='preparing_selected_images_and_encoder'
            write_status(status)
            for attempt in range(3):
                try:
                    command(ASSET_PYTHON,'prepare_assets.py',log_dir/'prepare_assets.log')
                    break
                except RuntimeError:
                    if attempt==2:
                        raise
                    time.sleep(30)
        for name in METHODS:
            jobs[name]['status']='waiting_gpu'
        status['phase']='waiting_for_idle_gpus'
        write_status(status)
        active={}
        idle_history={}
        terminal={'completed','failed'}
        while any(jobs[name]['status'] not in terminal for name in METHODS):
            for name,(process,handle,run,gpu,phase) in list(active.items()):
                code=process.poll()
                stage_status=run/'run_status.json'
                if stage_status.exists():
                    jobs[name]['completed_stage']=json.loads(stage_status.read_text()).get('completed_stage',0)
                if code is None:
                    continue
                handle.close()
                if code:
                    jobs[name].update({'status':'failed','exit_code':code})
                    active.pop(name)
                elif phase=='gpu_preflight':
                    gpu_check=run/'gpu_preflight.json'
                    if not gpu_check.is_file() or not json.loads(gpu_check.read_text()).get('passed'):
                        raise RuntimeError('GPU preflight did not confirm completion')
                    log=run/'logs/train.log'
                    handle=log.open('a')
                    process=subprocess.Popen([TRAIN_PYTHON,str(HERE/'run.py'),'--method',name,'--seed','42',
                                '--run-dir',str(run)],cwd=ROOT,env=env(gpu),stdout=handle,stderr=subprocess.STDOUT)
                    active[name]=(process,handle,run,gpu,'training')
                    jobs[name].update({'status':'running','pid':process.pid})
                    status['training_started']=True
                else:
                    job_status=json.loads(stage_status.read_text()) if stage_status.is_file() else {}
                    jobs[name]['status']='completed' if job_status.get('status')=='completed' else 'failed'
                    active.pop(name)
            inventory=gpu_inventory()
            busy={entry[3] for entry in active.values()}
            for gpu in inventory:
                index=gpu['index']
                idle=gpu['used_mib']<1024 and gpu['utilization']<10 and gpu['total_mib']-gpu['used_mib']>=9500
                idle_history[index]=idle_history.get(index,0)+1 if idle else 0
                if idle_history[index]<2 or index in busy:
                    continue
                waiting=next((name for name in METHODS if jobs[name]['status']=='waiting_gpu'),None)
                if waiting is None:
                    break
                # Two consecutive observations reduce the chance of colliding with transient idle periods.
                run=HERE/'runs'/waiting/'seed_42'/run_id
                (run/'logs').mkdir(parents=True,exist_ok=True)
                handle=(run/'logs/gpu_preflight.log').open('a')
                process=subprocess.Popen([TRAIN_PYTHON,str(HERE/'run.py'),'--method',waiting,'--seed','42',
                            '--run-dir',str(run),'--preflight-only'],cwd=ROOT,env=env(index),
                            stdout=handle,stderr=subprocess.STDOUT)
                active[waiting]=(process,handle,run,index,'gpu_preflight')
                jobs[waiting].update({'status':'gpu_preflight','gpu':index,'pid':process.pid,
                                     'run_dir':str(run.relative_to(ROOT))})
                busy.add(index)
            status['gpu_snapshot']=inventory
            status['phase']='running_or_waiting_gpu' if active else 'waiting_for_idle_gpus'
            write_status(status)
            time.sleep(30)
        status['phase']='available_adapters_finished' if all(jobs[name]['status']=='completed' for name in METHODS) else 'run_failed'
        write_status(status)
    except Exception as error:
        status.update({'phase':'preparation_failed','error':str(error)})
        write_status(status)
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--run-id')
    args=parser.parse_args()
    if args.worker:
        worker(args.run_id)
        return
    # Read-only lock probe makes repeated starts harmless without signalling old PIDs.
    with (HERE/'.launcher.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Existing experiment launcher is active:',(HERE/'launcher.pid').read_text().strip())
            return
    run_id=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    log_dir=HERE/'pipeline_logs'/run_id
    log_dir.mkdir(parents=True,exist_ok=True)
    with (log_dir/'launcher.log').open('a') as log:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--worker','--run-id',run_id],
                cwd=ROOT,env=env(),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps({'launcher_pid':process.pid,'run_id':run_id,'log':str((log_dir/'launcher.log').relative_to(ROOT))}))


if __name__=='__main__':
    main()
