"""Persistent preparation and two-condition queue; GPU0 is never admitted."""
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'summary'/HERE.name
ASSETS=ROOT/'.assets-env/bin/python'
VISION=ROOT/'.vision-env/bin/python'
METHODS=['SeqFT_LoRA','Hyper_LLaVA','PASs_MoE','SpaRTA','OSFT','ALER','PaRSP','FV_guided_Training','SMoLoRA']


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def assets_ready():
    return all((ROOT/'data'/name/file).exists() for name,file in [
        ('LLaVA_Med','source_manifest.json'),('GeoChat_Instruct','full_assets_ready.json'),('AQUA','full_assets_ready.json')]) and json.loads((ROOT/'data/LLaVA_Med/source_manifest.json').read_text()).get('images_complete',False)


def status_report(state):
    lines=['# 全量与 1% 持续学习实验进度','',f"状态：{state['phase']}；更新：{state['updated_at']}",'',
           'GPU0 保留；GPU3 专用于全量组，GPU1/2 用于 1% 训练组。两组均完整评测各自同一套公开有标注测试集。','',
           '| 条件 | 方法 | 状态 | 完成训练阶段及全部测试 | 当前阶段 |','|---|---|---|---:|---:|']
    for key,j in state['jobs'].items():
        lines.append(f"| {j['condition']} | {j['method']} | {j['status']} | {j.get('completed_stage',0)}/10 | {j.get('active_stage','—')} |")
    lines+=['','数据准备及任务数量见 `data/TenTaskCL/ten_task_cl_20261015_full_1pct/preparation_status.json`。',
            '每个方法和每组均强制记录 resources.jsonl、memory_summary.json、训练 loss、裁剪前后梯度 L2、学习率和 token 数；没有完整测试矩阵不填最终指标。']
    (REPORT/'progress.md').write_text('\n'.join(lines)+'\n')


def idle(gpu):
    q=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(int,q.strip().split(','))
    q=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return free>16000 and util<10 and not q


def launch(job,phase,gpu):
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    run=HERE/'runs'/job['condition']/job['method']/f'{phase}_{stamp}'
    run.mkdir(parents=True)
    cmd=[str(VISION),'-u',str(HERE/'run.py'),'--method',job['method'],'--condition',job['condition'],'--run-dir',str(run)]
    if phase=='preflight':cmd+=['--preflight-only']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    log=(run/'supervisor.log').open('a')
    p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    log.close()
    job.update(status=phase,pid=p.pid,gpu=gpu,run_dir=str(run.relative_to(ROOT)),completed_stage=0)
    return p


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(HERE/'.pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=REPORT/'run_status.json'
    if path.exists():state=json.loads(path.read_text())
    else:state={'experiment_id':HERE.name,'phase':'preparing_assets','jobs':{
        c+':'+m:{'condition':c,'method':m,'status':'waiting_data'} for c in ['full','pct01'] for m in METHODS}}
    # Results-only policy: no joining interrupted training stages to a newly initialized model.
    for j in state['jobs'].values():
        if j['status'] in ['training','preflight']:
            pid=j.get('pid');p=Path('/proc')/str(pid)
            alive=p.exists() and str(HERE/'run.py').encode() in p.joinpath('cmdline').read_bytes()
            if not alive:j.update(status='interrupted_no_checkpoint',reason='Fresh run needed; no stage concatenation')
    processes={};preparing=None;more_assets=None
    while True:
        now=time.time();policy=json.loads((HERE/'execution_policy.json').read_text())
        if not assets_ready():
            state['phase']='preparing_assets'
            if more_assets is None or more_assets.poll() is not None:
                if now-state.get('last_more_assets_attempt',0)>120:
                    log=(REPORT/'prepare_more_assets.log').open('a')
                    more_assets=subprocess.Popen([str(ASSETS),'-u',str(HERE/'prepare_more_assets.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    log.close();state['last_more_assets_attempt']=now
        elif not policy.get('training_enabled'):
            state['phase']='preparing_native_full_and_pct01_splits'
            if preparing is None or preparing.poll() is not None:
                if now-state.get('last_prepare_attempt',0)>120:
                    log=(REPORT/'prepare_data.log').open('a')
                    preparing=subprocess.Popen([str(ASSETS),'-u',str(HERE/'prepare_data.py')],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
                    log.close();state['last_prepare_attempt']=now
        else:
            state['phase']='running_two_conditions'
            occupied=set()
            for key,j in state['jobs'].items():
                if j['status'] not in ['training','preflight']:continue
                p=processes.get(key)
                # Adopt an existing same-experiment worker after queue restart.
                active=p.poll() is None if p is not None else Path('/proc',str(j['pid'])).exists()
                if active:
                    occupied.add(j['gpu'])
                    status=ROOT/j['run_dir']/'run_status.json'
                    if status.exists():
                        s=json.loads(status.read_text());j['completed_stage']=s.get('completed_stage',0);j['active_stage']=s.get('active_stage')
                    continue
                run=ROOT/j['run_dir']
                if j['status']=='preflight':
                    proof=run/'gpu_preflight.json'
                    passed=proof.exists() and json.loads(proof.read_text()).get('passed') and (run/'memory_summary.json').exists()
                    if passed:j.update(status='ready_formal',preflight_run=str(run.relative_to(ROOT)))
                    else:j.update(status='failed_preflight',failure_log=str((run/'supervisor.log').relative_to(ROOT)))
                else:
                    s=json.loads((run/'run_status.json').read_text()) if (run/'run_status.json').exists() else {}
                    complete=s.get('status')=='completed' and (run/'memory_summary.json').exists() and (run/'score_matrices.json').exists()
                    j.update(status='completed' if complete else 'failed_training',completed_stage=s.get('completed_stage',0))
                processes.pop(key,None)
            for condition,gpus in [('full',[3]),('pct01',[1,2])]:
                for gpu in gpus:
                    if gpu in occupied or not idle(gpu):continue
                    waiting=[(k,j) for k,j in state['jobs'].items() if j['condition']==condition and j['status'] in ['waiting_data','ready_formal']]
                    if not waiting:continue
                    key,j=waiting[0];phase='training' if j['status']=='ready_formal' else 'preflight'
                    processes[key]=launch(j,phase,gpu);occupied.add(gpu)
            if all(j['status']=='completed' for j in state['jobs'].values()):state['phase']='completed'
        state['updated_at']=datetime.now(timezone(timedelta(hours=8))).isoformat();state['updated_unix']=now
        state['reserved_gpus']=[0];write(path,state);status_report(state)
        if state['phase']=='completed':break
        time.sleep(15)


if __name__=='__main__':main()
