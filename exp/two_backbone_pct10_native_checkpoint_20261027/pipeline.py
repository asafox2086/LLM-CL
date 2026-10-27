"""Two host queues, disjoint ownership, native-task GPU and checkpoint reload gates."""
import fcntl
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime,timezone,timedelta
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
REPORT=ROOT/'summary'/HERE.name
PYTHON=ROOT/'.vision-env/bin/python'
HOST=os.environ['LLMCL_HOST']
BACKBONE=os.environ['LLMCL_BACKBONE']
OLDROOT='/data2/liyapeng_grp/program/LLMCL'


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def sources():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}


def idle(gpu,minimum):
    s=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    free,util=map(int,s.strip().split(','))
    q=subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    return free>=minimum and util<10 and not q


def alive(job):
    try:
        args=Path('/proc',str(job['pid']),'cmdline').read_bytes().split(b'\0')
        return any(str(HERE/f).encode() in args for f in ['run.py','checkpoint_probe.py'])
    except OSError:return False


def paused_checkpoint(run):
    """Validate small pause metadata only; checkpoints.load verifies the full CP on resume."""
    status_path=run/'run_status.json'
    if not status_path.exists():return None
    status=json.loads(status_path.read_text())
    if status.get('status')!='paused_checkpoint':return None
    receipt_path=(run/status['pause_receipt']).resolve()
    assert receipt_path.is_relative_to((run/'pause_history').resolve())
    assert hashlib.sha256(receipt_path.read_bytes()).hexdigest()==status['pause_receipt_sha256']
    receipt=json.loads(receipt_path.read_text())
    latest=run/'checkpoints/latest.json';checkpoint=json.loads(latest.read_text())
    assert receipt['checkpoint']==status['checkpoint']==checkpoint
    assert receipt['completed_stage']==status['completed_stage']==checkpoint['stage']
    assert receipt['full_stage_complete'] and checkpoint['full_architecture_and_runtime']
    assert hashlib.sha256(latest.read_bytes()).hexdigest()==receipt['checkpoint_metadata_sha256']
    assert hashlib.sha256((run/'config.json').read_bytes()).hexdigest()==receipt['config_sha256']
    assert receipt['execution_source_sha256']==sources(), 'Paused run needs its original source snapshot'
    checkpoint_path=(run/checkpoint['path']).resolve()
    assert checkpoint_path.is_relative_to((run/'checkpoints').resolve())
    assert checkpoint_path.stat().st_size==checkpoint['bytes']
    return status


def resume_requested(run):
    status=paused_checkpoint(run)
    request=run/'RESUME_REQUEST.json'
    if status is None or not request.exists():return False
    value=json.loads(request.read_text())
    assert value['pause_receipt_sha256']==status['pause_receipt_sha256'], 'Stale resume request'
    return True


def launch(job,phase,gpu,resume=False):
    if phase=='reload' or resume: run=ROOT/job['run_dir']
    else:
        run=HERE/'runs'/BACKBONE/job['condition']/job['method']/(phase+'_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
        run.mkdir(parents=True)
    if phase=='reload':cmd=[str(PYTHON),'-u',str(HERE/'checkpoint_probe.py'),'--run-dir',str(run)]
    else:
        cmd=[str(PYTHON),'-u',str(HERE/'run.py'),'--method',job['method'],'--condition',job['condition'],'--run-dir',str(run)]
        if phase=='preflight':cmd+=['--preflight-only']
        if resume:cmd+=['--resume']
    env=dict(os.environ,LLMCL_HOST=HOST,CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='8',MKL_NUM_THREADS='8')
    with (run/'supervisor.log').open('a') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    job.update(status=phase,pid=p.pid,gpu=gpu,host=HOST,run_dir=str(run.relative_to(ROOT)),completed_stage=0 if not resume else job.get('completed_stage',0))
    return p


def remote_snapshot():
    ssh=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=5',
         '-i',str(ROOT/'.deployment/old_pct01_id_ed25519'),'-p','8666','liyapeng_grp@10.254.29.8']
    command=f'cat {OLDROOT}/summary/{HERE.name}/old2080_run_status.json'
    p=subprocess.run(ssh+[command],capture_output=True,text=True,timeout=12)
    if p.returncode: return None
    return json.loads(p.stdout)


def report(state):
    rows=['# 五领域原生任务持续学习','',f"更新时间：{state['updated_at']}",'',
          '原生19阶段；初始全测，中间仅当前/下一任务，最终全测。ForgetRate不计算。',
          '同一公共框架：新机Qwen2-VL，旧机OneVision；新GPU0保留、GPU1研究。','',
          '| 条件 | 方法 | 主机 | 状态 | 已完成阶段 |','|---|---|---|---|---:|']
    for j in state['jobs'].values():rows.append(f"| {j['condition']} | {j['method']} | {j.get('host',HOST)} | {j['status']} | {j.get('completed_stage',0)}/19 |")
    (REPORT/'progress.md').write_text('\n'.join(rows)+'\n')


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(HERE/f'.pipeline_{HOST}.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    policy=json.loads((HERE/'execution_policy.json').read_text());assert policy['training_enabled']
    own=set(policy['ownership'][HOST])
    if HOST=='new3090':own.update('full:'+m for m in policy['methods'])
    path=REPORT/f'{HOST}_run_status.json'
    state=json.loads(path.read_text()) if path.exists() else {'experiment_id':HERE.name,'host':HOST,'jobs':{
        k:{'condition':k.split(':')[0],'method':k.split(':')[1],'host':HOST,'status':'waiting'} for k in sorted(own,key=lambda k:(k.split(':')[0],policy['methods'].index(k.split(':')[1])))}}
    for j in state['jobs'].values():
        if j['status'] in ['training','preflight','reload'] and not alive(j):
            # Resolve completed artifacts below before deciding to resume/restart.
            j['dead_at_startup']=True
    processes={};last_remote=0;remote=None;collectors={}
    while True:
        occupied=set();now=time.time()
        for key,j in state['jobs'].items():
            if j.get('host')!=HOST or j['status'] not in ['training','preflight','reload']:continue
            proc=processes.get(key)
            active=proc.poll() is None if proc else alive(j)
            if active:
                occupied.add(j['gpu']);p=ROOT/j['run_dir']/'run_status.json'
                if p.exists():j.update({k:v for k,v in json.loads(p.read_text()).items() if k in ['completed_stage','active_stage']})
                continue
            run=ROOT/j['run_dir'];log=(run/'supervisor.log').read_text(errors='replace')[-7000:]
            if j['status']=='preflight':
                p=run/'gpu_preflight.json';proof=json.loads(p.read_text()) if p.exists() else {}
                ok=(proof.get('passed') and proof.get('task_ids')==list(range(1,policy['task_count']+1))
                    and proof.get('execution_source_sha256')==sources()
                    and proof.get('manifest_sha256')==policy['conditions'][j['condition']]['manifest_sha256'])
                j['status']='ready_reload' if ok else 'failed_preflight'
            elif j['status']=='reload':
                p=run/'checkpoint_reload_proof.json';proof=json.loads(p.read_text()) if p.exists() else {}
                ok=proof.get('passed') and proof.get('execution_source_sha256')==sources()
                j['status']='ready_training' if ok else 'failed_checkpoint_reload'
                if ok:j['preflight_run']=j['run_dir']
            else:
                p=run/'run_status.json';s=json.loads(p.read_text()) if p.exists() else {}
                if s.get('status')=='completed':j.update(status='completed',completed_stage=policy['task_count'])
                elif s.get('status')=='paused_checkpoint':
                    assert proc is None or proc.returncode==0, 'Pause publication followed by a worker failure'
                    pause=paused_checkpoint(run)
                    j.update(status='paused_checkpoint',completed_stage=pause['completed_stage'],
                             pause_receipt=pause['pause_receipt'],pause_receipt_sha256=pause['pause_receipt_sha256'],
                             pause_exit_code=proc.returncode if proc is not None else None)
                else:j['status']='failed_training'
            if j['status'].startswith('failed'):
                j['failure_log']=str((run/'supervisor.log').relative_to(ROOT))
                j['oom']='out of memory' in log.lower();j['error_tail']=log[-1500:]
            processes.pop(key,None)
        # Explicit authorization only. A cooperative pause and a training fault are
        # distinct states; neither a missing process nor a checkpoint triggers retry.
        for j in state['jobs'].values():
            if j.get('host')==HOST and j['status']=='paused_checkpoint' and resume_requested(ROOT/j['run_dir']):
                j['status']='ready_resume'
        if False and HOST=='new3090' and now-last_remote>45:
            last_remote=now
            try:
                remote=remote_snapshot()
                if remote:
                    write(REPORT/'old2080_snapshot.json',remote)
                    for k,j in remote['jobs'].items():
                        if k not in state['jobs']:state['jobs'][k]=dict(j)
                        elif state['jobs'][k].get('host')=='old2080':state['jobs'][k]=dict(j)
                        # Failed old-card admission may be retried on 24GB, never duplicate a live worker.
                        if j['status'].startswith('failed') and j.get('oom') and state['jobs'][k].get('host')=='old2080':
                            state['jobs'][k]={'condition':j['condition'],'method':j['method'],'host':HOST,'status':'waiting',
                                             'old_attempt':j,'reassigned_reason':'old2080 OOM; no native data/model reduction'}
            except (OSError,ValueError,subprocess.TimeoutExpired) as exc:state['remote_error']=str(exc)
        for condition,gpus in policy['hosts'][HOST]['conditions'].items():
            for gpu in gpus:
                if gpu in occupied or not idle(gpu,policy['hosts'][HOST]['minimum_free_mib']):continue
                eligible=[(k,j) for k,j in state['jobs'].items() if j.get('host')==HOST and j['condition']==condition and j['status'] in ['waiting','ready_reload','ready_training','ready_resume']]
                if not eligible:continue
                key,j=eligible[0];phase={'waiting':'preflight','ready_reload':'reload','ready_training':'training','ready_resume':'training'}[j['status']]
                processes[key]=launch(j,phase,gpu,resume=j['status']=='ready_resume');occupied.add(gpu)
        if HOST=='new3090' and remote:
            for key,j in remote['jobs'].items():
                if j['status']!='completed':continue
                destination=ROOT/j['run_dir'];flag=destination/'collected_from_old.json'
                if flag.exists():continue
                if key in collectors:
                    p=collectors[key]
                    if p.poll() is None:continue
                    if p.returncode==0:
                        meta=json.loads((destination/'checkpoints/latest.json').read_text())
                        h=hashlib.sha256()
                        with (destination/meta['path']).open('rb') as f:
                            for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
                        assert h.hexdigest()==meta['sha256']
                        write(flag,{'source':'old2080','final_checkpoint_sha256':h.hexdigest(),'verified':True})
                    del collectors[key]
                    continue
                destination.mkdir(parents=True,exist_ok=True)
                command=['rsync','-a','--partial','--exclude=*.partial','-e',
                    f'ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -i {ROOT}/.deployment/old_pct01_id_ed25519 -p 8666',
                    f'liyapeng_grp@10.254.29.8:{OLDROOT}/{j["run_dir"]}/',str(destination)+'/']
                with (REPORT/'result_collection.log').open('a') as log:
                    collectors[key]=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        state['updated_at']=datetime.now(timezone(timedelta(hours=8))).isoformat();state['reserved_gpus']=policy['hosts'][HOST]['reserved_gpus']
        state['phase']='running_or_waiting';write(path,state);report(state)
        if HOST=='new3090':write(REPORT/'run_status.json',state)
        time.sleep(15)


if __name__=='__main__':main()
