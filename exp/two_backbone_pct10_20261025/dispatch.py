"""One metadata coordinator for the frozen pct10 native/FOREVER runners.

No training code is implemented here. Each worker uses pipeline.launch unchanged.
The old host owns fourteen jobs; native-BF16 SMoLoRA waits for a new-host handoff.
"""
from pathlib import Path
import argparse,fcntl,hashlib,importlib.util,json,os,shutil,subprocess,sys,time

ROOT=Path(os.environ['LLMCL_ROOT']).resolve()
REPORT=ROOT/'summary/two_backbone_pct10_20261025'
METHODS=('SeqFT_LoRA','PASs_MoE','CL_MoE','OSFT','PaRSP','SpaRTA','FOREVER','SMoLoRA')
FOLDERS={'native':'two_backbone_pct10_native_20261025','forever':'two_backbone_pct10_forever_20261025'}

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8');tmp.replace(p)
def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().rsplit(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(pid=int(pid),startticks=s[19],uid=p.stat().st_uid,pgid=int(s[2]))
    except (OSError,ValueError):return None
def pipe(family,machine,backbone):
    host=machine+'_pct10_'+backbone
    os.environ.update(LLMCL_HOST=host,LLMCL_BACKBONE=backbone)
    path=ROOT/'exp'/FOLDERS[family]/'pipeline.py'
    spec=importlib.util.spec_from_file_location('pct10_'+family+'_'+backbone,path)
    v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    assert v.ROOT==ROOT and v.HOST==host and v.BACKBONE==backbone
    return v

def phase(request_path):
    request=read(request_path);job=request['job'];gpu=request['gpu']
    family='forever' if job['method']=='FOREVER' else 'native'
    p=pipe(family,request['machine'],job['backbone'])
    ready=read(ROOT/'exp'/FOLDERS[family]/'execution_policy.json')
    assert ready['conditions']['pct10']['manifest_sha256']==request['manifest_sha256']
    assert p.sources()==request['sources']
    assert p.idle(gpu,8500 if request['machine']=='old' else 16000)
    process=p.launch(job,request['phase'],gpu)
    child=identity(process.pid);assert child and child['uid']==os.getuid() and child['pgid']==child['pid']
    status=Path(request['status_path'])
    write(status,dict(unix=time.time(),job=job,wrapper=identity(os.getpid()),worker=child,
        phase=request['phase'],run_dir=job['run_dir'],sources=request['sources'],GPU=gpu))
    code=process.wait()
    write(Path(request['exit_path']),dict(unix=time.time(),returncode=code,worker=child,
        wrapper=identity(os.getpid()),phase=request['phase'],run_dir=job['run_dir']))
    if code:raise SystemExit(code)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--machine',choices=['old','new'],default='old')
    ap.add_argument('--phase-request',type=Path);args=ap.parse_args()
    if args.phase_request:phase(args.phase_request);return
    if args.machine=='new':
        assert os.environ.get('LLMCL_PCT01_DISPATCH_HANDOFF')=='reviewed', 'Coordinate with the live pct01 dispatcher before sharing NEW GPUs'
    REPORT.mkdir(parents=True,exist_ok=True)
    lock=(REPORT/('dispatch_'+args.machine+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    location=REPORT/('queue_'+args.machine+'.json')
    state=read(location) if location.exists() else dict(created=time.time(),machine=args.machine,jobs={})
    if not state['jobs']:
        for method in METHODS:
            for backbone in ('onevision','qwen'):
                key=backbone+':'+method
                state['jobs'][key]=dict(key=key,condition='pct10',method=method,backbone=backbone,
                    status='waiting_native_bf16_on_new' if args.machine=='old' and method=='SMoLoRA' else 'waiting')
    processes={};modules={(family,b):pipe(family,args.machine,b) for family in FOLDERS for b in ('onevision','qwen')}
    manifest=read(ROOT/'exp'/FOLDERS['native']/'execution_policy.json')['conditions']['pct10']['manifest_sha256']
    while True:
        active=[]
        for key,job in state['jobs'].items():
            if job['status'] not in ('launching','preflight','reload','training'):continue
            wrapper=identity(job['wrapper']['pid']);status_path=Path(job['phase_status']);exit_path=Path(job['phase_exit'])
            proc=processes.get(key)
            if status_path.exists():
                info=read(status_path);job.update(status=info['phase'],pid=info['worker']['pid'],worker=info['worker'],run_dir=info['run_dir'])
            if wrapper==job['wrapper'] and (proc is None or proc.poll() is None):
                active.append(key);continue
            if not exit_path.exists():
                # A missing wrapper is never a reason to restart a possibly live child.
                job['status']='orphaned_live' if job.get('worker') and identity(job['worker']['pid'])==job['worker'] else 'failed_supervision'
                continue
            ex=read(exit_path);job['last_exit']=ex
            if job.get('worker') and identity(job['worker']['pid'])==job['worker']:
                job['status']='orphaned_live';continue
            run=ROOT/ex['run_dir'];phase_name=ex['phase'];family='forever' if job['method']=='FOREVER' else 'native'
            module=modules[family,job['backbone']]
            if ex['returncode']!=0:
                job['status']='failed_'+phase_name
                log=run/'supervisor.log';job['error_tail']=log.read_text(errors='replace')[-2200:] if log.exists() else ''
                processes.pop(key,None);continue
            if phase_name in ('preflight','reload'):
                filename='gpu_preflight.json' if phase_name=='preflight' else 'checkpoint_reload_proof.json'
                proof=read(run/filename)
                assert proof['passed'] and proof['manifest_sha256']==manifest and proof['execution_source_sha256']==module.sources()
                if phase_name=='preflight':assert proof['task_ids']==list(range(1,20))
                job['status']='ready_reload' if phase_name=='preflight' else 'ready_training'
                if phase_name=='reload':job['preflight_run']=job['run_dir']
            else:
                status=read(run/'run_status.json')
                assert status['status']=='completed' and status['completed_stage']==19
                job.update(status='completed_unverified',completed_stage=19)
            processes.pop(key,None)
        used={state['jobs'][key]['gpu'] for key in active}
        for gpu in ([0,1,2,3] if args.machine=='old' else [1,2,3]):
            if gpu in used:continue
            for key,job in state['jobs'].items():
                if job['status'] not in ('waiting','ready_reload','ready_training'):continue
                family='forever' if job['method']=='FOREVER' else 'native';module=modules[family,job['backbone']]
                if not module.idle(gpu,8500 if args.machine=='old' else 16000):break
                # Preserve room for checkpoints/records; never delete older evidence to admit a job.
                if shutil.disk_usage(ROOT).free<12*1024**3:break
                phase_name={'waiting':'preflight','ready_reload':'reload','ready_training':'training'}[job['status']]
                token=str(time.time_ns());request=REPORT/'requests'/(key.replace(':','_')+'_'+phase_name+'_'+token+'.json')
                status_path=request.with_suffix('.status.json');exit_path=request.with_suffix('.exit.json')
                write(request,dict(job=dict(job),machine=args.machine,gpu=gpu,phase=phase_name,
                    manifest_sha256=manifest,sources=module.sources(),status_path=str(status_path),exit_path=str(exit_path)))
                log_path=request.with_suffix('.wrapper.log')
                with log_path.open('a') as log:
                    process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--phase-request',str(request)],
                        env=dict(os.environ,LLMCL_ROOT=str(ROOT)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                who=identity(process.pid);assert who and who['uid']==os.getuid() and who['pgid']==who['pid']
                job.update(status='launching',wrapper=who,gpu=gpu,phase=phase_name,phase_status=str(status_path),phase_exit=str(exit_path),request=str(request))
                processes[key]=process;used.add(gpu);break
        state.update(updated=time.time(),controller=identity(os.getpid()),manifest_sha256=manifest,
            active=active,reserved_new_GPU=0,all_training_from_pretrained=True,full_test_QA=21454,
            original_pct01_sources_and_jobs_untouched=True)
        write(location,state)
        if all(j['status'] in ('completed_unverified','completed_verified','waiting_native_bf16_on_new') or j['status'].startswith('failed') for j in state['jobs'].values()):return
        time.sleep(15)

if __name__=='__main__':main()
