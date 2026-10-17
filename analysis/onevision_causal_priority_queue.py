"""Prepared, approval-gated CPU dispatcher: cap1 -> matched48 -> cap3.

No active worker source changes. The current ordering remains in force until
explicit user preference is recorded and the old CPU dispatchers are retired.
"""
import argparse,fcntl,hashlib,json,os,subprocess,time,traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'summary/onevision_diagnostics_20261016'
REPORT=BASE/'causal_priority_queue'
OLD_OWNERS=['llmcl-onevision-deferred-full.service','llmcl-onevision-full-repair.service','llmcl-onevision-matched-task-domain.service']


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def choose(cap1,matched,cap3,owners,pids,idle):
    if owners or pids or not idle:return ('wait',None)
    if not cap1:return ('full',1)
    for index,done in enumerate(matched):
        if not done:return ('matched',index)
    if not cap3:return ('full',3)
    return ('complete',None)


def plan():
    spec=json.loads((BASE/'matched_task_domain/CPU_PLAN_CHECK.json').read_text())['spec']
    assert len(spec['cases'])==48
    for name,digest in spec['source_sha256'].items():assert sha(ROOT/name)==digest
    lock=json.loads((ROOT/'summary/primary_metric_20261017/metric_lock.json').read_text())
    assert lock['option_mapping'] is False and sha(ROOT/'analysis/answer_f1.py')==lock['metric_sha256']
    assert sha(ROOT/'data/FiveDomainCL/five_domain_native_cl_20261016_sparse/pct01/manifest.json')==spec['manifest_sha256']
    return spec


def complete_full(cap):
    p=BASE/'full_input_repair'/f'cap{cap}_full_confirm_complete.json'
    if not p.exists():return False
    proof=json.loads(p.read_text());assert proof['complete'] and proof['full_test_qa']==21454 and proof['formal_full_test']
    for name,digest in proof['source_sha256'].items():assert sha(ROOT/name)==digest
    trained=json.loads((BASE/'full_input_repair'/f'cap{cap}_complete.json').read_text())
    assert proof['trained_state_sha256']==trained['final_state_sha256']
    run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_capacity_20261016').resolve())
    renderer=json.loads((run/'renderer_protocol.json').read_text())
    assert renderer['renderer_sha256']==sha(ROOT/'analysis/onevision_full_input_repair.py')
    assert json.loads((run/'renderer_gpu_gate.json').read_text())['passed']
    for relative,digest in proof['artifact_sha256'].items():
        path=run/relative;assert path.resolve().is_relative_to(run.resolve()) and sha(path)==digest
    return True


def complete_case(case,spec):
    a,s,m=case;name=f'T{a:02d}_{s}__{m}';path=BASE/'matched_task_domain'/f'{name}_complete.json'
    if not path.exists():return False
    proof=json.loads(path.read_text());assert proof['complete'] and proof['stage1_exact'] and proof['spec']==spec
    run=ROOT/proof['run'];assert run.resolve().is_relative_to((ROOT/'exp/onevision_matched_task_domain_20261017').resolve())
    for relative,digest in proof['artifact_sha256'].items():
        path=run/relative;assert path.resolve().is_relative_to(run.resolve()) and sha(path)==digest
    return True


def snapshot(spec):
    failures=[str(p.relative_to(ROOT)) for folder in [BASE/'full_input_repair',BASE/'matched_task_domain',REPORT]
              for p in folder.glob('*needs_attention.json')]
    assert not failures,failures
    owners={u:int(subprocess.check_output(['systemctl','--user','show',u,'-p','MainPID','--value'],text=True).strip() or 0) for u in OLD_OWNERS}
    pids=subprocess.check_output(['nvidia-smi','--id=1','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip().splitlines()
    used,util=map(int,subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).split(','))
    # A live worker is always respected, even if a completion file already exists.
    if any(owners.values()) or pids or used>=500 or util>=10:
        return ('wait',None),{'old_CPU_owners':owners,'GPU1_PIDs':pids,'used_MiB':used,'utilization':util}
    full1=complete_full(1);cases=[complete_case(c,spec) for c in spec['cases']];full3=complete_full(3)
    return choose(full1,cases,full3,False,False,True),{'cap1_complete':full1,'matched_cases':sum(cases),'cap3_complete':full3}


def check():
    spec=plan();checks=[]
    assert choose(False,[False]*48,False,False,True,True)==('wait',None);checks.append('live GPU blocks launch even if task looks incomplete')
    assert choose(True,[False]*48,False,True,False,True)==('wait',None);checks.append('old CPU dispatcher blocks duplicate ownership')
    assert choose(False,[False]*48,False,False,False,True)==('full',1);checks.append('cap1 resumes first when needed')
    assert choose(True,[False]*48,False,False,False,True)==('matched',0);checks.append('first matched case precedes cap3')
    assert choose(True,[True]*17+[False]*31,False,False,False,True)==('matched',17);checks.append('case order and completed-case skip preserved')
    assert choose(True,[True]*48,False,False,False,True)==('full',3);checks.append('cap3 retained after matched controls')
    assert choose(True,[True]*48,True,False,False,True)==('complete',None);checks.append('requires all full and matched proofs')
    action,state=snapshot(spec)
    write(REPORT/'CPU_PREPARATION.json',{'passed':True,'CPU_only':True,'policy_tests':checks,'current_action':action,
          'current_state':state,'planned_order':['cap1_unchanged','matched48_unchanged','cap3_unchanged'],
          'source_sha256':spec['source_sha256'],'dispatcher_sha256':sha(Path(__file__)),
          'authorization_exists':(REPORT/'AUTHORIZATION.json').exists(),'activated':False,
          'critical_activation_note':'Old services currently may use KillMode=control-group. Must set/verify KillMode=process before stopping only CPU coordinators; preserve actual GPU PID/startticks.'})
    print(json.dumps({'CPU_preparation_passed':True,'tests':len(checks),'current_action':action,'activated':False}))


def run():
    authorization=json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert authorization['approved_by_user'] is True and authorization['order']==['cap1','matched48','cap3']
    assert authorization['dispatcher_sha256']==sha(Path(__file__))
    lock=(REPORT/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    spec=plan()
    while True:
        action,state=snapshot(spec)
        write(REPORT/'queue_state.json',{'phase':action[0],'target':action[1],'pid':os.getpid(),'state':state,'GPU1_only':True})
        if action[0]=='wait':time.sleep(30);continue
        if action[0]=='complete':return
        kind,index=action
        if kind=='full':
            command=[str(ROOT/'analysis/onevision_full_input_repair.py'),'--cap',str(index)]
            logpath=BASE/'full_input_repair'/f'cap{index}_full_confirm_nohup.log'
        else:
            command=[str(ROOT/'analysis/onevision_matched_task_domain.py'),'--case',str(index)]
            a,s,m=spec['cases'][index];logpath=BASE/'matched_task_domain'/f'T{a:02d}_{s}__{m}_nohup.log'
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='1',OMP_NUM_THREADS='6',MKL_NUM_THREADS='6',PYTHONUNBUFFERED='1')
        with logpath.open('a') as out:
            child=subprocess.Popen(['/usr/bin/nohup',str(ROOT/'.vision-env/bin/python'),'-u']+command,cwd=ROOT,env=env,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
            write(REPORT/'queue_state.json',{'phase':'running','kind':kind,'index':index,'pid':os.getpid(),'worker_pid':child.pid,'GPU1_only':True})
            code=child.wait()
        assert code==0,'Worker failed; preserve evidence and stop, no blind retry'
        assert complete_full(index) if kind=='full' else complete_case(spec['cases'][index],spec)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');args=p.parse_args()
    if args.check:check()
    else:
        try:run()
        except Exception:
            write(REPORT/'needs_attention.json',{'traceback':traceback.format_exc(),'pid':os.getpid()});raise
