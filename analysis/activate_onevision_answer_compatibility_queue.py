"""Admit the fixed feedback trial after preserving current neutral controls."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/onevision_diagnostics_20261016/answer_compatibility_20261018/v1'
REPORT=STUDY/'queue'
OLD=ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v2/queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_answer_compatibility_queue.py'
SOURCES=[NEW]+['analysis/'+n for n in ['onevision_answer_compatibility_core.py',
    'onevision_answer_compatibility_trial.py','onevision_answer_compatibility_entry.py',
    'onevision_answer_compatibility_report.py','onevision_answer_compatibility_dispatch_check.py',
    'activate_onevision_answer_compatibility_queue.py']]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def process(pid):
    p=Path('/proc')/str(pid)
    try:
        s=(p/'stat').read_text();fields=s[s.rfind(')')+2:].split()
        return dict(startticks=int(fields[19]),UID=p.stat().st_uid,cmdline_sha256=sha(p/'cmdline'))
    except (FileNotFoundError,ProcessLookupError):return None


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    assert read(STUDY/'CPU_CORE_CHECK.json')['core_sha256']==sha(ROOT/'analysis/onevision_answer_compatibility_core.py')
    assert read(STUDY/'CPU_CORE_CHECK.json')['checks']==8
    gate=read(STUDY/'CPU_WORKER_IMPORT_GATE.json')
    assert gate['worker_import_preserves_GPU_env'] and not gate['CUDA_initialized']
    assert gate['worker_sha256']==sha(ROOT/'analysis/onevision_answer_compatibility_trial.py')
    dispatch=read(STUDY/'CPU_DISPATCH_CHECK.json')
    assert dispatch['actual_old_branch_forwarded'] and dispatch['gold_and_task_field_changes_do_not_change_route']
    for n,s in dispatch['source_sha256'].items():assert sha(ROOT/n)==s
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        assert read(REPORT/'AUTHORIZATION.json')['answer_compatibility_sources']=={n:sha(ROOT/n) for n in SOURCES}
        return
    auth=read(OLD/'AUTHORIZATION.json')
    auth.update(answer_compatibility_sources={n:sha(ROOT/n) for n in SOURCES},
        answer_compatibility_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        answer_compatibility_order=['preserve_current_GPU_natural_finish','finish_all3_v2_neutral',
            'TRAIN_feedback60_and_two_new_diagnostic_views','remaining_old18data_controls',
            'both_new_full21454','all_original_jobs_retained'])
    write(REPORT/'AUTHORIZATION.json',auth)
    write(STUDY/'ADMISSION.json',dict(spec=spec,single_authoritative_research_queue=True,
        all3_neutral_natural_completion_first=True,no_TEST_choice=True,all_original_jobs_retained=True,
        full_confirmation_both_required_deferred_until_old18=True,
        origin='existing user delegation of evidence-driven lightweight learned reuse research'))
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'FEEDBACK_QUEUE_CPU_CHECK.json')['checks']==9
    subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/'analysis/onevision_answer_compatibility_report.py')],cwd=ROOT,check=True)
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists()
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old_pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert old_pid and b'onevision_function_neutral_init_queue_v2.py' in (Path('/proc')/str(old_pid)/'cmdline').read_bytes()
    state=read(OLD/'queue_state.json');assert state['pid']==old_pid
    if state.get('phase')=='running':assert process(state['worker_pid']),'Wait for old coordinator to seal an exited worker'
    protected={};gpu_pids={int(p) for p in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==old_pid:continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                ident=process(int(p.name))
                if ident:protected[p.name]=ident
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    write(REPORT/'BEFORE.json',dict(old_CPU_pid=old_pid,old_queue=state,protected_processes=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    state=read(OLD/'queue_state.json')
    if state.get('phase')=='running':
        ident=process(state['worker_pid']) or protected.get(str(state['worker_pid']))
        assert ident,'Preserve exit race and inspect proof; no blind restart'
        if process(state['worker_pid']):protected[str(state['worker_pid'])]=ident
        write(REPORT/'FEEDBACK_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],
            worker_startticks=ident['startticks'],kind=state['kind'],target=state['target']))
    assert all(process(int(p))==v for p,v in protected.items())
    dropin=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzzzz-answer-compatibility.conf'
    dropin.parent.mkdir(parents=True,exist_ok=True);assert not dropin.exists()
    dropin.write_text('[Service]\nExecStart=\nExecStart='+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'));assert pid and process(pid)
    assert all(process(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=process(pid),retired_CPU_only=old_pid,
        all_protected_process_identities_preserved=protected,no_GPU_interruption=True,
        authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),all_original_jobs_retained=True))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    if a.activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True,GPU_started=False)))
