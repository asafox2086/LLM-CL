"""CPU-only queue handoff, preserving every actual GPU process identity."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v1'
REPORT=STUDY/'queue'
OLD=ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_function_neutral_init_queue.py'
SOURCES=[NEW]+['analysis/'+n for n in ['onevision_function_neutral_init_core.py',
    'onevision_function_neutral_init_check.py','onevision_function_neutral_init_trial.py',
    'onevision_function_neutral_init_entry.py','onevision_function_neutral_init_report.py',
    'activate_onevision_function_neutral_init_queue.py']]


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
    preflight=read(STUDY/'CPU_PREFLIGHT.json')
    assert preflight['CPU_only'] and not preflight['GPU_started'] and preflight['synthetic']['checks']==8
    assert all(len(v['layers'])==48 for v in preflight['actual_sealed_sources'].values())
    for n,s in preflight['source_sha256'].items():assert sha(ROOT/n)==s
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    for n,s in spec['source_sha256'].items():assert sha(ROOT/n)==s
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        assert read(REPORT/'AUTHORIZATION.json')['neutral_initialization_sources']=={n:sha(ROOT/n) for n in SOURCES}
        return
    auth=read(OLD/'AUTHORIZATION.json')
    auth.update(neutral_initialization_sources={n:sha(ROOT/n) for n in SOURCES},
        neutral_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        neutral_order=['preserve_live_GPU_natural_finish','finish_all5_original_source_controls',
                      'cold_compensated_exact_gate','within_compensated507','cross_compensated507','all_original_jobs'],
        neutral_diagnostic_not_new_routing_method=True)
    write(REPORT/'AUTHORIZATION.json',auth)
    write(STUDY/'ADMISSION.json',dict(spec=spec,single_authoritative_research_queue=True,
        original_five_natural_completion_first=True,no_TEST_choice=True,
        origin='existing user delegation of evidence-driven lightweight research',
        all_original_jobs_retained=True,no_GPU_worker_started_by_prepare=True))
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'NEUTRAL_QUEUE_CPU_CHECK.json')['checks']==8
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists(),'No duplicate owner'
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old_pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert old_pid and b'onevision_initialization_source_queue.py' in (Path('/proc')/str(old_pid)/'cmdline').read_bytes()
    state=read(OLD/'queue_state.json');assert state['pid']==old_pid
    if state.get('phase')=='running':
        assert process(state['worker_pid']),'Wait for old coordinator to seal an exited worker'
    protected={}
    gpu_pids={int(p) for p in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==old_pid:continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                ident=process(int(p.name))
                if ident:protected[p.name]=ident
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    write(REPORT/'BEFORE.json',dict(old_CPU_pid=old_pid,old_queue=state,protected_processes=protected,
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))==0
    state=read(OLD/'queue_state.json')
    if state.get('phase')=='running':
        ident=process(state['worker_pid']) or protected.get(str(state['worker_pid']))
        assert ident,'Preserve exit race; inspect proof before any different handoff'
        write(REPORT/'NEUTRAL_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],
            worker_startticks=ident['startticks'],kind=state['kind'],target=state['target']))
    assert all(process(int(p))==v for p,v in protected.items()),'Protected process changed; inspect'
    dropin=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzz-neutral-initialization.conf'
    dropin.parent.mkdir(parents=True,exist_ok=True);assert not dropin.exists()
    dropin.write_text('[Service]\nExecStart=\nExecStart='+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'));assert pid and process(pid)
    assert all(process(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=process(pid),retired_CPU_only=old_pid,
        all_protected_process_identities_preserved=protected,no_GPU_interruption=True,
        authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),
        all_original_jobs_retained=True,formal_full_deferred_not_deleted=True,
        authorization_sha256=sha(REPORT/'AUTHORIZATION.json')))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    if a.activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True,GPU_started=False)))
