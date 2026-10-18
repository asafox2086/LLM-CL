"""Recover identified pre-model v1 import failure using independent v2 sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'summary/onevision_diagnostics_20261016/function_neutral_init_20261018/v1'
STUDY=OLD.parent/'v2';REPORT=STUDY/'queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_function_neutral_init_queue_v2.py'
SOURCES=[NEW]+['analysis/'+n for n in ['onevision_function_neutral_init_trial_v2.py',
    'onevision_function_neutral_init_entry_v2.py','onevision_function_neutral_init_report_v2.py',
    'onevision_function_neutral_gradients.py','onevision_function_neutral_import_check_v2.py',
    'activate_onevision_function_neutral_queue_v2.py']]


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
    gate=read(STUDY/'DEVICE_IMPORT_GATE.json')
    assert gate['old_bug_positive_control'] and gate['v2_devices_0_and_1_preserved'] and not gate['CUDA_initialized']
    for n,s in gate['source_sha256'].items():assert sha(ROOT/n)==s
    spec=read(STUDY/'CPU_PLAN_CHECK.json')['spec']
    original_spec=read(OLD/'CPU_PLAN_CHECK.json')['spec']
    assert {k:v for k,v in spec.items() if k!='source_sha256'}=={k:v for k,v in original_spec.items() if k!='source_sha256'}
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        assert read(REPORT/'AUTHORIZATION.json')['neutral_initialization_sources']=={n:sha(ROOT/n) for n in SOURCES}
        return
    failure=read(OLD/'needs_attention.json');assert 'CUDA_VISIBLE_DEVICES' in failure['traceback']
    assert not list(OLD.glob('*_attempt.json')) and not list(OLD.glob('*_complete.json'))
    paths=[OLD/'needs_attention.json',OLD/'queue/needs_attention.json',OLD/'cold_compensated_nohup.log']
    write(STUDY/'V1_FAILURE_EVIDENCE.json',dict(
        files={str(p.relative_to(ROOT)):dict(sha256=sha(p),text=p.read_text()) for p in paths},
        zero_model_attempt_markers=True,identified_cause='CPU geometry audit import clears CUDA mask',
        no_model_or_TRAIN_started=True,no_failed_predictions_spliced=True))
    (STUDY/'V1_CPU_MECHANICS_REFERENCE.json').write_bytes((OLD/'CPU_PREFLIGHT.json').read_bytes())
    assert read(OLD/'queue/NEUTRAL_INHERITED_COMPLETE.json')['verified']
    auth=read(ROOT/'summary/onevision_diagnostics_20261016/initialization_source_controls_20261018/v1/queue/AUTHORIZATION.json')
    auth.update(neutral_initialization_sources={n:sha(ROOT/n) for n in SOURCES},
        neutral_protocol_sha256=sha(STUDY/'PROTOCOL.md'),identified_v1_failure_sha256=sha(STUDY/'V1_FAILURE_EVIDENCE.json'),
        device_import_gate_sha256=sha(STUDY/'DEVICE_IMPORT_GATE.json'),
        scientific_settings_and_exact_gates_unchanged=True,old_factorial_case2_normally_completed=True,
        neutral_order=['cold_compensated_exact','within_compensated507','cross_compensated507','all_original_jobs'])
    write(REPORT/'AUTHORIZATION.json',auth)
    write(STUDY/'ADMISSION.json',dict(spec=spec,independent_repair_version=2,
        identified_v1_failure_preserved=True,single_authoritative_research_queue=True,
        all_original_jobs_retained=True,no_TEST_selection=True,original_scientific_gates_unchanged=True))
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'NEUTRAL_QUEUE_CPU_CHECK.json')['checks']==8
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists()
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    assert cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    assert cmd('systemctl','--user','show',UNIT,'-p','ActiveState','--value')=='failed'
    assert 'onevision_function_neutral_init_queue.py' in cmd('systemctl','--user','show',UNIT,'-p','ExecStart','--value')
    protected={};gpu_pids={int(p) for p in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                ident=process(int(p.name))
                if ident:protected[p.name]=ident
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    write(REPORT/'BEFORE.json',dict(previous_service_failed=True,protected_processes=protected))
    dropin=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzzz-neutral-initialization-v2.conf'
    dropin.parent.mkdir(parents=True,exist_ok=True);assert not dropin.exists()
    dropin.write_text('[Service]\nExecStart=\nExecStart='+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'));assert pid and process(pid)
    assert all(process(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=process(pid),
        previous_v1_failed_not_restarted=True,all_protected_process_identities_preserved=protected,
        no_GPU_interruption=True,authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),
        device_import_gate_sha256=sha(STUDY/'DEVICE_IMPORT_GATE.json'),all_original_jobs_retained=True))
    write(OLD/'queue/queue_state.json',dict(phase='redirect',old_failed_version_preserved=True,
        authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),new_CPU_pid=pid))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    if a.activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True)))
