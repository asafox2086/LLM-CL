"""Admit paper-inspired direct trials without interrupting any GPU worker."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/v2'
REPORT=STUDY/'queue'
OLD=ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/v1/queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_directional_coeff_queue_v2.py'
SOURCES=['analysis/onevision_directional_coeff_core.py']+['analysis/onevision_directional_coeff_'+n+'_v2.py' for n in ['trial','entry','cpu','report','queue']]+['analysis/activate_onevision_directional_coeff_queue_v2.py']


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(*args):return subprocess.check_output(args,text=True).strip()


def write(p,value):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def identity(pid):
    try:
        p=Path('/proc')/str(pid);s=(p/'stat').read_text().split(') ',1)[1].split()
        if s[0]=='Z':return None
        return dict(startticks=int(s[19]),UID=p.stat().st_uid,cmdline_sha256=sha(p/'cmdline'))
    except (FileNotFoundError,ProcessLookupError):return None


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    spec=read(STUDY/'CPU_PLAN.json')['spec'];gate=read(STUDY/'CPU_CHECK.json')
    assert gate['checks']==8 and gate['device_imports_preserve_0_and_1'] and gate['all_frozen_ids_preserved']
    assert spec['trainable_parameters']==12288 and spec['train_QA']==1313
    for rel,h in gate['source_sha256'].items():assert sha(ROOT/rel)==h
    if (REPORT/'READY.json').exists():
        auth=read(REPORT/'AUTHORIZATION.json')
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        assert auth['directional_coeff_sources']=={n:sha(ROOT/n) for n in SOURCES}
        return
    auth=read(OLD/'AUTHORIZATION.json')
    auth.update(directional_coeff_sources={n:sha(ROOT/n) for n in SOURCES},
        directional_coeff_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        latest_user_basis='Directly adapt published update-direction ideas for small-data performance; detailed attribution is not a prerequisite; no fixed module count',
        directional_coeff_order=['preserve_current_GPU_natural_finish','both_coefficient_train19','both_coefficient_full21454','all_original_jobs_and_fulls_retained'])
    write(REPORT/'AUTHORIZATION.json',auth)
    write(STUDY/'ADMISSION.json',dict(spec=spec,single_authoritative_queue=True,no_TEST_choice=True,
        both_fulls_required=True,all_previous_jobs_retained=True,no_fixed_module_count=True,
        scope='paper-inspired direct continual adaptation; not claimed original or effective yet'))
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as f:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=f,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'COEFF_QUEUE_CPU_CHECK.json')['checks']==8
    subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/'analysis/onevision_directional_coeff_report_v2.py')],cwd=ROOT,check=True)
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists()
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old_pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert old_pid==0 and cmd('systemctl','--user','show',UNIT,'-p','ActiveState','--value')=='failed'
    failure=read(STUDY/'V1_FAILURE_EVIDENCE.json')
    assert failure['old_worker_absent'] and failure['no_optimizer_updates']
    assert sha(OLD/'needs_attention.json')==failure['old_queue_failure_sha256']
    assert read(ROOT/'summary/onevision_diagnostics_20261016/directional_coeff_20261018/backend_repair/PROBE_COMPLETE.json')['complete']
    gpu_pids={int(x) for x in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines()}
    protected={}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name)==old_pid:continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid==os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                value=identity(int(p.name))
                if value:protected[p.name]=value
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    write(REPORT/'BEFORE.json',dict(old_CPU_pid=old_pid,old_queue=read(OLD/'queue_state.json'),protected_processes=protected))
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')=='0'
    assert all(identity(int(p))==v for p,v in protected.items())
    drop=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzzzzzzzzzz-directional-coeff-v2.conf'
    drop.parent.mkdir(parents=True,exist_ok=True);assert not drop.exists()
    drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert pid and identity(pid) and all(identity(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(new_CPU_pid=pid,new_CPU_identity=identity(pid),retired_CPU_only=old_pid,
        all_protected_process_identities_preserved=protected,no_GPU_interruption=True,
        authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),all_old_jobs_and_fulls_retained=True))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    if a.activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True,GPU_started=False)))
