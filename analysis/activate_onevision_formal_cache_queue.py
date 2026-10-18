"""One reviewed CPU queue handoff; preserve all GPU/baseline process identities."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'summary/onevision_diagnostics_20261016/formal_v2_state_cache_20261018/v1'
REPORT=STUDY/'queue'
OLD=ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/flex_queue'
UNIT='llmcl-onevision-causal-priority.service'
NEW='analysis/onevision_formal_cache_queue.py'
SOURCES=[NEW,'analysis/onevision_formal_state_cache.py','analysis/onevision_formal_state_export.py',
    'analysis/onevision_formal_cache_entry.py','analysis/onevision_formal_cache_report.py',
    'analysis/onevision_formal_cache_cpu_pipeline.py','analysis/activate_onevision_formal_cache_queue.py']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def write(p,v):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(v,indent=2)+'\n');tmp.replace(p)
def cmd(*a):return subprocess.check_output(a,text=True).strip()
def process(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    s=(p/'stat').read_text();fields=s[s.rfind(')')+2:].split()
    return dict(startticks=int(fields[19]),UID=p.stat().st_uid,cmdline_sha256=sha(p/'cmdline'))


def prepare():
    REPORT.mkdir(parents=True,exist_ok=True)
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'READY.json')['authorization_sha256']==sha(REPORT/'AUTHORIZATION.json')
        assert all(sha(ROOT/n)==s for n,s in read(REPORT/'AUTHORIZATION.json')['formal_new_sources'].items())
        return
    spec=read(STUDY/'CPU_INPUT_CHECK.json')['spec']
    smoke=read(STUDY/'CPU_PIPELINE_CHECK.json')
    assert smoke['worker_source_sha256']==sha(ROOT/'analysis/onevision_formal_state_cache.py')
    assert smoke['actual_feature_function_synthetic_answer_domain_invariance_and_question_positive_control']
    auth=read(OLD/'AUTHORIZATION.json')
    auth.update(formal_new_sources={n:sha(ROOT/n) for n in SOURCES},
        formal_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        formal_scientific_settings='original shared V2 FP16, all original source files untouched',
        formal_order=['preserve_current_GPU_natural_finish','formal_build_all4','formal_both21454','all_original_jobs_retained'],
        old_inheritance_fields_preserved=True)
    write(REPORT/'AUTHORIZATION.json',auth)
    write(STUDY/'ADMISSION.json',dict(spec=spec,single_authoritative_research_queue=True,
        worker_source_sha256=sha(ROOT/'analysis/onevision_formal_state_cache.py'),
        entry_source_sha256=sha(ROOT/'analysis/onevision_formal_cache_entry.py'),
        origin='user requires fair method comparison; delegated research scheduling',
        no_human_choice_of_individual_samples_or_TEST_scores=True))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='')
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'),str(ROOT/NEW),'--check'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert read(REPORT/'FORMAL_CPU_CHECK.json')['checks']==7
    write(REPORT/'READY.json',dict(CPU_only=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json'),GPU_started=False))
    # Copy only metadata; all adapter/reference payloads remain outside Git.
    (STUDY/'SOURCE_EXPORT_COMPLETE.json').write_bytes((ROOT/'exp/onevision_formal_state_cache_20261018/v1/source_states/COMPLETE.json').read_bytes())


def activate():
    prepare();assert not (REPORT/'ACTIVATION.json').exists(), 'No duplicate CPU owner'
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    old_pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))
    assert old_pid and b'onevision_state_cache_flexible_queue.py' in (Path('/proc')/str(old_pid)/'cmdline').read_bytes()
    state=read(OLD/'queue_state.json');assert state['pid']==old_pid
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
        ident=process(state['worker_pid'])
        assert ident is not None, 'Revalidate completed proof before a different handoff; never blind-retry'
        write(REPORT/'FORMAL_INHERITED_WORKER.json',dict(worker_pid=state['worker_pid'],
            worker_startticks=ident['startticks'],kind=state['kind'],target=state['target']))
    assert all(process(int(p))==v for p,v in protected.items()), 'A protected process changed; inspect before continuing'
    dropin=Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzz-formal-cache.conf'
    dropin.parent.mkdir(parents=True,exist_ok=True);assert not dropin.exists()
    dropin.write_text('[Service]\nExecStart=\nExecStart='+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run(['systemctl','--user','start',UNIT],check=True)
    pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'));assert pid and process(pid)
    assert all(process(int(p))==v for p,v in protected.items())
    write(REPORT/'ACTIVATION.json',dict(CPU_only=True,new_CPU_pid=pid,new_CPU_identity=process(pid),
        retired_CPU_only=old_pid,all_protected_process_identities_preserved=protected,
        no_GPU_interruption=True,all_original_jobs_retained=True,
        authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),
        no_hot_GPU_source_edit=True,authorization_sha256=sha(REPORT/'AUTHORIZATION.json')))
    write(OLD/'queue_state.json',dict(phase='redirect',authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid,old_CPU_intentionally_retired=True,no_GPU_interruption=True))
    print(json.dumps(dict(activated=True,new_CPU_pid=pid,protected_GPU_pids=sorted(gpu_pids))))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--activate',action='store_true');a=p.parse_args()
    if a.activate:activate()
    else:prepare();print(json.dumps(dict(prepared=True,CPU_only=True,GPU_started=False)))
