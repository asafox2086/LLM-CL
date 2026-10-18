"""One-shot CPU queue recovery; preserve every live GPU job and frozen trainer."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME = ROOT/'exp/two_backbone_native_cl_20261017_v2'
STATE = ROOT/'summary/two_backbone_native_cl_20261017_v2/new3090_run_status.json'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/boot_recovery_20261018'
UNIT = 'llmcl-two-backbone-native.service'


def cmd(*args): return subprocess.check_output(args, text=True).strip()
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text())
def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,indent=2)+'\n');temp.replace(p)
def identity(pid):
    p=Path('/proc')/str(pid)
    if not p.exists():return None
    stat=p.joinpath('stat').read_text();fields=stat[stat.rfind(')')+2:].split()
    return dict(startticks=int(fields[19]), UID=p.stat().st_uid,
                cmdline_sha256=hashlib.sha256(p.joinpath('cmdline').read_bytes()).hexdigest())


def audit():
    state=read(STATE);j=state['jobs']['pct01:CL_MoE']
    assert j['status']=='failed_training' and j['completed_stage']==0 and j['dead_at_startup']
    assert j['run_dir'].endswith('CL_MoE/training_20261018_164257')
    assert identity(j['pid']) is None, 'Never duplicate a live method'
    run=ROOT/j['run_dir'];assert not (run/'checkpoints/latest.json').exists()
    assert read(run/'run_status.json').get('status')!='completed'
    assert not list((run/'stages').glob('stage_*.json')), 'No stage may be silently dropped'
    sources={p.name:sha(p) for p in FRAME.glob('*.py')}
    pre=ROOT/j['preflight_run'];a=read(pre/'gpu_preflight.json');b=read(pre/'checkpoint_reload_proof.json')
    assert a['passed'] and b['passed']
    assert a['execution_source_sha256']==b['execution_source_sha256']==sources
    assert cmd('systemctl','--user','show',UNIT,'-p','KillMode','--value')=='process'
    assert int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))>0
    return dict(CPU_only=True, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        previous_job=j, trainer_source_sha256=sources, interrupted_run_status=read(run/'run_status.json'),
        previous_config_sha256=sha(run/'config.json'), previous_supervisor_sha256=sha(run/'supervisor.log'),
        previous_preflight_valid=True, no_durable_stage_or_resume_checkpoint=True,
        action='fresh independent run through original preflight/reload/training gates; no prediction splice',
        cause='process lost across actual host reboot; abrupt termination/corrupt log cause not independently established')


def main(apply):
    assert not (REPORT/'APPLIED.json').exists(), 'One recovery, no blind repetition'
    evidence=audit()
    if not apply:
        print(json.dumps(evidence));return
    REPORT.mkdir(parents=True,exist_ok=True);write(REPORT/'AUDIT.json',evidence)
    pids=set(int(p) for p in cmd('nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits').splitlines())
    protected={str(p):identity(p) for p in pids};assert all(protected.values())
    subprocess.run(['systemctl','--user','stop',UNIT],check=True)
    assert int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value'))==0
    try:
        state=read(STATE);old=state['jobs']['pct01:CL_MoE']
        assert old==evidence['previous_job']
        write(REPORT/'state_before.json',state)
        state['jobs']['pct01:CL_MoE']={
            'condition':'pct01','method':'CL_MoE','host':'new3090','status':'waiting',
            'boot_recovery':{'audit':str((REPORT/'AUDIT.json').relative_to(ROOT)),
                             'previous_job':old,'fresh_run_required':True,
                             'no_old_initial_predictions_or_training_reused':True}}
        write(STATE,state)
        assert {p:identity(int(p)) for p in protected}==protected
        assert {p.name:sha(p) for p in FRAME.glob('*.py')}==evidence['trainer_source_sha256']
    finally:
        subprocess.run(['systemctl','--user','start',UNIT],check=True)
    assert {p:identity(int(p)) for p in protected}==protected
    write(REPORT/'APPLIED.json',dict(applied=True, CPU_only=True,
        queue_pid=int(cmd('systemctl','--user','show',UNIT,'-p','MainPID','--value')),
        audit_sha256=sha(REPORT/'AUDIT.json'), protected_GPU_processes=protected,
        all_GPU_process_identities_preserved=True, trainer_sources_unchanged=True,
        only_CL_MoE_requeued=True, current_SMo_preflight_and_SeqFT_resume_preserved=True,
        no_new_GPU_started_by_recovery=True, no_training_or_score_settings_changed=True))
    print(json.dumps(dict(applied=True, CL_MoE='queued fresh run', live_GPU_jobs='preserved')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');main(p.parse_args().apply)
