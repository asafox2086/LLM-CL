"""Admit a traceable trainable-budget SVD control, preserving every actual GPU process."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'summary/onevision_diagnostics_20261016/osft_budget_20261018/v1'
REPORT = STUDY/'queue'
OLD = ROOT/'summary/onevision_diagnostics_20261016/feedback_coverage_20261018/v1/queue'
UNIT = 'llmcl-onevision-causal-priority.service'
NEW = 'analysis/onevision_osft_budget_queue.py'
SOURCES = [NEW]+['analysis/'+n for n in ['onevision_osft_budget_core.py',
    'onevision_osft_budget_trial.py', 'onevision_osft_budget_entry.py',
    'onevision_osft_budget_report.py', 'onevision_osft_budget_cpu.py',
    'onevision_osft_budget_worker_check.py', 'activate_onevision_osft_budget_queue.py']]


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2)+'\n')
    tmp.replace(p)


def cmd(*args):
    return subprocess.check_output(args, text=True).strip()


def identity(pid):
    p = Path('/proc')/str(pid)
    try:
        s = (p/'stat').read_text()
        return dict(startticks=int(s[s.rfind(')')+2:].split()[19]), UID=p.stat().st_uid,
                    cmdline_sha256=sha(p/'cmdline'))
    except (FileNotFoundError, ProcessLookupError):
        return None


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    spec = read(STUDY/'CPU_PLAN.json')['spec']
    gate = read(STUDY/'CPU_WORKER_CHECK.json')
    assert gate['checks'] == 5 and gate['actual_worker_import_preserves_GPU_env'] == ['0', '1']
    assert gate['initial_and_final_frozen_ids_exact'] and not gate['CUDA_initialized']
    for name, expected in gate['source_sha256'].items():
        assert sha(ROOT/name) == expected
    if (REPORT/'READY.json').exists():
        assert read(REPORT/'READY.json')['authorization_sha256'] == sha(REPORT/'AUTHORIZATION.json')
        assert read(REPORT/'AUTHORIZATION.json')['osft_budget_sources'] == {n: sha(ROOT/n) for n in SOURCES}
        return
    auth = read(OLD/'AUTHORIZATION.json')
    auth.update(osft_budget_sources={n: sha(ROOT/n) for n in SOURCES},
        osft_budget_protocol_sha256=sha(STUDY/'PROTOCOL.md'),
        osft_budget_order=['preserve_current_GPU_natural_finish', 'both_budget_SVD_train19',
            'remaining_original18_controls', 'both_budget_SVD_full21454', 'all_original_jobs_and_fulls_retained'])
    write(REPORT/'AUTHORIZATION.json', auth)
    write(STUDY/'ADMISSION.json', dict(spec=spec, single_authoritative_research_queue=True,
        no_TEST_choice=True, all_original_jobs_retained=True, both_new_fulls_required=True,
        origin='User asks to learn from OSFT and permits direct current-model updates; independent matched-parameter mechanism controls, no TEST tuning'))
    with (REPORT/'CPU_CHECK_STDOUT.log').open('w') as log:
        subprocess.run([str(ROOT/'.assets-env/bin/python'), str(ROOT/NEW), '--check'], cwd=ROOT,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=log, stderr=subprocess.STDOUT, check=True)
    assert read(REPORT/'SVD_QUEUE_CPU_CHECK.json')['checks'] == 10
    subprocess.run([str(ROOT/'.assets-env/bin/python'), str(ROOT/'analysis/onevision_osft_budget_report.py')],
                   cwd=ROOT, check=True)
    write(REPORT/'READY.json', dict(CPU_only=True, authorization_sha256=sha(REPORT/'AUTHORIZATION.json'), GPU_started=False))


def activate():
    prepare()
    assert not (REPORT/'ACTIVATION.json').exists()
    assert cmd('systemctl', '--user', 'show', UNIT, '-p', 'KillMode', '--value') == 'process'
    old_pid = int(cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value'))
    assert old_pid and b'onevision_feedback_coverage_queue.py' in (Path('/proc')/str(old_pid)/'cmdline').read_bytes()
    state = read(OLD/'queue_state.json')
    assert state['pid'] == old_pid
    gpu_pids = {int(p) for p in cmd('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits').splitlines()}
    protected = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == old_pid:
            continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid == os.getuid() and b'LLMCL' in (p/'cmdline').read_bytes()):
                item = identity(int(p.name))
                if item:
                    protected[p.name] = item
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    write(REPORT/'BEFORE.json', dict(old_CPU_pid=old_pid, old_queue=state, protected_processes=protected))
    subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True)
    assert cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value') == '0'
    state = read(OLD/'queue_state.json')
    if state.get('phase') == 'running':
        item = identity(state['worker_pid']) or protected.get(str(state['worker_pid']))
        if identity(state['worker_pid']):
            protected[str(state['worker_pid'])] = item
        write(REPORT/'SVD_INHERITED_WORKER.json', dict(worker_pid=state['worker_pid'],
            worker_startticks=item['startticks'] if item else None, kind=state['kind'], target=state['target'],
            terminal_exit_race_requires_original_completion_proof=item is None))
    assert all(identity(int(p)) == item for p, item in protected.items())
    dropin = Path.home()/'.config/systemd/user'/f'{UNIT}.d/99-zzzzzzzzzzz-osft-budget.conf'
    dropin.parent.mkdir(parents=True, exist_ok=True)
    assert not dropin.exists()
    dropin.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(ROOT/'.assets-env/bin/python')+' -u '+str(ROOT/NEW)+'\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'start', UNIT], check=True)
    pid = int(cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value'))
    assert pid and identity(pid)
    assert all(identity(int(p)) == item for p, item in protected.items())
    write(REPORT/'ACTIVATION.json', dict(new_CPU_pid=pid, new_CPU_identity=identity(pid),
        retired_CPU_only=old_pid, all_protected_process_identities_preserved=protected,
        no_GPU_interruption=True, authoritative_queue=str((REPORT/'queue_state.json').relative_to(ROOT)),
        all_original_jobs_and_fulls_retained=True))
    write(OLD/'queue_state.json', dict(phase='redirect', authoritative=str((REPORT/'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid, old_CPU_intentionally_retired=True, no_GPU_interruption=True))
    print(json.dumps(dict(activated=True, new_CPU_pid=pid, protected_GPU_pids=sorted(gpu_pids))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--activate', action='store_true')
    args = p.parse_args()
    if args.activate:
        activate()
    else:
        prepare()
        print(json.dumps(dict(prepared=True, CPU_only=True, GPU_started=False)))
