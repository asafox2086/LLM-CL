"""Swap only the research CPU owner; all active GPU processes survive."""
import argparse
import os
from pathlib import Path
import subprocess
import activate_onevision_train_update_priority_v2 as previous

base = previous.base
ROOT = base.ROOT
STUDY = ROOT / 'summary/onevision_diagnostics_20261016/lora_rite_continuous_20261019/v1'
REPORT = STUDY / 'queue'
OLD = previous.REPORT
UNIT = previous.UNIT
ENTRY = 'analysis/onevision_lora_rite_queue_20261019.py'
SOURCES = [ENTRY, 'analysis/activate_onevision_lora_rite_queue_20261019.py',
    'analysis/onevision_lora_rite_trial_20261019.py', 'analysis/onevision_lora_rite_entry_20261019.py',
    'analysis/onevision_lora_rite_cpu_20261019.py', 'analysis/onevision_lora_rite_report_20261019.py',
    'analysis/onevision_lora_optimizer_counterfactual_20261019.py', 'code/LoRA_RITE/source_manifest.json']


def prepare():
    previous.prepare()
    REPORT.mkdir(parents=True, exist_ok=True)
    plan = base.read(STUDY / 'CPU_PLAN.json')
    assert len(plan['checks']) == 7 and plan['GPU_model_NOT_run']
    for rel, expected in plan['spec']['source_sha256'].items():
        assert base.sha(ROOT / rel) == expected
    if (REPORT / 'READY.json').exists():
        auth = base.read(REPORT / 'AUTHORIZATION.json')
        assert auth['lora_rite_sources'] == {n: base.sha(ROOT / n) for n in SOURCES}
        assert base.read(REPORT / 'READY.json')['authorization_sha256'] == base.sha(REPORT / 'AUTHORIZATION.json')
        return
    # Preserve the parent schedule's immutable SHA gate while adding our own
    # independent protocol. This copy does not revise the parent document.
    (REPORT / 'SCHEDULE_AMENDMENT.md').write_bytes((OLD / 'SCHEDULE_AMENDMENT.md').read_bytes())
    auth = base.read(OLD / 'AUTHORIZATION.json')
    auth.update(lora_rite_sources={n: base.sha(ROOT / n) for n in SOURCES},
        lora_rite_CPU_plan_sha256=base.sha(STUDY / 'CPU_PLAN.json'),
        lora_rite_protocol_sha256=base.sha(STUDY / 'PROTOCOL.md'),
        lora_rite_order=['current_GPU_natural_finish', 'existing_TRAIN_only12pilot',
                        'RITE19_TRAIN', 'RITE_full21454', 'all_previous_fulls_and_work'],
        lora_rite_authorization='User delegated GPU1 direct published-method adaptations; no TEST selection; all old obligations retained.')
    base.write(REPORT / 'AUTHORIZATION.json', auth)
    with (REPORT / 'CPU_CHECK_STDOUT.log').open('w') as out:
        subprocess.run([str(ROOT / '.assets-env/bin/python'), str(ROOT / ENTRY), '--check'], cwd=ROOT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), stdout=out, stderr=subprocess.STDOUT, check=True)
    assert base.read(REPORT / 'RITE_QUEUE_CPU_CHECK.json')['checks'] == 8
    base.write(REPORT / 'READY.json', dict(CPU_only=True, authorization_sha256=base.sha(REPORT / 'AUTHORIZATION.json'),
                                         scientific_sources_FROZEN=True, GPU_started=False))


def activate():
    prepare(); assert not (REPORT / 'ACTIVATION.json').exists()
    assert base.cmd('systemctl', '--user', 'show', UNIT, '-p', 'KillMode', '--value') == 'process'
    old = int(base.cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value'))
    assert old and base.identity(old)
    assert b'onevision_train_update_priority_v2.py' in (Path('/proc') / str(old) / 'cmdline').read_bytes()
    gpu_pids = {int(s) for s in base.cmd('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits').splitlines()}
    protected = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) == old:
            continue
        try:
            if int(p.name) in gpu_pids or (p.stat().st_uid == os.getuid() and b'LLMCL' in (p / 'cmdline').read_bytes()):
                item = base.identity(int(p.name))
                if item:
                    protected[p.name] = item
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    base.write(REPORT / 'BEFORE.json', dict(old_CPU_pid=old, old_state=base.read(OLD / 'queue_state.json'), protected_processes=protected))
    subprocess.run(['systemctl', '--user', 'stop', UNIT], check=True)
    assert base.cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value') == '0'
    state = base.read(OLD / 'queue_state.json'); base.write(REPORT / 'OLD_STATE_AFTER_CPU_STOP.json', state)
    inherited = state.get('state', {}).get('inherited_worker')
    if state.get('phase') == 'running':
        item = base.identity(state['worker_pid'])
        if item:
            inherited = dict(worker_pid=state['worker_pid'], worker_startticks=item['startticks'], kind=state['kind'], target=state['target'])
    if inherited:
        item = base.identity(inherited['worker_pid'])
        if item:
            assert item['startticks'] == inherited['worker_startticks']
            base.write(REPORT / 'PILOT_INHERITED_WORKER.json', inherited)
        else:
            import onevision_directional_coeff_report_v2 as reporter
            spec = base.read(reporter.REPORT / 'CPU_PLAN.json')['spec']
            assert inherited['kind'] == 'directional_coeff' and reporter.complete(inherited['target']['phase'], inherited['target']['view'], spec)
    assert all(base.identity(int(p)) == v for p, v in protected.items())
    drop = Path.home() / '.config/systemd/user' / (UNIT + '.d') / ('99-' + 'z' * 18 + '-lora-rite.conf')
    assert not drop.exists()
    drop.write_text('[Service]\nExecStart=\nExecStart=/usr/bin/nohup ' + str(ROOT / '.assets-env/bin/python') + ' -u ' + str(ROOT / ENTRY) + '\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'start', UNIT], check=True)
    pid = int(base.cmd('systemctl', '--user', 'show', UNIT, '-p', 'MainPID', '--value'))
    assert base.identity(pid) and all(base.identity(int(p)) == v for p, v in protected.items())
    base.write(REPORT / 'ACTIVATION.json', dict(new_CPU_pid=pid, new_CPU_identity=base.identity(pid), retired_CPU_only=old,
        protected_processes_preserved=protected, no_GPU_interruption=True, all_old_work_retained=True))
    base.write(OLD / 'queue_state.json', dict(phase='redirect', authoritative=str((REPORT / 'queue_state.json').relative_to(ROOT)),
        new_CPU_pid=pid, old_CPU_intentionally_retired=True, no_GPU_interruption=True))
    print(__import__('json').dumps(dict(activated=True, new_CPU_pid=pid, protected_GPU_pids=sorted(gpu_pids))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--activate', action='store_true')
    if parser.parse_args().activate:
        activate()
    else:
        prepare(); print('{"prepared": true, "CPU_only": true}')
