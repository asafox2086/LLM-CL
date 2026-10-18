"""One-time CPU scheduling handoff; preserve the actual GPU worker and jobs."""
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import onevision_causal_feedback_queue as q


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    unit = 'llmcl-onevision-causal-priority.service'
    old_report = q.BASE/'cold_data_queue'
    old_pid = int(command('systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'))
    assert old_pid == 366824 and q.startticks(old_pid) == 14902957
    assert command('systemctl', '--user', 'show', unit, '-p', 'KillMode', '--value') == 'process'
    phase = json.loads((old_report/'queue_state.json').read_text())
    assert phase['phase'] == 'wait' and phase['state']['waiting_for_natural_completion']
    inherited = phase['state']['inherited_worker']
    assert inherited['kind'] == 'functional_full' and inherited['target'] == 0
    assert inherited['worker_pid'] == 358091
    assert q.startticks(358091) == inherited['worker_startticks'] == 14710479
    n = q.checks()
    spec = q.cold.study_plan()
    q.cold.previous.warm_plan()
    q.REPORT.mkdir(parents=True, exist_ok=True)
    assert not (q.REPORT/'ACTIVATION.json').exists(), 'One-time admission, not retry'
    inherited = dict(inherited, retired_CPU_pid=old_pid, unchanged_GPU_source=True)
    q.write(q.REPORT/'INHERITED_WORKER.json', inherited)
    q.install(spec)
    assert q.cold.checks() == 25 and q.cold.previous.checks() == 28
    action, waiting = q.cold.previous.next_action(None, None, None, None, None, None)
    assert action == ('wait', None) and waiting['waiting_for_natural_completion']
    status = q.cold.previous.inherited_status
    assert status(inherited, inherited['worker_startticks'], lambda: False) == 'live'
    assert status(inherited, None, lambda: True) == 'complete'
    try:
        status(inherited, None, lambda: False)
    except AssertionError:
        pass
    else:
        raise AssertionError('Missing completion proof permitted retry')
    q.write(q.REPORT/'INHERITANCE_CPU_CHECK.json', {
        'passed': True, 'CPU_only': True, 'checks': 6, 'actual_live_worker': inherited,
        'GPU_started': False, 'ancestor_policy_checks': [25, 28], 'next_action': action})
    auth = {'delegated_GPU1_plan': True, 'basis': 'Primary goal is causal small-data diagnosis; after current first full naturally completes, advance unchanged warm19 and all18 support controls before second full. No TEST scores used.',
            'wrapper_entrypoint_sha256': q.sha(q.ROOT/'analysis/onevision_causal_feedback_queue.py'),
            'cold_queue_sha256': q.sha(q.ROOT/'analysis/onevision_cold_data_queue.py'),
            'entrypoint_sha256': q.sha(q.ROOT/'analysis/onevision_warm_start_queue.py'),
            'unchanged_previous_queue_sha256': q.sha(q.ROOT/'analysis/onevision_functional_confirmation_queue.py'),
            'warm_plan_sha256': q.sha(q.cold.previous.STUDY/'CPU_PLAN_CHECK.json'),
            'study_plan_sha256': q.sha(q.cold.STUDY/'CPU_PLAN_CHECK.json'),
            'order': q.ORDER, 'all_previous_jobs_retained': True, 'no_TEST_choice': True,
            'GPU_source_edits': False, 'new_CPU_policy_checks': n,
            'worker_CPU_checks': [37, 142], 'ancestor_policy_checks': [25, 28]}
    q.write(q.REPORT/'AUTHORIZATION.json', auth)
    previous = json.loads((old_report/'ACTIVATION.json').read_text())
    ids = {int(p) for p in previous['protected_after']} | {358091, 340149, 363745, 367273, 336398, 336411}
    gpu = command('nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits').splitlines()
    ids.update(int(p.strip()) for p in gpu if p.strip().isdigit())
    protected = {str(p): q.startticks(p) for p in sorted(ids) if q.startticks(p) is not None}
    q.write(q.REPORT/'ACTIVATION_BEFORE.json', {
        'old_CPU_pid': old_pid, 'old_state': phase, 'protected_before': protected,
        'authorization': auth, 'source_sha256': q.sha(Path(__file__))})
    subprocess.run(['systemctl', '--user', 'stop', unit], check=True)
    assert q.startticks(old_pid) is None
    assert all(q.startticks(int(p)) == ticks for p, ticks in protected.items())
    dropin = Path.home()/'.config/systemd/user'/(unit+'.d')/'95-causal-feedback.conf'
    assert not dropin.exists()
    text = ('[Service]\nExecStart=\nExecStart=/usr/bin/nohup '+str(q.ROOT/'.assets-env/bin/python')
            +' -u '+str(q.ROOT/'analysis/onevision_causal_feedback_queue.py')
            +'\nStandardOutput=append:'+str(q.REPORT/'coordinator_nohup.log')
            +'\nKillMode=process\nRestart=no\n')
    dropin.write_text(text)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'start', unit], check=True)
    new_pid = int(command('systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'))
    assert new_pid and new_pid != old_pid
    deadline = time.monotonic()+20
    while not (q.REPORT/'queue_state.json').exists() and time.monotonic() < deadline:
        time.sleep(.5)
    new = json.loads((q.REPORT/'queue_state.json').read_text())
    assert new['pid'] == new_pid and new['phase'] == 'wait' and new['state']['waiting_for_natural_completion']
    after = {p: q.startticks(int(p)) for p in protected}
    assert protected == after
    result = {'UTC': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'old_CPU_pid': old_pid, 'old_startticks': 14902957,
              'new_CPU_pid': new_pid, 'new_startticks': q.startticks(new_pid),
              'protected_before': protected, 'protected_after': after,
              'actual_phase': new, 'no_GPU_source_edits': True, 'no_GPU_process_stopped': True,
              'dropin': text, 'authorization': auth, 'activation_source_sha256': q.sha(Path(__file__))}
    q.write(q.REPORT/'ACTIVATION.json', result)
    q.write(old_report/'queue_state.json', {'phase': 'redirect',
            'authoritative': str((q.REPORT/'queue_state.json').relative_to(q.ROOT)),
            'retired_CPU_pid': old_pid, 'new_CPU_pid': new_pid, 'GPU_worker_preserved': 358091})
    print(json.dumps({'activated': True, 'new_CPU_pid': new_pid,
                      'new_startticks': result['new_startticks'], 'protected': protected,
                      'CPU_policy_checks': n, 'inherited_GPU': 358091, 'phase': new['phase']}))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        q.REPORT.mkdir(parents=True, exist_ok=True)
        q.write(q.REPORT/'ACTIVATION_FAILURE.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
