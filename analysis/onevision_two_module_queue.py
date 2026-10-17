"""CPU-only revised research order: cap1 -> matched48 -> pilot12 -> cap3.

Reuses immutable full/matched workers and proof validators. No source hot edit.
"""
import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import onevision_causal_priority_queue as prior

ROOT = prior.ROOT
BASE = prior.BASE
REPORT = BASE/'two_module_priority_queue'
PILOT = BASE/'two_module_prototype'
ORDER = ['cap1', 'matched48', 'two_module_pilot12', 'cap3']


def plans():
    matched = prior.plan()
    cpu = json.loads((PILOT/'CPU_VALIDATION.json').read_text())
    assert cpu['passed'] and cpu['CPU_only']
    for name, digest in cpu['source_sha256'].items():
        assert prior.sha(ROOT/name) == digest
    pilot = json.loads((PILOT/'CPU_PILOT_PLAN.json').read_text())['spec']
    assert len(pilot['cases']) == 12 and pilot['formal_CL_ranking'] is False
    assert pilot['manifest_sha256'] == matched['manifest_sha256']
    for name, digest in pilot['source_sha256'].items():
        assert prior.sha(ROOT/name) == digest
    return matched, pilot


def pilot_complete(case, spec):
    task, variant = case
    path = PILOT/f'T{task:02d}_{variant}_complete.json'
    if not path.exists():
        return False
    proof = json.loads(path.read_text())
    assert proof['complete'] and proof['spec'] == spec and proof['formal_CL_result'] is False
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_two_module_20261017/pilot').resolve())
    for name, digest in proof['artifact_sha256'].items():
        artifact = run/name
        assert artifact.resolve().is_relative_to(run.resolve()) and prior.sha(artifact) == digest
    gate = json.loads((run/'renderer_gpu_gate.json').read_text())
    assert gate['passed'] and len(gate['cases']) == 6
    if variant == 'control':
        assert json.loads((run/'original_trainer_exact_gate.json').read_text())['passed']
    if variant == 'B_only':
        assert json.loads((run/'matched_raw_gradient_gate.json').read_text())['exact_equal']
    # A rejected method step is a recorded feasibility failure, never a valid CL
    # score; predeclared other pilots may still expose its task dependence.
    training = json.loads((run/'training.json').read_text())
    assert proof['valid_training'] == training['valid_training']
    assert training['nonzero_effective_updates'] == int(training['valid_training'])
    return True


def choose(cap1, matched, pilots, cap3, busy):
    if busy:
        return 'wait', None
    if not cap1:
        return 'full', 1
    for index, done in enumerate(matched):
        if not done:
            return 'matched', index
    for index, done in enumerate(pilots):
        if not done:
            return 'pilot', index
    return ('complete', None) if cap3 else ('full', 3)


def snapshot(matched, pilot):
    failures = [str(p.relative_to(ROOT)) for folder in
                (BASE/'full_input_repair', BASE/'matched_task_domain', PILOT, REPORT)
                for p in folder.glob('*needs_attention.json')]
    assert not failures, failures
    owners = {u: int(subprocess.check_output(['systemctl', '--user', 'show', u, '-p', 'MainPID', '--value'],
                                            text=True).strip() or 0) for u in prior.OLD_OWNERS}
    pids = subprocess.check_output(['nvidia-smi', '--id=1', '--query-compute-apps=pid',
                                   '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    used, util = map(int, subprocess.check_output(['nvidia-smi', '--id=1',
        '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
    if any(owners.values()) or pids or used >= 500 or util >= 10:
        return ('wait', None), {'old_CPU_owners': owners, 'GPU1_PIDs': pids, 'used_MiB': used, 'utilization': util}
    full1, full3 = prior.complete_full(1), prior.complete_full(3)
    cases = [prior.complete_case(c, matched) for c in matched['cases']]
    pilots = [pilot_complete(c, pilot) for c in pilot['cases']]
    return choose(full1, cases, pilots, full3, False), {'cap1_complete': full1,
        'matched_complete_cases': sum(cases), 'pilot_complete_cases': sum(pilots), 'cap3_complete': full3}


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    matched, pilot = plans()
    if check:
        checks = [
            choose(False, [False]*48, [False]*12, False, True) == ('wait', None),
            choose(False, [False]*48, [False]*12, False, False) == ('full', 1),
            choose(True, [False]*48, [False]*12, False, False) == ('matched', 0),
            choose(True, [True]*48, [False]*12, False, False) == ('pilot', 0),
            choose(True, [True]*48, [True]*7+[False]*5, False, False) == ('pilot', 7),
            choose(True, [True]*48, [True]*12, False, False) == ('full', 3),
            choose(True, [True]*48, [True]*12, True, False) == ('complete', None)]
        assert all(checks)
        action, state = snapshot(matched, pilot)
        prior.write(REPORT/'CPU_PREPARATION.json', {'passed': True, 'policy_checks': len(checks),
            'CPU_only': True, 'active_GPU_not_changed': True, 'order': ORDER, 'current_action': action,
            'state': state, 'dispatcher_sha256': prior.sha(Path(__file__)),
            'previous_dispatcher_sha256': prior.sha(ROOT/'analysis/onevision_causal_priority_queue.py')})
        print(json.dumps({'passed': True, 'checks': len(checks), 'order': ORDER, 'GPU_started': False}))
        return
    authorization = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert authorization['approved_by_user'] is True and authorization['order'] == ORDER
    assert authorization['dispatcher_sha256'] == prior.sha(Path(__file__))
    lock = (REPORT/'coordinator.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        action, state = snapshot(matched, pilot)
        prior.write(REPORT/'queue_state.json', {'phase': action[0], 'target': action[1], 'pid': os.getpid(), 'state': state})
        if action[0] == 'wait':
            time.sleep(30);continue
        if action[0] == 'complete':
            return
        kind, index = action
        if kind == 'full':
            command = [str(ROOT/'analysis/onevision_full_input_repair.py'), '--cap', str(index)]
            log = BASE/'full_input_repair'/f'cap{index}_full_confirm_nohup.log'
        elif kind == 'matched':
            command = [str(ROOT/'analysis/onevision_matched_task_domain.py'), '--case', str(index)]
            a, s, m = matched['cases'][index]
            log = BASE/'matched_task_domain'/f'T{a:02d}_{s}__{m}_nohup.log'
        else:
            command = [str(ROOT/'analysis/onevision_two_module_pilot.py'), '--case', str(index)]
            task, variant = pilot['cases'][index];log = PILOT/f'T{task:02d}_{variant}_nohup.log'
        with log.open('a') as stream:
            child = subprocess.Popen(['/usr/bin/nohup', str(ROOT/'.vision-env/bin/python'), '-u', *command], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', PYTHONUNBUFFERED='1'),
                stdout=stream, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            prior.write(REPORT/'queue_state.json', {'phase': 'running', 'kind': kind, 'index': index,
                                                   'pid': os.getpid(), 'worker_pid': child.pid, 'GPU1_only': True})
            code = child.wait()
        assert code == 0, 'Worker failed; preserve evidence and stop, no blind retry'
        assert (prior.complete_full(index) if kind == 'full' else
                prior.complete_case(matched['cases'][index], matched) if kind == 'matched' else
                pilot_complete(pilot['cases'][index], pilot))


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true');args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        prior.write(REPORT/'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
