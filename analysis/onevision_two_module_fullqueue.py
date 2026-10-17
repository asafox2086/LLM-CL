"""New CPU dispatcher version; no changes to previous queued GPU sources.

After the running cap1: pilot12 -> full19x4 -> final21454x4 -> original matched48
-> original cap3 -> shared initial21454 -> sparse full transfer evaluation.
"""
import argparse
import fcntl
import json
import os
import subprocess
import time
import traceback
from pathlib import Path

import onevision_two_module_queue as previous

ROOT = previous.ROOT
BASE = previous.BASE
REPORT = BASE/'two_module_fullstream_queue'
FULL = BASE/'two_module_fullstream'
CONFIRM = FULL/'confirmation'
VARIANTS = ['control', 'A_only', 'B_only', 'A_B']
ORDER = ['cap1', 'pilot12', 'full19x4', 'final21454x4', 'matched48', 'cap3', 'shared_initial21454', 'sparse_fullx4']


def verify_artifacts(proof, parent):
    directory = ROOT/proof['run']
    assert directory.resolve().is_relative_to((ROOT/parent).resolve())
    for name, sha in proof['artifact_sha256'].items():
        path = directory/name
        assert path.resolve().is_relative_to(directory.resolve()) and previous.prior.sha(path) == sha
    return directory


def plans():
    matched, pilot = previous.plans()
    for name, source_key in [('CPU_ENGINE_VALIDATION.json', 'source_sha256'), ('CPU_WORKER_VALIDATION.json', 'sources')]:
        gate = json.loads((FULL/name).read_text());assert gate['passed'] and gate['CPU_only']
        for relative, sha in gate[source_key].items():
            assert previous.prior.sha(ROOT/relative) == sha
    training = json.loads((FULL/'CPU_FULLSTREAM_PLAN.json').read_text())['spec']
    confirmation = json.loads((CONFIRM/'CPU_CONFIRMATION_PLAN.json').read_text())['spec']
    assert confirmation['training_spec'] == training
    assert training['target_TRAIN_QA'] == 1313 and training['full_TEST_QA'] == 21454
    assert list(training['variants']) == VARIANTS
    assert matched['manifest_sha256'] == pilot['manifest_sha256'] == training['manifest_sha256']
    for relative, sha in confirmation['source_sha256'].items():
        assert previous.prior.sha(ROOT/relative) == sha
    return matched, pilot, training, confirmation


def full_training(variant, spec):
    complete = FULL/(variant+'_complete.json');invalid = FULL/(variant+'_invalid.json')
    assert not (complete.exists() and invalid.exists())
    if complete.exists():
        proof = json.loads(complete.read_text())
        assert proof['complete'] and proof['spec'] == spec and proof['trained_stages'] == 19 and proof['TRAIN_QA'] == 1313
        directory = verify_artifacts(proof, 'exp/onevision_two_module_20261017/fullstream')
        assert json.loads((directory/'renderer_gpu_gate.json').read_text())['passed']
        for stage in range(1, 20):
            training = json.loads((directory/'stages'/f'T{stage:02d}'/'training.json').read_text())
            assert training['effective_updates'] > 0
            if variant == 'control' or stage == 1:
                assert json.loads((directory/'stages'/f'T{stage:02d}'/'original_state_exact_gate.json').read_text())['passed']
        return 'valid'
    if invalid.exists():
        proof = json.loads(invalid.read_text());assert proof['spec'] == spec and proof['formal_score_available'] is False
        directory = ROOT/proof['run'];stage = proof['stage']
        assert directory.resolve().is_relative_to((ROOT/'exp/onevision_two_module_20261017/fullstream').resolve())
        folder = directory/'stages'/f'T{stage:02d}'
        assert previous.prior.sha(folder/'training.json') == proof['training_sha256']
        assert previous.prior.sha(folder/'training_memory.json') == proof['memory_sha256']
        assert json.loads((folder/'training.json').read_text())['effective_updates'] == 0
        assert (directory/'invalid_training.json').exists()
        return 'invalid'
    return 'pending'


def confirmation_complete(variant, phase, spec):
    name = 'shared_initial' if phase == 'initial' else variant+'_'+phase
    path = CONFIRM/(name+'_complete.json')
    if not path.exists():
        return False
    proof = json.loads(path.read_text())
    assert proof['complete'] and proof['spec'] == spec and proof['phase'] == phase and proof['optimizer_updates'] == 0
    owner = 'control' if phase == 'initial' else variant
    assert previous.prior.sha(FULL/(owner+'_complete.json')) == proof['training_proof_sha256']
    directory = verify_artifacts(proof, 'exp/onevision_two_module_20261017/confirmation')
    assert json.loads((directory/'renderer_gpu_gate.json').read_text())['passed']
    if phase == 'final':
        assert proof['full_final_QA'] == 21454
        assert json.loads((directory/'fresh_process_reload_gate.json').read_text())['passed']
    return True


def choose(cap1, pilots, training, final, matched, cap3, initial, sparse, busy=False):
    assert len(pilots) == 12 and len(training) == len(final) == len(sparse) == 4 and len(matched) == 48
    assert all(s in ('pending', 'valid', 'invalid') for s in training)
    if busy:
        return 'wait', None
    if not cap1:
        return 'full', 1
    for index, complete in enumerate(pilots):
        if not complete:
            return 'pilot', index
    for index, state in enumerate(training):
        if state == 'pending':
            return 'training', VARIANTS[index]
    for index, complete in enumerate(final):
        if training[index] == 'valid' and not complete:
            return 'final', VARIANTS[index]
    for index, complete in enumerate(matched):
        if not complete:
            return 'matched', index
    if not cap3:
        return 'full', 3
    assert training[0] == 'valid', 'Original SeqFT control must perform valid training'
    if not initial:
        return 'initial', 'control'
    for index, complete in enumerate(sparse):
        if training[index] == 'valid' and not complete:
            return 'sparse', VARIANTS[index]
    return 'complete', None


def snapshot(specs):
    failures = [str(p.relative_to(ROOT)) for directory in [BASE/'full_input_repair', BASE/'matched_task_domain',
                previous.PILOT, previous.REPORT, FULL, CONFIRM, REPORT] for p in directory.glob('*needs_attention.json')]
    assert not failures, failures
    owners = {unit: int(subprocess.check_output(['systemctl', '--user', 'show', unit, '-p', 'MainPID', '--value'], text=True).strip() or 0)
              for unit in previous.prior.OLD_OWNERS}
    pids = subprocess.check_output(['nvidia-smi', '--id=1', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip().splitlines()
    used, util = map(int, subprocess.check_output(['nvidia-smi', '--id=1', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).split(','))
    if any(owners.values()) or pids or used >= 500 or util >= 10:
        return ('wait', None), {'old_CPU_owners': owners, 'GPU1_PIDs': pids, 'used_MiB': used, 'utilization': util}
    matched, pilot, training, confirmation = specs
    cap1, cap3 = previous.prior.complete_full(1), previous.prior.complete_full(3)
    pilots = [previous.pilot_complete(case, pilot) for case in pilot['cases']]
    states = [full_training(v, training) for v in VARIANTS]
    final = [confirmation_complete(v, 'final', confirmation) if s == 'valid' else False for v, s in zip(VARIANTS, states)]
    sparse = [confirmation_complete(v, 'sparse', confirmation) if s == 'valid' else False for v, s in zip(VARIANTS, states)]
    initial = confirmation_complete('control', 'initial', confirmation)
    matches = [previous.prior.complete_case(case, matched) for case in matched['cases']]
    return choose(cap1, pilots, states, final, matches, cap3, initial, sparse), {
        'cap1': cap1, 'pilots': sum(pilots), 'training': dict(zip(VARIANTS, states)),
        'full_final': sum(final), 'matched': sum(matches), 'cap3': cap3, 'shared_initial': initial, 'sparse': sum(sparse)}


def check(specs):
    p, t, f, m, s = [True]*12, ['valid']*4, [True]*4, [True]*48, [True]*4
    tests = [choose(False, p, t, f, m, True, True, s, True) == ('wait', None),
             choose(False, p, t, f, m, True, True, s) == ('full', 1),
             choose(True, [False]*12, t, f, [False]*48, False, False, s) == ('pilot', 0),
             choose(True, p, ['pending']*4, f, m, False, False, s) == ('training', 'control'),
             choose(True, p, t, [False]*4, m, False, False, s) == ('final', 'control'),
             choose(True, p, t, f, [False]*48, False, False, s) == ('matched', 0),
             choose(True, p, t, f, m, False, False, s) == ('full', 3),
             choose(True, p, t, f, m, True, False, s) == ('initial', 'control'),
             choose(True, p, t, f, m, True, True, [False]*4) == ('sparse', 'control'),
             choose(True, p, ['valid', 'invalid', 'valid', 'invalid'], [True, False, True, False], m, True, True,
                    [True, False, True, False]) == ('complete', None),
             choose(True, p, t, f, m, True, True, s) == ('complete', None)]
    assert all(tests)
    action, state = snapshot(specs)
    previous.prior.write(REPORT/'CPU_DISPATCH_VALIDATION.json', {'passed': True, 'checks': len(tests),
        'CPU_only': True, 'dispatcher_sha256': previous.prior.sha(Path(__file__)), 'order': ORDER,
        'current_action': action, 'state': state, 'GPU_started': False})
    print(json.dumps({'passed': True, 'checks': len(tests), 'current_action': action, 'GPU_started': False}))


def main(check_only=False):
    REPORT.mkdir(parents=True, exist_ok=True);specs = plans()
    if check_only:
        return check(specs)
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert auth['approved_by_user'] and auth['order'] == ORDER and auth['dispatcher_sha256'] == previous.prior.sha(Path(__file__))
    lock = (REPORT/'coordinator.lock').open('a');fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        action, state = snapshot(specs);kind, target = action
        previous.prior.write(REPORT/'queue_state.json', {'phase': kind, 'target': target, 'pid': os.getpid(), 'state': state})
        if kind == 'wait':
            time.sleep(30);continue
        if kind == 'complete':
            return
        if kind == 'full':
            command = ['analysis/onevision_full_input_repair.py', '--cap', str(target)]
            log = BASE/'full_input_repair'/f'cap{target}_full_confirm_nohup.log'
        elif kind == 'pilot':
            command = ['analysis/onevision_two_module_pilot.py', '--case', str(target)]
            task, variant = specs[1]['cases'][target];log = previous.PILOT/f'T{task:02d}_{variant}_nohup.log'
        elif kind == 'matched':
            command = ['analysis/onevision_matched_task_domain.py', '--case', str(target)]
            a, sequence, method = specs[0]['cases'][target];log = BASE/'matched_task_domain'/f'T{a:02d}_{sequence}__{method}_nohup.log'
        elif kind == 'training':
            command = ['analysis/onevision_two_module_fullstream.py', '--variant', target]
            log = FULL/(target+'_nohup.log')
        else:
            command = ['analysis/onevision_two_module_confirmation.py', '--variant', target, '--phase', kind]
            log = CONFIRM/(target+'_'+kind+'_nohup.log')
        with log.open('a') as output:
            child = subprocess.Popen(['/usr/bin/nohup', str(ROOT/'.vision-env/bin/python'), '-u', *command], cwd=ROOT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6', PYTHONUNBUFFERED='1'),
                stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
            previous.prior.write(REPORT/'queue_state.json', {'phase': 'running', 'kind': kind, 'target': target,
                                                           'pid': os.getpid(), 'worker_pid': child.pid, 'GPU1_only': True})
            code = child.wait()
        assert code == 0, 'Worker failed; preserve evidence, no blind retry'
        if kind == 'full':
            assert previous.prior.complete_full(target)
        elif kind == 'pilot':
            assert previous.pilot_complete(specs[1]['cases'][target], specs[1])
        elif kind == 'matched':
            assert previous.prior.complete_case(specs[0]['cases'][target], specs[0])
        elif kind == 'training':
            assert full_training(target, specs[2]) in ('valid', 'invalid')
        else:
            assert confirmation_complete(target, kind, specs[3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true');args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        previous.prior.write(REPORT/'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
