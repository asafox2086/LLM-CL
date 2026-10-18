"""User-authorized idle GPU0/1 selection, at most three GPUs owned by our jobs."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import traceback
import onevision_state_cache_queue as previous
import onevision_research_device_entry as device

ROOT, STUDY = previous.ROOT, previous.STUDY
REPORT = STUDY/'flex_queue'
ENTRY = 'analysis/onevision_research_device_entry.py'


def extend(original, spec):
    OLD_EXTEND(original, spec)
    warm = original.prior.cold.previous
    old_next, old_command, old_verify, old_write = warm.next_action, warm.command_for, warm.previous.verify_worker, warm.write
    data_spec = original.prior.cold.study_plan()

    def next_action(*args):
        action, state = old_next(*args)
        if state.get('waiting_for_natural_completion'):
            return action, state
        # Preserve all original admitted jobs. Only new cache jobs and unchanged
        # native factorial controls get the additional idle-GPU0 entry.
        if action[0] not in ('wait', 'state_cache_build', 'state_cache_full', 'data_factorial'):
            return action, state
        cards = device.inventory();gpu = device.select(cards, os.getuid())
        state.update(research_allowed_GPUs=[0, 1], max_our_GPU_cards=3,
                     actual_inventory=cards, selected_idle_GPU=gpu)
        if gpu is None:
            return ('wait', None), state
        if action[0] != 'wait':
            if action[0] == 'data_factorial':
                target = {'gpu': gpu, 'kind': 'factorial', 'index': action[1]}
            else:
                target = {'gpu': gpu, 'kind': 'cache', 'phase': 'build' if action[0] == 'state_cache_build' else 'full',
                          'view': None if action[0] == 'state_cache_build' else spec['full_candidates'][action[1]]}
            return ('flex_research', target), state
        # A wait can mean active GPU1. Idle GPU0 may serve the same frozen work;
        # source-failure checks in old_next ran before this hardware decision.
        if not previous.complete('build', None, spec):
            target = {'gpu': gpu, 'kind': 'cache', 'phase': 'build', 'view': None}
        else:
            pending = [i for i in range(18) if not original.prior.cold.study_complete(i, data_spec)]
            if pending:
                target = {'gpu': gpu, 'kind': 'factorial', 'index': pending[0]}
            else:
                full = [v for v in spec['full_candidates'] if not previous.complete('full', v, spec)]
                if not full:
                    return action, state
                target = {'gpu': gpu, 'kind': 'cache', 'phase': 'full', 'view': full[0]}
        return ('flex_research', target), state

    def command_for(kind, target, specs):
        if kind != 'flex_research':
            return old_command(kind, target, specs)
        command = [ENTRY, '--gpu', str(target['gpu']), '--kind', target['kind']]
        if target['kind'] == 'cache':
            command += ['--phase', target['phase']]
            if target['view'] is not None:
                command += ['--view', target['view']]
            label = target['phase']+'_'+str(target['view'])
        else:
            command += ['--case-index', str(target['index'])];label = 'factorial_'+str(target['index'])
        return command, STUDY/('physical_gpu'+str(target['gpu'])+'_'+label+'_nohup.log')

    def verify(kind, target, *args):
        if kind != 'flex_research':
            return old_verify(kind, target, *args)
        if target['kind'] == 'cache':
            name = 'build' if target['phase'] == 'build' else 'full_'+target['view']
            pointer = STUDY/(name+'_complete.json')
            valid = previous.complete(target['phase'], target['view'], spec)
        else:
            pointer = original.prior.cold.STUDY/f"case_{target['index']:02d}_complete.json"
            valid = original.prior.cold.study_complete(target['index'], data_spec)
        assert valid
        proof = json.loads(pointer.read_text());run = ROOT/proof['run']
        receipt_path = run/'physical_device_execution.json'
        receipt = json.loads(receipt_path.read_text())
        assert proof['artifact_sha256']['physical_device_execution.json'] == previous.prior.sha(receipt_path)
        auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
        assert receipt['device_entry_sha256'] == auth['new_device_sources'][ENTRY]
        assert receipt['physical_GPU'] == target['gpu']
        assert receipt['actual_CUDA_VISIBLE_DEVICES'] == str(target['gpu'])
        assert receipt['only_guard_literal_changed'] and receipt['max_our_GPU_cards'] == 3
        assert not receipt['admission_inventory'][str(target['gpu'])]['owners']
        return True

    def write(path, value):
        if Path(path) == REPORT/'queue_state.json':
            value = dict(value)
            value.pop('GPU1_only', None)
            value.update(research_allowed_GPUs=[0, 1], max_our_GPU_cards=3)
            if value.get('kind') == 'flex_research':
                value['physical_GPU'] = value['target']['gpu']
        return old_write(path, value)

    warm.next_action, warm.command_for, warm.previous.verify_worker, warm.write = next_action, command_for, verify, write
    for gpu in (0, 1):
        target = {'gpu': gpu, 'kind': 'cache', 'phase': 'build', 'view': None}
        assert command_for('flex_research', target, None)[0] == [ENTRY, '--gpu', str(gpu), '--kind', 'cache', '--phase', 'build']
    assert command_for('functional_full', 0, None) == old_command('functional_full', 0, None)


def checks():
    uid = 1004
    def cards(busy, ours):
        return {i: {'owners': [] if i not in busy else [{'uid': uid if i in ours else 1024}],
                    'memory_MiB': 3000 if i in busy else 6, 'utilization': 50 if i in busy else 0} for i in range(4)}
    assert device.select(cards({2, 3}, {2, 3}), uid) == 1
    assert device.select(cards({1, 2, 3}, {2, 3}), uid) == 0
    assert device.select(cards({0, 1, 2, 3}, {2, 3}), uid) is None
    assert device.select(cards({1, 2, 3}, {1, 2, 3}), uid) is None
    assert device.select(cards({0, 2, 3}, {2, 3}), uid) == 1
    assert device.select(cards({0, 2}, {0, 2}), uid) is None
    # Compile actual immutable target functions without importing torch/model.
    from types import SimpleNamespace
    for filename, function in [('onevision_state_cache_trial.py', 'run_trial'), ('onevision_cold_data_factorial.py', 'run_case')]:
        module = SimpleNamespace(__file__=str(ROOT/'analysis'/filename))
        for gpu in (0, 1):
            node, _ = device.compiled_function(module, function, gpu)
            assert node.body[0].test.comparators[0].value == str(gpu)
    return 10


OLD_EXTEND = previous.extend


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert device.ROOT == ROOT
    for n, expected in auth['new_device_sources'].items():
        assert previous.prior.sha(ROOT/n) == expected
    assert auth['research_allowed_GPUs'] == [0, 1] and auth['max_our_GPU_cards'] == 3
    previous.REPORT = REPORT
    previous.extend = extend
    previous.main(check)
    if check:
        n = checks()
        (REPORT/'DEVICE_CPU_CHECK.json').write_text(json.dumps({'CPU_only': True, 'checks': n,
            'only_CUDA_guard_literal_changed': True, 'GPU_started': False}, indent=2)+'\n')
        print(json.dumps({'device_CPU_checks': n, 'GPU_started': False}))


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--check', action='store_true');args = p.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps({'traceback': traceback.format_exc(),
            'no_blind_retry': True}, indent=2)+'\n')
        raise
