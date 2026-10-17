"""CPU-only priority amendment: original causal48 before full-method training.

Preserves immutable workers and the12 small admission pilots. New entrypoint,
no active or queued-source edit. Tests/facts do not choose any model setting.
"""
import argparse
import json
import os
import traceback
from pathlib import Path

import onevision_two_module_fullqueue as original

ROOT = original.ROOT
REPORT = original.BASE/'causal_before_full_method_queue'
ORDER = ['cap1', 'pilot12', 'matched48', 'full19x4', 'final21454x4', 'cap3', 'shared_initial21454', 'sparse_fullx4']
original_choose = original.choose


def choose(cap1, pilots, training, final, matched, cap3, initial, sparse, busy=False):
    # Original policy still validates every vector and all train-state values.
    action = original_choose(cap1, pilots, training, final, matched, cap3, initial, sparse, busy)
    if busy or not cap1 or not all(pilots):
        return action
    for index, done in enumerate(matched):
        if not done:
            return 'matched', index
    return action


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    # Changes are process-local assignments in this new CPU entrypoint, not edits
    # to source or imports used by any GPU worker / proof / report observer.
    original.REPORT = REPORT;original.ORDER = ORDER;original.choose = choose
    if check:
        specs = original.plans();original.check(specs)
        pending, complete, valid = [False]*48, [True]*48, ['valid']*4
        checks = [choose(True, [True]*12, ['pending']*4, [False]*4, pending, False, False, [False]*4) == ('matched', 0),
                  choose(True, [True]*12, valid, [False]*4, pending, False, False, [False]*4) == ('matched', 0),
                  choose(True, [True]*12, ['pending']*4, [False]*4, complete, False, False, [False]*4) == ('training', 'control'),
                  choose(True, [True]*12, ['pending']*4, [False]*4, pending, False, False, [False]*4, True) == ('wait', None)]
        assert all(checks)
        original.previous.prior.write(REPORT/'CPU_PRIORITY_VALIDATION.json', {'passed': True, 'CPU_only': True,
            'inherited_policy_checks': 11, 'additional_priority_checks': len(checks), 'order': ORDER,
            'entrypoint_sha256': original.previous.prior.sha(Path(__file__)),
            'immutable_dispatch_library_sha256': original.previous.prior.sha(ROOT/'analysis/onevision_two_module_fullqueue.py'),
            'GPU_started': False})
        print(json.dumps({'CPU_only': True, 'checks': 15, 'order': ORDER, 'GPU_started': False}))
        return
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert auth['entrypoint_sha256'] == original.previous.prior.sha(Path(__file__))
    assert auth['approved_by_user'] and auth['order'] == ORDER
    original.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--check', action='store_true');args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        original.previous.prior.write(REPORT/'needs_attention.json', {'pid': os.getpid(), 'traceback': traceback.format_exc()})
        raise
