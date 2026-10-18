"""Traceable recovery and warm-feedback scheduling after the aux-ID failure.

All original conditions survive. No old source, partial predictions or model
weights are modified. Warm19/pilot/data controls now precede the new full retry
to avoid another long full run blocking the first initialization result.
"""
import argparse
import json
from pathlib import Path
import traceback

import onevision_cold_quality_queue as original

ROOT, BASE = original.ROOT, original.BASE
STUDY = BASE/'feature_identity_repair_20261018'
REPORT = STUDY/'queue_v2'
FUNCTIONAL = STUDY/'functional_router'
ENTRY = 'analysis/onevision_identity_repair_worker.py'
WRAPPED = {'onevision_functional_router_probe.py', 'onevision_warm_start_trial.py',
           'onevision_learned_reuse_trial.py', 'onevision_warm_quality_probe.py',
           'onevision_cold_data_factorial.py'}
ORDER = ['ready_dependencies', 'warm19', 'warm_quality_pilot', 'all18_data_controls',
         'cold_quality_hard_full21454_fresh', 'all_original_remaining_jobs_unchanged']


def prioritize(action, trained, pilot, done, first_full):
    if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot'):
        return action
    if not trained:
        return 'warm_train', None
    if not pilot:
        return 'warm_quality_pilot', None
    for i, finished in enumerate(done):
        if not finished:
            return 'data_factorial', i
    if not first_full:
        return 'functional_full', 0
    return action


def checks():
    tests = []
    for a in [('wait', None), ('acquired_anchor', 4), ('cosine_train', 1), ('functional_pilot', None)]:
        tests.append(prioritize(a, False, False, [False]*18, False) == a)
    for a in [('functional_full', 0), ('cosine_full', 1), ('complete', None)]:
        tests += [prioritize(a, False, False, [False]*18, False) == ('warm_train', None),
                  prioritize(a, True, False, [False]*18, False) == ('warm_quality_pilot', None),
                  prioritize(a, True, True, [False]*18, False) == ('data_factorial', 0),
                  prioritize(a, True, True, [True]*18, False) == ('functional_full', 0),
                  prioritize(a, True, True, [True]*18, True) == a]
    for i in range(18):
        tests.append(prioritize(('functional_full', 0), True, True,
                                [j < i for j in range(18)], False) == ('data_factorial', i))
    assert all(tests)
    return len(tests)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    count = checks()
    auth = json.loads((STUDY/'AUTHORIZATION.json').read_text())
    assert auth['effective_order'] == ORDER and auth['failure_preserved'] and auth['no_old_partial_predictions']
    for name, expected in auth['new_source_sha256'].items():
        assert original.sha(ROOT/name) == expected
    assert auth['original_method_sources'] == original.method_sources()
    router = original.prior.cold.previous.previous.previous
    router.STUDY = FUNCTIONAL
    original.REPORT = REPORT
    install_original = original.install

    def install(spec):
        install_original(spec)
        warm = original.prior.cold.previous
        old_next, old_command, old_checks = warm.next_action, warm.command_for, warm.checks

        def next_action(*args):
            assert not list(STUDY.glob('worker_*_needs_attention.json')), 'Preserve repair failure'
            action, state = old_next(*args)
            if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot'):
                return action, state
            trained = state['warm_train_completed']
            done = [original.prior.cold.study_complete(i, spec) for i in range(18)] if trained else [False]*18
            first = router.complete('full', 'quality_hard', args[4])
            chosen = prioritize(action, trained, state.get('warm_quality_pilot_completed', False), done, first)
            state.update(feature_identity_repair=True, failed_old_full_excluded=True,
                         recovery_priority='warm feedback before fresh complete retry')
            return chosen, state

        def command_for(kind, target, specs):
            command, log = old_command(kind, target, specs)
            if Path(command[0]).name in WRAPPED:
                command = [ENTRY, '--worker', command[0], '--', *command[1:]]
                log = STUDY/('worker_'+kind+'_'+str(target)+'_nohup.log')
            return command, log

        def integrated_checks():
            # Keep every unchanged ancestor command assertion intact on the
            # unchanged underlying command. Then verify each NEW entry wraps
            # that command exactly, rather than deleting the old assertions.
            current = warm.command_for
            warm.command_for = old_command
            try:
                previous_count = old_checks()
            finally:
                warm.command_for = current
            cases = [('warm_train', None), ('warm_full', None), ('functional_full', 0),
                     ('functional_full', 1), ('cosine_full', 1), ('data_factorial', 0),
                     ('warm_quality_pilot', None), ('warm_quality_full', None)]
            for kind, target in cases:
                before, _ = old_command(kind, target, None)
                after, log = command_for(kind, target, None)
                expected = [ENTRY, '--worker', before[0], '--', *before[1:]]
                assert after == expected and log.parent == STUDY
            return previous_count+len(cases)

        warm.next_action, warm.command_for, warm.checks = next_action, command_for, integrated_checks

    original.install = install
    if check:
        install(original.prior.cold.study_plan())
        integrated = original.prior.cold.previous.checks()
        print(json.dumps({'CPU_only': True, 'policy_checks': count,
                          'ancestor_policy_checks': original.checks(),
                          'integrated_ancestor_and_actual_command_checks': integrated,
                          'GPU_started': False, 'all_original_conditions_retained': True}))
        return
    original.main(False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        original.write(REPORT/'needs_attention.json', {'traceback': traceback.format_exc(), 'no_blind_retry': True})
        raise
