"""CPU-only priority change: causal controls before the second full confirmation.

Never interrupt the current full worker; retain every predeclared condition.
All model workers and prior queue sources remain immutable.
"""
import argparse
import json
from pathlib import Path
import traceback

import onevision_cold_data_queue as cold

ROOT, BASE = cold.ROOT, cold.BASE
REPORT = BASE/'causal_feedback_queue'
ORDER = ['ready_sharedS1', 'original_cosine_train2', 'fixed_pool_functional_pilot4views',
         'existing_quality_hard_full21454_natural_completion',
         'warm_initialization_only_train19',
         'all18_uniqueTRAIN_initialization_trajectories_both1and4_endpoints',
         'predeclared_quality_soft_full21454', 'original_learned_cosine_full',
         'warm_initialization_only_full21454', 'languagecue4',
         'matched48_SeqFT_first', 'remaining_original_cosine_full',
         'all_original_remaining_jobs']
sha, write, startticks = cold.sha, cold.write, cold.startticks


def choose(action, first_full, trained, done):
    assert len(done) == 18
    if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot') or not first_full:
        return action
    if not trained:
        return 'warm_train', None
    for i, complete in enumerate(done):
        if not complete:
            return 'data_factorial', i
    return action


def checks():
    tests = []
    no, yes = [False]*18, [True]*18
    for a in [('wait', None), ('acquired_anchor', 4), ('cosine_train', 1), ('functional_pilot', None)]:
        for first in (False, True):
            for trained in (False, True):
                tests.append(choose(a, first, trained, no) == a)
    for a in [('functional_full', 1), ('cosine_full', 1), ('matched', 24), ('complete', None)]:
        tests += [choose(a, False, False, no) == a,
                  choose(a, False, True, no) == a,
                  choose(a, True, False, no) == ('warm_train', None),
                  choose(a, True, True, no) == ('data_factorial', 0),
                  choose(a, True, True, [True]*5+[False]*13) == ('data_factorial', 5),
                  choose(a, True, True, [True, False]+[True]*16) == ('data_factorial', 1),
                  choose(a, True, True, yes) == a]
    # Same original commands, all 18 trajectories; no TEST score is an input.
    for i in range(18):
        done = [j < i for j in range(18)]
        tests.append(choose(('functional_full', 1), True, True, done) == ('data_factorial', i))
        tests.append(cold.command_for('data_factorial', i, None)[0] ==
                     ['analysis/onevision_cold_data_factorial.py', '--case-index', str(i)])
    tests += [cold.command_for('warm_train', None, None)[0] ==
              ['analysis/onevision_warm_start_trial.py', '--phase', 'train'],
              cold.command_for('functional_full', 1, None)[0] ==
              ['analysis/onevision_functional_router_probe.py', '--scope', 'full', '--view', 'quality_soft']]
    assert all(tests)
    return len(tests)


def install(spec):
    cold.REPORT, cold.ORDER = REPORT, ORDER
    cold.install(spec)  # Retains original verifier/commands, including factorial.

    def next_action(*args):
        assert not list(cold.STUDY.glob('*needs_attention.json')), 'Preserve failure; no retry'
        # Original warm dispatcher includes actual inherited-worker liveness,
        # completion/source checks and foreign-GPU admission gates.
        action, state = cold.OLD_NEXT(*args)
        if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot'):
            return action, state
        functional = args[4]
        router = cold.previous.previous.previous
        first = router.complete('full', router.FULL[0], functional)
        trained = state['warm_train_completed']
        done = [cold.study_complete(i, spec) for i in range(18)] if trained else [False]*18
        state.update(first_functional_full_complete=first,
                     cold_data_factorial_completed=sum(done),
                     scheduling_basis='causal feedback before second full; all conditions retained')
        return choose(action, first, trained, done), state

    cold.previous.next_action = next_action


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    n, spec = checks(), cold.study_plan()
    warm = cold.previous.warm_plan()
    if check:
        write(REPORT/'CPU_POLICY_CHECK.json', {
            'passed': True, 'CPU_only': True, 'checks': n, 'GPU_started': False,
            'wrapper_entrypoint_sha256': sha(Path(__file__)),
            'cold_queue_sha256': sha(ROOT/'analysis/onevision_cold_data_queue.py'),
            'study_plan_sha256': sha(cold.STUDY/'CPU_PLAN_CHECK.json'),
            'warm_plan_sha256': sha(cold.previous.STUDY/'CPU_PLAN_CHECK.json'),
            'order': ORDER, 'all_previous_jobs_retained': True, 'no_TEST_choice': True})
        print(json.dumps({'passed': True, 'CPU_checks': n, 'GPU_started': False}))
        return
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert auth['wrapper_entrypoint_sha256'] == sha(Path(__file__))
    assert auth['cold_queue_sha256'] == sha(ROOT/'analysis/onevision_cold_data_queue.py')
    assert auth['study_plan_sha256'] == sha(cold.STUDY/'CPU_PLAN_CHECK.json')
    assert auth['warm_plan_sha256'] == sha(cold.previous.STUDY/'CPU_PLAN_CHECK.json')
    assert auth['order'] == ORDER and auth['all_previous_jobs_retained'] and auth['no_TEST_choice']
    install(spec)
    cold.previous.main(False)  # Unchanged loop, locking, admission and failure gates.


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write(REPORT/'needs_attention.json', {'traceback': traceback.format_exc()})
        raise
