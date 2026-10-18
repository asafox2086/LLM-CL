"""Admit the inherited-pool quality cell without editing any queued GPU source."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import traceback

import onevision_causal_feedback_queue as prior

ROOT, BASE = prior.ROOT, prior.BASE
REPORT = BASE/'cold_quality_queue'
STUDY = BASE/'warm_quality_20261018'
WORKER = 'analysis/onevision_warm_quality_probe.py'
FULL = ['quality_hard']
ORDER = ['ready_sharedS1', 'original_cosine_train2', 'fixed_pool_functional_pilot4views',
         'existing_quality_hard_full21454_natural_completion', 'warm_initialization_only_train19',
         'warm_quality_pilot4views', 'all18_uniqueTRAIN_initialization_trajectories_both1and4_endpoints',
         'original_learned_cosine_full', 'warm_initialization_only_full21454',
         'warm_quality_hard_full21454', 'predeclared_quality_soft_full21454',
         'languagecue4', 'matched48_SeqFT_first', 'remaining_original_cosine_full',
         'all_original_remaining_jobs']
sha, write, startticks = prior.sha, prior.write, prior.startticks


def method_sources():
    names = [WORKER, 'analysis/onevision_cold_quality_queue.py',
             'analysis/onevision_cold_quality_report.py',
             'analysis/onevision_functional_router_probe.py',
             'analysis/onevision_functional_router_core.py',
             'analysis/onevision_warm_start_trial.py',
             'analysis/onevision_causal_feedback_queue.py']
    return {name: sha(ROOT/name) for name in names}


def checked_plan():
    p = STUDY/'CPU_PLAN_CHECK.json'
    if not p.exists():
        return None
    result = json.loads(p.read_text())
    assert result['passed'] and result['CPU_only'] and not result['GPU_started']
    spec = result['spec']
    assert spec['parent_variant'] == prior.cold.previous.VARIANT
    assert spec['factorial_cell'] == 'inherited_complete_AB + TRAIN_quality_hard'
    assert spec['full_candidates'] == FULL and spec['head_parameters'] == 16
    assert spec['LoRA_optimizer_updates'] == 0 and spec['additional_LoRAs'] == 0
    assert spec['all_four_diagnostic_views_required'] and not spec['TEST_used_for_calibration_or_choice']
    assert spec['calibration_TRAIN_ids'] and len(spec['calibration_TRAIN_ids']) == 43
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    parent = prior.cold.previous.STUDY/(spec['parent_variant']+'_train_complete.json')
    assert sha(parent) == spec['parent_completion_sha256']
    value = json.loads(parent.read_text())
    assert sha(ROOT/value['run']/'latest.pt') == spec['parent_state_sha256']
    return spec


def prepare_plan(warm):
    assert prior.cold.previous.warm_complete('train', warm), 'Sealed warm19 required'
    spec = checked_plan()
    if spec is None:
        STUDY.mkdir(parents=True, exist_ok=True)
        assert not list(STUDY.glob('*needs_attention.json')), 'No failed-attempt retry'
        with (STUDY/'parent_plan_CPU.log').open('a') as output:
            result = subprocess.run([str(ROOT/'.vision-env/bin/python'), '-u', WORKER, '--check'],
                cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='6', MKL_NUM_THREADS='6'),
                stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        assert result.returncode == 0, 'CPU parent plan failed; preserve evidence'
        spec = checked_plan()
    assert spec is not None
    return spec


def complete(scope, view, spec):
    path = STUDY/('pilot_complete.json' if scope == 'pilot' else 'full_'+view+'_complete.json')
    if not path.exists():
        return False
    assert spec is not None
    p = json.loads(path.read_text())
    assert p['complete'] and p['spec'] == spec and p['LoRA_optimizer_updates'] == 0 and p['extra_LoRAs'] == 0
    run = prior.cold.previous.previous.previous.original.verify_artifacts(p, spec['run_namespace'])
    assert p['parent_state_sha256'] == spec['parent_state_sha256']
    assert json.loads((run/'original_exact_gate.json').read_text())['passed']
    if scope == 'pilot':
        assert p['all_views'] == spec['views']
        c = json.loads((run/'calibration_summary.json').read_text())
        assert c['parameters'] == 16 and c['TRAIN_QA'] == 43 and c['NLL_forwards'] == 172
        assert c['CPU_head_steps'] == spec['head_steps'] and c['sealed_before_intervention_TEST']
    else:
        assert view == 'quality_hard' and p['full_QA'] == 21454
        assert json.loads((run/'candidate_reload_exact.json').read_text())['passed']
    return True


def choose(action, warm_train, pilot, data_done, cold_full, warm_full, combined_full):
    if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot', 'warm_train') or not warm_train:
        return action
    if not pilot:
        return 'warm_quality_pilot', None
    if data_done != 18:
        return action  # Unchanged feedback policy retains every data trajectory.
    if not cold_full:
        return 'cosine_full', 1
    if not warm_full:
        return 'warm_full', None
    if not combined_full:
        return 'warm_quality_full', None
    return action


def checks():
    tests = []
    for action in [('wait', None), ('acquired_anchor', 4), ('cosine_train', 1),
                   ('functional_pilot', None), ('warm_train', None)]:
        for trained in (False, True):
            for pilot in (False, True):
                tests.append(choose(action, trained, pilot, 0, False, False, False) == action)
    for action in [('data_factorial', 0), ('functional_full', 1), ('matched', 24), ('complete', None)]:
        tests += [choose(action, False, False, 0, False, False, False) == action,
                  choose(action, True, False, 0, False, False, False) == ('warm_quality_pilot', None),
                  choose(action, True, True, 0, False, False, False) == action,
                  choose(action, True, True, 18, False, False, False) == ('cosine_full', 1),
                  choose(action, True, True, 18, True, False, False) == ('warm_full', None),
                  choose(action, True, True, 18, True, True, False) == ('warm_quality_full', None),
                  choose(action, True, True, 18, True, True, True) == action]
    for index in range(18):
        tests.append(choose(('data_factorial', index), True, True, index, False, False, False) == ('data_factorial', index))
    assert all(tests)
    return len(tests)


def install(spec):
    prior.REPORT, prior.ORDER = REPORT, ORDER
    prior.install(spec)
    warm = prior.cold.previous
    old_next, old_command = warm.next_action, warm.command_for
    old_verify = warm.previous.verify_worker

    def next_action(*args):
        assert not list(STUDY.glob('*needs_attention.json')), 'Preserve combined failure'
        action, state = old_next(*args)
        if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot', 'warm_train') or not state.get('warm_train_completed'):
            return action, state
        # This CPU check may be prepared only from the actually sealed warm pool.
        plan = prepare_plan(args[5])
        pilot = complete('pilot', None, plan)
        full = complete('full', 'quality_hard', plan)
        done = state['cold_data_factorial_completed']
        state.update(warm_quality_pilot_completed=pilot, warm_quality_full_completed=full,
                     warm_quality_plan_sha256=sha(STUDY/'CPU_PLAN_CHECK.json'))
        chosen = choose(action, True, pilot, done, state['cold_learned_full_completed'], state['warm_full_completed'], full)
        return chosen, state

    def command_for(kind, target, specs):
        if kind == 'warm_quality_pilot':
            return [WORKER, '--scope', 'pilot'], STUDY/'pilot_nohup.log'
        if kind == 'warm_quality_full':
            return [WORKER, '--scope', 'full', '--view', 'quality_hard'], STUDY/'full_quality_hard_nohup.log'
        return old_command(kind, target, specs)

    def verify(kind, target, *args):
        if kind in ('warm_quality_pilot', 'warm_quality_full'):
            return complete('pilot' if kind == 'warm_quality_pilot' else 'full',
                            None if kind == 'warm_quality_pilot' else 'quality_hard', checked_plan())
        return old_verify(kind, target, *args)

    warm.next_action, warm.command_for, warm.previous.verify_worker = next_action, command_for, verify


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    n, spec = checks(), prior.cold.study_plan()
    prior.cold.previous.warm_plan()
    if check:
        write(REPORT/'CPU_POLICY_CHECK.json', {'passed': True, 'CPU_only': True, 'checks': n,
            'GPU_started': False, 'source_sha256': method_sources(), 'order': ORDER,
            'warm_parent_may_be_pending': True, 'no_TEST_choice': True, 'all_previous_jobs_retained': True})
        print(json.dumps({'passed': True, 'CPU_only': True, 'checks': n, 'GPU_started': False}))
        return
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    assert auth['source_sha256'] == method_sources()
    assert auth['order'] == ORDER and auth['all_previous_jobs_retained'] and auth['no_TEST_choice']
    assert auth['wrapper_entrypoint_sha256'] == sha(ROOT/'analysis/onevision_causal_feedback_queue.py')
    assert auth['cold_queue_sha256'] == sha(ROOT/'analysis/onevision_cold_data_queue.py')
    assert auth['warm_plan_sha256'] == sha(prior.cold.previous.STUDY/'CPU_PLAN_CHECK.json')
    assert auth['study_plan_sha256'] == sha(prior.cold.STUDY/'CPU_PLAN_CHECK.json')
    install(spec)
    prior.cold.previous.main(False)


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
