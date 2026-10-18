"""Add bounded-cache method trials; keep all original GPU1 work and guards."""
import argparse
import json
from pathlib import Path
import traceback
import onevision_identity_repair_queue_v4 as prior
import onevision_state_cache_report as reporter

ROOT, BASE = prior.ROOT, prior.STUDY.parent
STUDY = BASE/'functional_state_cache_20261018/v1'
REPORT = STUDY/'queue'
WORKER = 'analysis/onevision_state_cache_trial.py'


def plan():
    spec = json.loads((STUDY/'CPU_PLAN_CHECK.json').read_text())['spec']
    for n, expected in spec['source_sha256'].items():
        assert prior.sha(ROOT/n) == expected, n
    assert prior.sha(STUDY/'PROTOCOL.md') == spec['protocol_sha256']
    return spec


def complete(phase, view, spec):
    name = 'build' if phase == 'build' else 'full_'+view
    p = STUDY/(name+'_complete.json')
    if not p.exists():
        return False
    proof = json.loads(p.read_text())
    assert proof['complete'] and proof['spec'] == spec
    run = ROOT/proof['run']
    assert run.resolve().is_relative_to((ROOT/'exp/onevision_state_cache_20261018/v1').resolve())
    for n, expected in proof['artifact_sha256'].items():
        p = run/n
        assert p.resolve().is_relative_to(run.resolve()) and prior.sha(p) == expected
    assert json.loads((run/'original_exact_gate.json').read_text())['passed']
    if phase == 'full':
        assert proof['full_QA'] == 21454
        assert json.loads((run/'candidate_reload_exact.json').read_text())['passed']
    reporter.main()
    return True


def choose(action, built, data_done, full_done):
    if action[0] in ('wait', 'acquired_anchor', 'cosine_train', 'functional_pilot', 'warm_train', 'warm_quality_pilot'):
        return action
    if not built:
        return 'state_cache_build', None
    if data_done < 18:
        return action
    for i, done in enumerate(full_done):
        if not done:
            return 'state_cache_full', i
    return action


def extend(original, spec):
    warm = original.prior.cold.previous
    old_next, old_command = warm.next_action, warm.command_for
    old_verify = warm.previous.verify_worker

    def next_action(*args):
        assert not list(STUDY.glob('*needs_attention.json')), 'Preserve new-method failure; no retry'
        inherited_path = REPORT/'INHERITED_WORKER.json'
        if inherited_path.exists() and not (REPORT/'INHERITED_COMPLETE.json').exists():
            inherited = json.loads(inherited_path.read_text())
            proc = Path('/proc')/str(inherited['worker_pid'])
            try:
                ticks = int((proc/'stat').read_text().split(') ', 1)[1].split()[19])
            except (FileNotFoundError, ProcessLookupError):
                ticks = None
            if ticks == inherited['worker_startticks']:
                return ('wait', None), {'inherited_worker': inherited, 'waiting_for_natural_completion': True}
            assert old_verify(inherited['kind'], inherited['target'], args[0], args[3], args[4]), \
                'Inherited worker ended without complete evidence; preserve, never retry'
            (REPORT/'INHERITED_COMPLETE.json').write_text(json.dumps({
                'verified': True, 'original_worker': inherited, 'no_interrupt_or_prediction_splice': True}, indent=2)+'\n')
        action, state = old_next(*args)
        if action[0] in ('wait', 'acquired_anchor'):
            return action, state
        built = complete('build', None, spec)
        full_done = [complete('full', v, spec) for v in spec['full_candidates']] if built else [False, False]
        chosen = choose(action, built, state.get('cold_data_factorial_completed', 0), full_done)
        state.update(state_cache_build_complete=built, state_cache_full_complete=full_done)
        return chosen, state

    def command_for(kind, target, specs):
        if kind == 'state_cache_build':
            return [WORKER, '--phase', 'build'], STUDY/'build_nohup.log'
        if kind == 'state_cache_full':
            view = spec['full_candidates'][target]
            return [WORKER, '--phase', 'full', '--view', view], STUDY/('full_'+view+'_nohup.log')
        return old_command(kind, target, specs)

    def verify(kind, target, *args):
        if kind == 'state_cache_build':
            return complete('build', None, spec)
        if kind == 'state_cache_full':
            return complete('full', spec['full_candidates'][target], spec)
        return old_verify(kind, target, *args)

    warm.next_action, warm.command_for, warm.previous.verify_worker = next_action, command_for, verify
    # Verify exact new commands and old command preservation after real installation.
    assert command_for('state_cache_build', None, None)[0] == [WORKER, '--phase', 'build']
    for i, view in enumerate(spec['full_candidates']):
        assert command_for('state_cache_full', i, None)[0] == [WORKER, '--phase', 'full', '--view', view]
    for kind, target in [('data_factorial', 0), ('warm_train', None), ('functional_full', 0),
                         ('warm_quality_full', None), ('cosine_full', 1)]:
        assert command_for(kind, target, None) == old_command(kind, target, None)


def checks():
    n = 0
    for a in [('wait', None), ('acquired_anchor', 4), ('cosine_train', 0),
              ('functional_pilot', None), ('warm_train', None), ('warm_quality_pilot', None)]:
        assert choose(a, False, 0, [False, False]) == a;n += 1
    for a in [('data_factorial', 2), ('functional_full', 0), ('complete', None)]:
        assert choose(a, False, 2, [False, False]) == ('state_cache_build', None);n += 1
        assert choose(a, True, 2, [False, False]) == a;n += 1
        assert choose(a, True, 18, [False, False]) == ('state_cache_full', 0);n += 1
        assert choose(a, True, 18, [True, False]) == ('state_cache_full', 1);n += 1
        assert choose(a, True, 18, [True, True]) == a;n += 1
    return n


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    spec = plan()
    assert prior.resolved_attention()
    auth = json.loads((REPORT/'AUTHORIZATION.json').read_text())
    ancestor = json.loads((prior.REPORT/'AUTHORIZATION.json').read_text())
    assert all(auth.get(k) == v for k, v in ancestor.items()), 'Ancestor authorization fields must stay exact'
    assert prior.sha(Path(__file__)) == auth['state_cache_queue_sha256']
    assert prior.sha(ROOT/'analysis/onevision_identity_repair_queue_v4.py') == auth['unchanged_v4_sha256']
    assert prior.sha(Path(reporter.__file__)) == auth['reporter_sha256']
    assert auth['all_original_jobs_retained'] and auth['GPU1_only']
    assert auth['state_cache_order'] == ['preserve_current_GPU_natural_finish', 'new_continuous19_fourviews',
        'all_remaining18data', 'new_both_full21454', 'all_original_remaining_jobs']
    body = prior.compiled_queue()
    body['REPORT'] = REPORT
    original = body['original']
    if check:
        body['main'](True)
        extend(original, spec)
        result = {'CPU_only': True, 'checks': checks(), 'new_and_old_command_checks': 8,
                  'GPU_started': False, 'all_original_jobs_retained': True}
        (REPORT/'CPU_CHECK.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))
        return
    old_main = original.main

    def intercepted_main(flag=False):
        old_install = original.install

        def installed(study_spec):
            old_install(study_spec)
            extend(original, spec)
        original.install = installed
        return old_main(flag)

    original.main = intercepted_main
    body['main'](False)


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--check', action='store_true');args = p.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps({'traceback': traceback.format_exc(),
            'no_blind_retry': True}, indent=2)+'\n')
        raise
