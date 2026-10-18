"""One stream, user-directed OSFT mechanism controls, all old work retained."""
import argparse
import json
import os
from pathlib import Path
import traceback

import onevision_feedback_coverage_queue as previous
import onevision_osft_budget_report_v2 as reporter
import onevision_research_device_entry as device

ROOT, STUDY = previous.ROOT, reporter.REPORT
REPORT = STUDY/'queue'
ENTRY = 'analysis/onevision_osft_budget_entry_v2.py'
OLD_EXTEND = previous.extend
read, sha = reporter.read, reporter.sha


def choose(action, inherited, coverage_built, trains, data_done, fulls, gpu):
    if inherited or action[0] == 'acquired_anchor' or not coverage_built:
        return action
    if all(trains) and (not all(data_done) or all(fulls)):
        return action
    if gpu is None:
        return 'wait', None
    if not all(trains):
        return 'osft_budget', dict(gpu=gpu, phase='train', view=reporter.VIEWS[trains.index(False)])
    return 'osft_budget', dict(gpu=gpu, phase='full', view=reporter.VIEWS[fulls.index(False)])


def extend(original, native_spec):
    OLD_EXTEND(original, native_spec)
    spec = read(STUDY/'CPU_PLAN.json')['spec']
    warm = original.prior.cold.previous
    old_next, old_command, old_verify = warm.next_action, warm.command_for, warm.previous.verify_worker
    data_spec = original.prior.cold.study_plan()
    def next_action(*args):
        inherited = REPORT/'SVD_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'SVD_INHERITED_COMPLETE.json').exists():
            item = read(inherited)
            try:
                s = (Path('/proc')/str(item['worker_pid'])/'stat').read_text()
                ticks = int(s[s.rfind(')')+2:].split()[19])
            except (FileNotFoundError, ProcessLookupError):
                ticks = None
            if ticks is not None and ticks == item['worker_startticks']:
                return ('wait', None), dict(waiting_for_natural_completion=True, inherited_worker=item)
            assert old_verify(item['kind'], item['target'], args[0], args[3], args[4])
            (REPORT/'SVD_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True, worker=item))+'\n')
        action, state = old_next(*args)
        if state.get('waiting_for_natural_completion'):
            return action, state
        assert not list(STUDY.glob('*needs_attention.json'))
        old_spec = read(previous.STUDY/'CPU_WORKER_PLAN.json')['spec']
        built = previous.reporter.complete('build', None, old_spec)
        trains = [reporter.complete('train', view, spec) for view in reporter.VIEWS]
        fulls = [reporter.complete('full', view, spec) for view in reporter.VIEWS]
        data = [original.prior.cold.study_complete(i, data_spec) for i in range(18)]
        reporter.main()
        state.update(SVD_train_complete=trains, SVD_full_complete=fulls,
                     all_original_jobs_retained=True, SVD_full_after_original18=True)
        return choose(action, False, built, trains, data, fulls,
                      device.select(device.inventory(), os.getuid())), state
    def command(kind, target, specs):
        if kind != 'osft_budget':
            return old_command(kind, target, specs)
        return [ENTRY, '--gpu', str(target['gpu']), '--phase', target['phase'], '--view', target['view']], STUDY/(target['phase']+'_'+target['view']+'_nohup.log')
    def verify(kind, target, *args):
        if kind != 'osft_budget':
            return old_verify(kind, target, *args)
        assert reporter.complete(target['phase'], target['view'], spec)
        run = ROOT/read(STUDY/(target['phase']+'_'+target['view']+'_complete.json'))['run']
        receipt = read(run/'physical_device_execution.json')
        assert receipt['physical_GPU'] == target['gpu']
        assert not receipt['inventory'][str(target['gpu'])]['owners']
        reporter.main()
        return True
    warm.next_action, warm.command_for, warm.previous.verify_worker = next_action, command, verify
    assert command('osft_budget', dict(gpu=0, phase='train', view=reporter.VIEWS[0]), None)[0] == [ENTRY, '--gpu', '0', '--phase', 'train', '--view', reporter.VIEWS[0]]
    assert command('feedback_coverage', dict(gpu=1, phase='build', view=None), None) == old_command('feedback_coverage', dict(gpu=1, phase='build', view=None), None)
    assert command('flex_research', dict(gpu=0, kind='factorial', index=4), None) == old_command('flex_research', dict(gpu=0, kind='factorial', index=4), None)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    auth = read(REPORT/'AUTHORIZATION.json')
    for name, expected in auth['osft_budget_sources'].items():
        assert sha(ROOT/name) == expected
    assert sha(STUDY/'PROTOCOL.md') == auth['osft_budget_protocol_sha256']
    previous.REPORT, previous.extend = REPORT, extend
    previous.main(check)
    if check:
        a = ('flex_research', dict(gpu=0, kind='factorial', index=4))
        assert choose(a, True, True, [False]*2, [False]*18, [False]*2, 0) == a
        assert choose(a, False, False, [False]*2, [False]*18, [False]*2, 0) == a
        assert choose(('acquired_anchor', 4), False, True, [False]*2, [False]*18, [False]*2, 0) == ('acquired_anchor', 4)
        assert choose(a, False, True, [False]*2, [False]*18, [False]*2, None) == ('wait', None)
        assert choose(a, False, True, [False]*2, [False]*18, [False]*2, 0)[1]['view'] == reporter.VIEWS[0]
        assert choose(a, False, True, [True, False], [False]*18, [False]*2, 1)[1]['view'] == reporter.VIEWS[1]
        assert choose(a, False, True, [True]*2, [False]*18, [False]*2, 0) == a
        assert choose(a, False, True, [True]*2, [True]*18, [False]*2, 0)[1]['phase'] == 'full'
        assert choose(a, False, True, [True]*2, [True]*18, [True, False], 1)[1]['view'] == reporter.VIEWS[1]
        assert choose(a, False, True, [True]*2, [True]*18, [True]*2, 0) == a
        (REPORT/'SVD_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True, checks=10,
            actual_new_and_old_commands=3, GPU_started=False))+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser();p.add_argument('--check', action='store_true')
    args = p.parse_args()
    try:
        main(args.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(), no_blind_retry=True))+'\n')
        raise
