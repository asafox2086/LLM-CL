"""One stream: preserve current work, test same-budget coverage, retain all jobs."""
import argparse
import json
import os
from pathlib import Path
import traceback

import onevision_answer_compatibility_queue as previous
import onevision_feedback_coverage_report as reporter
import onevision_research_device_entry as device

ROOT, STUDY = previous.ROOT, reporter.REPORT
REPORT = STUDY/'queue'
ENTRY = 'analysis/onevision_feedback_coverage_entry.py'
OLD_EXTEND = previous.extend
read, sha = reporter.read, reporter.sha


def choose(action, inherited, parent_built, built, data_done, full_done, gpu):
    if inherited or action[0] == 'acquired_anchor' or not parent_built:
        return action
    if built and (not all(data_done) or all(full_done)):
        return action
    if gpu is None:
        return 'wait', None
    if not built:
        return 'feedback_coverage', dict(gpu=gpu, phase='build', view=None)
    return 'feedback_coverage', dict(gpu=gpu, phase='full', view=reporter.VIEWS[full_done.index(False)])


def extend(original, native_spec):
    OLD_EXTEND(original, native_spec)
    spec = read(STUDY/'CPU_WORKER_PLAN.json')['spec']
    warm = original.prior.cold.previous
    old_next, old_command, old_verify = warm.next_action, warm.command_for, warm.previous.verify_worker
    data_spec = original.prior.cold.study_plan()
    def next_action(*args):
        inherited = REPORT/'COVERAGE_INHERITED_WORKER.json'
        if inherited.exists() and not (REPORT/'COVERAGE_INHERITED_COMPLETE.json').exists():
            item = read(inherited)
            try:
                s = (Path('/proc')/str(item['worker_pid'])/'stat').read_text()
                ticks = int(s[s.rfind(')')+2:].split()[19])
            except (FileNotFoundError, ProcessLookupError):
                ticks = None
            if ticks is not None and ticks == item['worker_startticks']:
                return ('wait', None), dict(waiting_for_natural_completion=True, coverage_inherited_worker=item)
            assert old_verify(item['kind'], item['target'], args[0], args[3], args[4])
            (REPORT/'COVERAGE_INHERITED_COMPLETE.json').write_text(json.dumps(dict(verified=True, original_worker=item))+'\n')
        action, state = old_next(*args)
        if state.get('waiting_for_natural_completion'):
            return action, state
        assert not list(STUDY.glob('*needs_attention.json')), 'Preserve failure; no blind retry'
        old_spec = read(previous.STUDY/'CPU_PLAN_CHECK.json')['spec']
        parent_built = previous.reporter.complete('build', None, old_spec)
        built = reporter.complete('build', None, spec)
        done = [original.prior.cold.study_complete(i, data_spec) for i in range(18)]
        full = [reporter.complete('full', view, spec) for view in reporter.VIEWS]
        reporter.main()
        new = choose(action, False, parent_built, built, done, full,
                     device.select(device.inventory(), os.getuid()))
        state.update(coverage_build_complete=built, coverage_full_complete=full,
                     coverage_full_after_original18=True, all_original_jobs_retained=True)
        return new, state
    def command(kind, target, specs):
        if kind != 'feedback_coverage':
            return old_command(kind, target, specs)
        cmd = [ENTRY, '--gpu', str(target['gpu']), '--phase', target['phase']]
        if target['view']:
            cmd += ['--view', target['view']]
        return cmd, STUDY/(target['phase']+'_'+str(target['view'])+'_nohup.log')
    def verify(kind, target, *args):
        if kind != 'feedback_coverage':
            return old_verify(kind, target, *args)
        assert reporter.complete(target['phase'], target['view'], spec)
        name = 'build' if target['phase'] == 'build' else 'full_'+target['view']
        proof = read(STUDY/(name+'_complete.json'))
        receipt = read(ROOT/proof['run']/'physical_device_execution.json')
        assert receipt['physical_GPU'] == target['gpu']
        assert not receipt['admission_inventory'][str(target['gpu'])]['owners']
        reporter.main()
        return True
    warm.next_action, warm.command_for, warm.previous.verify_worker = next_action, command, verify
    assert command('feedback_coverage', dict(gpu=0, phase='build', view=None), None)[0] == [ENTRY, '--gpu', '0', '--phase', 'build']
    assert command('answer_compatibility', dict(gpu=1, phase='build', view=None), None) == old_command('answer_compatibility', dict(gpu=1, phase='build', view=None), None)
    assert command('flex_research', dict(gpu=0, kind='factorial', index=3), None) == old_command('flex_research', dict(gpu=0, kind='factorial', index=3), None)


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    auth = read(REPORT/'AUTHORIZATION.json')
    for name, expected in auth['feedback_coverage_sources'].items():
        assert sha(ROOT/name) == expected
    assert sha(STUDY/'PROTOCOL.md') == auth['feedback_coverage_protocol_sha256']
    spec = read(STUDY/'CPU_WORKER_PLAN.json')['spec']
    for name, expected in spec['source_sha256'].items():
        assert sha(ROOT/name) == expected
    previous.REPORT = REPORT
    previous.extend = extend
    previous.main(check)
    if check:
        a = ('flex_research', dict(gpu=0, kind='factorial', index=4))
        assert choose(a, True, True, False, [False]*18, [False]*2, 0) == a
        assert choose(a, False, False, False, [False]*18, [False]*2, 0) == a
        assert choose(('acquired_anchor', 4), False, True, False, [False]*18, [False]*2, 0) == ('acquired_anchor', 4)
        assert choose(a, False, True, False, [False]*18, [False]*2, None) == ('wait', None)
        assert choose(a, False, True, False, [False]*18, [False]*2, 0)[1] == dict(gpu=0, phase='build', view=None)
        assert choose(a, False, True, True, [False]*18, [False]*2, 0) == a
        assert choose(a, False, True, True, [True]*18, [False]*2, 1)[1]['view'] == reporter.VIEWS[0]
        assert choose(a, False, True, True, [True]*18, [True, False], 1)[1]['view'] == reporter.VIEWS[1]
        assert choose(a, False, True, True, [True]*18, [True]*2, 0) == a
        (REPORT/'COVERAGE_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(CPU_only=True,
            checks=9, actual_new_and_old_commands=3, GPU_started=False))+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    a = p.parse_args()
    try:
        main(a.check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT/'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(), no_blind_retry=True))+'\n')
        raise
