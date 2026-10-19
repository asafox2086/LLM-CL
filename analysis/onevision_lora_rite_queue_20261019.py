"""Add one published-optimizer trial after the existing TRAIN-only pilot."""
import argparse
import json
import os
import traceback
import onevision_train_update_priority_v2 as previous
import onevision_lora_rite_report_20261019 as reporter
import onevision_research_device_entry as device

ROOT = previous.ROOT
STUDY = reporter.REPORT
REPORT = STUDY / 'queue'
OLD_EXTEND = previous.extend


def choose(action, inherited, pilot_done, train_done, full_done, gpu):
    if inherited or action[0] in ('acquired_anchor', 'TRAIN_update_pilot') or not pilot_done or full_done:
        return action
    if gpu is None:
        return 'wait', None
    return 'lora_rite', dict(gpu=gpu, phase='full' if train_done else 'train')


def extend(original, native_spec):
    OLD_EXTEND(original, native_spec)
    warm = original.prior.cold.previous
    old_next, old_command, old_verify = warm.next_action, warm.command_for, warm.previous.verify_worker
    spec = reporter.read(STUDY / 'CPU_PLAN.json')['spec']
    def next_action(*args):
        action, state = old_next(*args)
        if state.get('waiting_for_natural_completion'):
            return action, state
        assert not (STUDY / 'needs_attention.json').exists()
        pilot_done = previous.previous.complete()
        trained, full = [reporter.complete(p, spec) for p in ('train', 'full')]
        state.update(lora_rite_train_complete=trained, lora_rite_full_complete=full,
                     lora_rite_after_existing_TRAIN_pilot=True, all_previous_fulls_retained=True)
        return choose(action, False, pilot_done, trained, full,
                      device.select(device.inventory(), os.getuid())), state
    def command(kind, target, specs):
        if kind != 'lora_rite':
            return old_command(kind, target, specs)
        return ['analysis/onevision_lora_rite_entry_20261019.py', '--gpu', str(target['gpu']),
                '--phase', target['phase']], STUDY / (target['phase'] + '_nohup.log')
    def verify(kind, target, *args):
        if kind != 'lora_rite':
            return old_verify(kind, target, *args)
        assert reporter.complete(target['phase'], spec)
        run = ROOT / reporter.read(STUDY / (target['phase'] + '_complete.json'))['run']
        receipt = reporter.read(run / 'physical_device_execution.json')
        assert receipt['gpu'] == target['gpu'] and not receipt['inventory'][str(target['gpu'])]['owners']
        reporter.main()
        return True
    warm.next_action, warm.command_for, warm.previous.verify_worker = next_action, command, verify
    assert command('lora_rite', dict(gpu=0, phase='train'), None)[0] == [
        'analysis/onevision_lora_rite_entry_20261019.py', '--gpu', '0', '--phase', 'train']


def main(check=False):
    REPORT.mkdir(parents=True, exist_ok=True)
    auth = reporter.read(REPORT / 'AUTHORIZATION.json')
    for rel, expected in auth['lora_rite_sources'].items():
        assert reporter.sha(ROOT / rel) == expected
    assert reporter.sha(STUDY / 'CPU_PLAN.json') == auth['lora_rite_CPU_plan_sha256']
    assert reporter.sha(STUDY / 'PROTOCOL.md') == auth['lora_rite_protocol_sha256']
    previous.REPORT, previous.extend = REPORT, extend
    previous.main(check)
    if check:
        a = ('directional_coeff', dict(gpu=1, phase='full', view='random_coeff_r16'))
        assert choose(a, True, True, False, False, 1) == a
        assert choose(('acquired_anchor', 4), False, True, False, False, 1) == ('acquired_anchor', 4)
        assert choose(a, False, False, False, False, 1) == a
        p = ('TRAIN_update_pilot', dict(gpu=1)); assert choose(p, False, True, False, False, 1) == p
        assert choose(a, False, True, False, False, None) == ('wait', None)
        assert choose(a, False, True, False, False, 0) == ('lora_rite', dict(gpu=0, phase='train'))
        assert choose(a, False, True, True, False, 1) == ('lora_rite', dict(gpu=1, phase='full'))
        assert choose(a, False, True, True, True, 1) == a
        (REPORT / 'RITE_QUEUE_CPU_CHECK.json').write_text(json.dumps(dict(checks=8, all_original_work_retained=True,
            single_research_stream=True, CPU_only=True, GPU_started=False)) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--check', action='store_true')
    try:
        main(parser.parse_args().check)
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / 'needs_attention.json').write_text(json.dumps(dict(traceback=traceback.format_exc(), no_blind_retry=True)) + '\n')
        raise
