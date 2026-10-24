"""Single CPU owner; three isolated phases and identity-bound independent caps."""
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common import CASES, HERE, REPORT, WORK, check_ready, inventory, process_identity, sha, write


def own_group_signal(process, expected, sig):
    current = process_identity(process.pid)
    assert current['uid'] == os.getuid() == expected['uid']
    assert current['pid'] == expected['pid'] and current['startticks'] == expected['startticks']
    assert current['pgid'] == process.pid == expected['pgid']
    os.killpg(current['pgid'], sig)


def child(phase, physical, ready):
    public = REPORT / 'phases' / phase
    public.mkdir(parents=True, exist_ok=True)
    started_path = public / 'STARTED.json'
    assert not started_path.exists()
    log = (public / 'worker.log').open('ab')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
    process = subprocess.Popen([sys.executable, '-u', str(HERE / 'run.py'), '--phase', phase,
        '--physical-gpu', str(physical)], stdout=log, stderr=subprocess.STDOUT, env=env,
        cwd=HERE, start_new_session=True)
    expected = None
    try:
        expected = process_identity(process.pid)
        assert expected['pgid'] == process.pid
        while process.poll() is None:
            remaining = 2.
            if started_path.exists():
                started = json.loads(started_path.read_text())
                for key in ('pid', 'uid', 'startticks', 'pgid', 'boot_id'):
                    assert started[key] == expected[key]
                cap = ready['init_budget_seconds'] if phase == 'init' else ready['case_budget_seconds']
                assert started['budget_seconds'] == cap
                remaining = started['monotonic'] + cap - time.monotonic()
                if remaining <= 0:
                    # Only this isolated child group, never another GPU owner.
                    own_group_signal(process, expected, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        own_group_signal(process, expected, signal.SIGKILL)
                        process.wait(timeout=10)
                    write(public / 'WATCHDOG_TIMEOUT.json', dict(phase=phase, worker=expected,
                        budget_seconds=cap, recovery_not_additional_model_budget=True))
                    raise TimeoutError('Independent worker deadline reached; entire fixed pilot closed')
            time.sleep(min(2., remaining))
        code = process.wait()
    except BaseException:
        # A controller validation failure must not leave its own model running.
        # Recheck immutable identity before every signal; never use a GPU PID list.
        if expected is None or expected['pgid'] != process.pid:
            write(public / 'CHILD_IDENTITY_UNCONFIRMED.json', dict(pid=process.pid,
                observed_returncode=process.poll(), no_signal_to_unconfirmed_identity=True,
                needs_attention=True, no_retry=True))
            raise
        if process.poll() is None:
            own_group_signal(process, expected, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                own_group_signal(process, expected, signal.SIGKILL)
                process.wait(timeout=10)
        write(public / 'CONTROLLER_CHILD_CLOSED.json', dict(worker=expected,
            returncode=process.wait(), child_reaped=True, no_retry=True))
        raise
    finally:
        log.close()
    if code == 75:
        assert not started_path.exists() and not (public / 'FAILURE.json').exists()
        return False
    complete = public / 'COMPLETE.json'
    write(public / 'EXIT.json', dict(returncode=code, observed_wait_returned=True,
        worker=expected, started_sha256=sha(started_path) if started_path.exists() else None,
        complete_sha256=sha(complete) if complete.exists() else None))
    assert code == 0 and complete.exists(), 'Started failure: no retry/continuation'
    return True


def main():
    ready = check_ready()
    admission = json.loads((REPORT / 'ADMISSION_REVIEW.json').read_text())
    assert admission['decision'] == 'accept_single_fixed_sparse_pilot_activation'
    assert admission['ready_sha256'] == sha(REPORT / 'READY.json')
    assert sys.executable == ready['executable']
    assert not WORK.exists() and not any((REPORT / 'phases' / p / 'STARTED.json').exists() for p in ('init', *CASES))
    REPORT.mkdir(parents=True, exist_ok=True)
    owner = process_identity()
    with (REPORT / 'CONTROLLER_OWNER.json').open('x', encoding='utf8') as stream:
        json.dump(owner, stream); stream.flush(); os.fsync(stream.fileno())
    physical = None
    for phase in ('init', *CASES):
        while True:
            cards = inventory()
            if physical is None:
                available = [k for k in ready['allowed_physical_GPUs'] if not cards[k]['owners']
                    and cards[k]['memory_MiB'] < 500 and cards[k]['utilization'] < 10]
                if available:
                    physical = min(available)
                    write(REPORT / 'DEVICE_SELECTION.json', dict(host='OLD', physical_GPU=physical,
                        uuid=cards[physical]['uuid'], controller=owner))
            free = physical is not None and not cards[physical]['owners'] and cards[physical]['memory_MiB'] < 500 and cards[physical]['utilization'] < 10
            if not free:
                write(REPORT / 'queue.json', dict(phase='wait', next_phase=phase, controller=owner,
                    physical_GPU=physical, no_new_model_attempt=True))
                time.sleep(60)
                continue
            write(REPORT / 'queue.json', dict(phase='launching', next_phase=phase, controller=owner, physical_GPU=physical))
            if child(phase, physical, ready):
                break
            time.sleep(60)
        if phase == 'init':
            subprocess.run([sys.executable, str(HERE / 'verify.py'), '--initial-only'], cwd=HERE,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
            initial = json.loads((REPORT / 'INITIAL_VERIFIED.json').read_text())
            if not initial['resource_gate_passed']:
                write(REPORT / 'queue.json', dict(phase='closed_initial_function_negative', controller=owner,
                    initial_verified_sha256=sha(REPORT / 'INITIAL_VERIFIED.json'), no_training_or_DEV=True))
                return
    subprocess.run([sys.executable, str(HERE / 'verify.py')], cwd=HERE,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    write(REPORT / 'queue.json', dict(phase='complete_verified', controller=owner,
        verified_sha256=sha(REPORT / 'VERIFIED.json'), no_automatic_full_confirmation=True))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write(REPORT / 'CONTROLLER_FAILURE.json', dict(traceback=traceback.format_exc(), no_retry=True))
        write(REPORT / 'queue.json', dict(phase='closed_failure', needs_attention=True, no_retry=True))
        raise
