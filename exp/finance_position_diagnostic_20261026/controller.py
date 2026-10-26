"""Run the one admitted four-state GPU readout; never allocate another slot.

The existing baseline coordinator owns this process and the GPU reservation.
Only its own matching worker process group may be stopped at the fixed deadline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ORDER = (('pct01', 15), ('pct01', 16), ('pct10', 15), ('pct10', 16))
BUDGET = 1200


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(pid):
    try:
        p = Path('/proc') / str(pid)
        fields = (p / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), startticks=fields[19], uid=p.stat().st_uid, pgid=int(fields[2]))
    except (OSError, ValueError):
        return None


def terminate_own(child, expected):
    if identity(child.pid) == expected:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if identity(child.pid) == expected:
                os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
    return identity(child.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--admission', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert os.getuid() == 1084 and args.gpu in range(4)
    assert args.admission.is_absolute() and args.output.is_absolute()
    admission = read(args.admission)
    assert admission['schema'] == 'finance-position-run-v1'
    assert Path(admission['source_dir']) == HERE
    for name, expected in admission['sources'].items():
        assert Path(name).name == name and sha(HERE / name) == expected
    assert set(admission['sources']) == {'worker.py', 'controller.py', 'queue_adapter.py', 'verify.py'}
    binding_path = Path(admission['binding']['path'])
    assert sha(binding_path) == admission['binding']['sha256']
    binding = read(binding_path)
    assert binding['worker_sha256'] == admission['sources']['worker.py']
    assert binding['protocol'] == admission['protocol']
    assert sha(Path(admission['protocol']['path'])) == admission['protocol']['sha256']
    assert sha(Path(admission['transform_binding']['path'])) == admission['transform_binding']['sha256']
    assert [(x['condition'], x['stage']) for x in binding['order']] == list(ORDER)
    assert binding['phase_budget_seconds'] == BUDGET and binding['planned_generations'] == 360
    output = args.output
    assert str(output) == admission['output_path'] and not output.exists() and output.parent.is_dir()
    assert all(not output.is_relative_to(Path(root)) for root in binding['data_roots'].values())
    output.mkdir(exist_ok=False)
    owner = identity(os.getpid())
    assert owner and owner['pgid'] == owner['pid']
    admission_sha = sha(args.admission)
    started = time.time()
    write(output / 'SUPERVISOR_STARTED.json', dict(unix=started, identity=owner,
        admission_sha256=admission_sha, GPU=args.gpu, phase_budget_seconds=BUDGET,
        cumulative_model_budget_seconds=4800, no_retry=True))
    phases, active, child = [], None, None
    try:
        for condition, stage in ORDER:
            name = f'{condition}_S{stage}'
            destination = output / name
            assert not destination.exists()
            env = dict(os.environ, LLMCL_ROOT=binding['data_roots'][condition],
                CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')
            command = [admission['python'], '-u', str(HERE / 'worker.py'),
                '--condition', condition, '--stage', str(stage), '--binding', str(binding_path),
                '--output', str(destination)]
            launch_unix, launch_clock = time.time(), time.monotonic()
            deadline = launch_clock + BUDGET
            timed_out = False
            with (output / (name + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=env['LLMCL_ROOT'], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active = identity(child.pid)
                assert active and active['uid'] == 1084 and active['pgid'] == child.pid
                write(output / ('PHASE_' + name + '_STARTED.json'), dict(unix=launch_unix,
                    deadline_unix=launch_unix+BUDGET, identity=active, condition=condition,
                    stage=stage, GPU=args.gpu, command=command, earliest_budget_start_includes_python_startup=True))
                try:
                    code = child.wait(timeout=max(.001, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_own(child, active)
                    code = child.returncode
            item = dict(condition=condition, stage=stage, identity=active,
                returncode=code, identity_absent=identity(child.pid) is None,
                timeout=timed_out, seconds=time.monotonic()-launch_clock)
            write(output / ('PHASE_' + name + '_EXIT.json'), item)
            phases.append(item)
            assert not timed_out and code == 0 and item['identity_absent'], name + ' failed; no continuation'
            active, child = None, None
            complete_path = destination / 'COMPLETE.json'
            complete = read(complete_path)
            assert complete['condition'] == condition and complete['stage'] == stage
            assert complete['worker_sha256'] == admission['sources']['worker.py']
            assert complete['binding_sha256'] == admission['binding']['sha256']
            assert complete['new_generations'] == 90 and complete['seconds'] <= BUDGET
            assert complete['full_parameters_and_registered_buffers_unchanged']
            assert all(complete[k] == 0 for k in ('new_scores', 'optimizer_updates', 'backwards', 'TEST_QA', 'CPU_model_passes'))
            for filename, expected in complete['files'].items():
                assert Path(filename).name == filename and sha(destination / filename) == expected
            item['complete_sha256'] = sha(complete_path)
        write(output / 'SUPERVISOR_COMPLETE.json', dict(unix=time.time(), identity=owner,
            admission_sha256=admission_sha, GPU=args.gpu, phases=phases, active_worker=None,
            no_retry=True, new_generations=360, models_seconds=sum(x['seconds'] for x in phases),
            total_seconds=time.time()-started, new_scores=0, new_training_updates=0))
    except BaseException:
        failure = traceback.format_exc()
        if child is not None and active is not None:
            try:
                terminate_own(child, active)
            except BaseException:
                failure += '\nCleanup observation: ' + traceback.format_exc()
        live = identity(active['pid']) if active else None
        write(output / 'SUPERVISOR_FAILURE.json', dict(unix=time.time(), identity=owner,
            admission_sha256=admission_sha, GPU=args.gpu, phases=phases, active_worker=live,
            last_attempted_worker=active, no_retry=True, traceback=failure,
            full_quality_missing=True, total_seconds=time.time()-started))
        raise
    # GPU model work has ended. A consumer error never rewrites model success
    # or creates a conflicting model-failure receipt, and is never retried here.
    tick = time.monotonic()
    command = [admission['python'], '-u', str(HERE / 'verify.py'),
        '--admission', str(args.admission), '--output', str(output)]
    try:
        with (output / 'verify.log').open('x') as log:
            result = subprocess.run(command, cwd=admission['root'], stdout=log,
                stderr=subprocess.STDOUT, timeout=120)
        write(output / 'CONSUMER_EXIT.json', dict(unix=time.time(), returncode=result.returncode,
            seconds=time.monotonic()-tick, no_retry=True, new_models=0, new_F1_scores=0))
        if result.returncode == 0:
            # The existing NEW archive already mirrors this report directory.
            # Copy only the consumer's public certificates, never prompts/QA.
            for name in ('OUTCOME.json', 'VERIFIED.json'):
                raw = (output / name).read_bytes()
                with (args.admission.parent / ('POSITION_' + name)).open('xb') as stream:
                    stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    except BaseException:
        write(output / 'CONSUMER_FAILURE.json', dict(unix=time.time(), seconds=time.monotonic()-tick,
            traceback=traceback.format_exc(), no_retry=True, model_complete_proof_unchanged=True))


if __name__ == '__main__':
    main()
