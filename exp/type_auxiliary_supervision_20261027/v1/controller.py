"""One supplied GPU, seven serial phases, no retry or resource allocation.

Only this controller's exact-identity worker process group can be terminated.
No model/scoring imports and no automatic consumer invocation.
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
ORDER = [('prefix','prefix')] + [(arm, phase) for arm in ('plain','true','shuffled') for phase in ('train','eval')]
LIMITS = {'prefix':3600, 'train':2400, 'eval':600}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), startticks=fields[19], uid=path.stat().st_uid,
                    pgid=int(fields[2]))
    except (OSError, ValueError, IndexError):
        return None


def terminate_own(child, expected):
    """Never signal on PID alone; a changed/missing identity is left alone."""
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
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    assert args.root.is_absolute() and args.root.is_dir()
    assert args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists()
    assert args.python.is_absolute() and args.python.is_file()
    assert args.gpu in range(4)
    # The worker uses physical nvidia-smi indices and matching CUDA indices.
    # This controller must not silently remap a supplied card to cuda:0.
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3'), 'Use the original unmasked GPU index space'
    sources = {name: sha(HERE / name) for name in ('controller.py', 'worker.py', 'helpers.py', 'training.py')}
    assert args.binding.is_absolute() and args.binding.is_file()
    binding = read(args.binding)
    assert binding['schema'] == 'type-auxiliary-supervision-v1' and binding['limits'] == LIMITS
    assert binding['sources'] == {n: sources[n] for n in ('worker.py', 'helpers.py', 'training.py')}
    binding_sha256 = sha(args.binding)
    protocol_path = Path(binding['protocol']['path'])
    assert protocol_path.is_absolute() and sha(protocol_path) == binding['protocol']['sha256']
    protocol = dict(path=str(protocol_path), sha256=sha(protocol_path))
    owner = identity(os.getpid())
    assert owner and owner['uid'] == os.getuid() and owner['pgid'] == owner['pid']
    args.output.mkdir(exist_ok=False)
    first_unix, first_clock = time.time(), time.monotonic()
    write(args.output / 'SUPERVISOR_STARTED.json', dict(unix=first_unix,
        identity=owner, GPU=args.gpu, source_dir=str(HERE), source_sha256=sources,
        binding_sha256=binding_sha256, protocol=protocol, order=[dict(arm=a, phase=p) for a, p in ORDER],
        phase_limits=LIMITS, cumulative_model_budget_seconds=12600,
        existing_allocator_supplied_GPU=True, no_retry=True, new_scores=0))
    def interrupted(signum, frame):
        raise InterruptedError('Controller interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    phases, child, active, last_attempted = [], None, None, None
    try:
        for arm, phase in ORDER:
            assert all(sha(HERE / name) == expected for name, expected in sources.items())
            assert sha(protocol_path) == protocol['sha256']
            assert sha(args.binding) == binding_sha256
            destination = args.output / arm
            if phase in ('prefix','train'):
                destination.mkdir(exist_ok=False)
            else:
                assert destination.is_dir() and (destination / 'train_COMPLETE.json').is_file()
            assert not (destination / (phase + '_STARTED.json')).exists()
            label, budget = arm + '_' + phase, LIMITS[phase]
            command = [str(args.python), '-u', str(HERE / 'worker.py'),
                '--root', str(args.root), '--run', str(destination), '--arm', arm,
                '--gpu', str(args.gpu), '--phase', phase, '--binding', str(args.binding)]
            if phase == 'train':
                command.extend(['--prefix-run',str(args.output/'prefix')])
            launched, clock = time.time(), time.monotonic()
            deadline = clock + budget
            timed_out = False
            with (destination / (phase + '.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen(command, cwd=args.root, env=dict(os.environ,
                    PYTHONUNBUFFERED='1', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'),
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active = identity(child.pid)
                assert active and active['uid'] == owner['uid'] and active['pgid'] == child.pid
                last_attempted = active
                write(args.output / ('PHASE_' + label + '_STARTED.json'), dict(
                    unix=launched, deadline_unix=launched+budget, identity=active,
                    arm=arm, phase=phase, GPU=args.gpu, run_dir=str(destination),
                    command=command, limit_seconds=budget, includes_python_startup=True))
                try:
                    code = child.wait(timeout=max(.001, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True
                    terminate_own(child, active)
                    code = child.returncode
            item = dict(arm=arm, phase=phase, identity=active, returncode=code,
                identity_absent=identity(child.pid) is None, timeout=timed_out,
                seconds=time.monotonic()-clock, run_dir=str(destination),
                launch_unix=launched, exit_unix=time.time())
            write(args.output / ('PHASE_' + label + '_EXIT.json'), item)
            phases.append(item)
            assert not timed_out and code == 0 and item['identity_absent'], label + ' failed; no continuation'
            complete_path = destination / (phase + '_COMPLETE.json')
            started_path = destination / (phase + '_STARTED.json')
            complete, started = read(complete_path), read(started_path)
            assert started['identity'] == complete['identity'] == active
            assert started['arm'] == complete['arm'] == arm
            assert started['phase'] == complete['phase'] == phase
            assert started['GPU'] == complete['GPU'] == args.gpu
            assert started['limit_seconds'] == budget and complete['seconds'] <= budget
            assert complete['new_scores'] == complete['TEST_QA'] == 0
            counts = {'prefix':(120,959,69), 'train':(80,254,18), 'eval':(105,0,0)}
            ng, nb, nu = counts[phase]
            assert complete['new_generations'] == ng
            assert complete['FIT_backwards'] == nb and complete['optimizer_updates'] == nu
            assert complete['data_binding']['binding_sha256'] == binding_sha256
            if phase == 'prefix':
                checkpoint = complete['checkpoint']
                assert checkpoint['stage'] == 15 and checkpoint['full_architecture_and_runtime']
                checkpoint = dict(checkpoint, path=str(destination/checkpoint['path']))
                assert Path(checkpoint['path']).stat().st_size == checkpoint['bytes']
            else:
                training = complete if phase == 'train' else read(destination/'train_COMPLETE.json')
                checkpoint = training['checkpoint']
                assert Path(checkpoint['path']) == destination/'complete.pt'
                assert Path(checkpoint['path']).stat().st_size == checkpoint['bytes']
                if phase == 'eval':
                    assert complete['reloaded']['digest'] == checkpoint['digest']
                    assert complete['reloaded']['layout'] == checkpoint['layout']
                    assert complete['complete_state_and_buffers_unchanged']
            item.update(started_sha256=sha(started_path), complete_sha256=sha(complete_path),
                checkpoint={k: checkpoint[k] for k in ('path', 'bytes', 'sha256')})
            active, child = None, None
        write(args.output / 'SUPERVISOR_COMPLETE.json', dict(unix=time.time(),
            identity=owner, GPU=args.gpu, source_dir=str(HERE), source_sha256=sources,
            binding_sha256=binding_sha256, protocol=protocol, phases=phases, active_worker=None, no_retry=True,
            new_generations=675, new_scores=0, FIT_backwards=1721,
            optimizer_updates=123, TEST_QA=0,
            model_phase_seconds=sum(p['seconds'] for p in phases),
            total_seconds=time.monotonic()-first_clock,
            cumulative_model_budget_seconds=12600,
            no_automatic_consumer=True))
    except BaseException:
        failure = traceback.format_exc()
        if child is not None and active is not None:
            try:
                terminate_own(child, active)
            except BaseException:
                failure += '\nCleanup observation: ' + traceback.format_exc()
        write(args.output / 'SUPERVISOR_FAILURE.json', dict(unix=time.time(),
            identity=owner, GPU=args.gpu, source_sha256=sources, protocol=protocol,
            phases=phases, active_worker=identity(active['pid']) if active else None,
            last_attempted_worker=last_attempted, no_retry=True, traceback=failure,
            full_quality_missing=True, total_seconds=time.monotonic()-first_clock))
        raise


if __name__ == '__main__':
    main()
