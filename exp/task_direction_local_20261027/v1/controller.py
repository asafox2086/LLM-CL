"""Fixed GPU1, four serial local-S16 phases; no retry or resource allocation.

Only this controller's exact-identity worker process group can be terminated.
No model/scoring imports and no automatic consumer invocation.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ORDER = [(arm, phase) for arm in ('aligned', 'random') for phase in ('train', 'eval')]
LIMITS = {'train': 1800, 'eval': 900}


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
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    assert args.root.is_absolute() and args.root.is_dir()
    assert args.output.is_absolute() and args.output.parent.is_dir() and not args.output.exists()
    assert args.python.is_absolute() and args.python.is_file()
    assert args.gpu == 1, 'The sealed shared-prefix graph and CUDA state use physical GPU1'
    # The worker uses physical nvidia-smi indices and matching CUDA indices.
    # This controller must not silently remap a supplied card to cuda:0.
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3'), 'Use the original unmasked GPU index space'
    assert args.binding.is_absolute()
    binding_sha256=sha(args.binding);binding=read(args.binding)
    assert binding['schema']=='task-direction-local-v1'
    sources=binding['sources']
    assert set(sources)=={'controller.py','worker.py','adaptation.py','calibration.py','verify.py','prefix.py'}
    assert all(sha(HERE/name)==expected for name,expected in sources.items())
    assert [p['stage'] for p in binding['training_plan']]==[16]
    assert sum(p['QA'] for p in binding['training_plan'])==69
    assert sum(p['updates'] for p in binding['training_plan'])==5
    assert all(p['calibration_QA']==min(16,p['QA']) for p in binding['training_plan'])
    calibration_backwards=sum(p['calibration_QA'] for p in binding['training_plan'])
    protocol_path=Path(binding['protocol']['path'])
    assert protocol_path.is_absolute()
    protocol=dict(path=str(protocol_path),sha256=sha(protocol_path))
    assert protocol==binding['protocol']
    for relative,expected in binding['inherited_sources'].items():
        path=Path(relative)
        assert not path.is_absolute() and '..' not in path.parts
        assert sha(args.root/path)==expected, relative
    # This is a metadata check, not loading a model or resuming an old run.
    spec=importlib.util.spec_from_file_location('local_direction_prefix',HERE/'prefix.py')
    prefix=importlib.util.module_from_spec(spec);spec.loader.exec_module(prefix)
    reused_prefix=prefix.validate_metadata(binding)
    assert isinstance(reused_prefix,dict)
    owner = identity(os.getpid())
    assert owner and owner['uid'] == os.getuid() and owner['pgid'] == owner['pid']
    args.output.mkdir(exist_ok=False)
    first_unix, first_clock = time.time(), time.monotonic()
    write(args.output/'REUSED_PREFIX.json',reused_prefix)
    reused_prefix_sha256=sha(args.output/'REUSED_PREFIX.json')
    write(args.output / 'SUPERVISOR_STARTED.json', dict(unix=first_unix,
        identity=owner, GPU=args.gpu, source_dir=str(HERE), source_sha256=sources,
        protocol=protocol,binding_path=str(args.binding),binding_sha256=binding_sha256, order=[dict(arm=a, phase=p) for a, p in ORDER],
        phase_limits=LIMITS, cumulative_model_budget_seconds=5400,
        existing_allocator_supplied_GPU=True, no_retry=True, new_scores=0,
        reused_prefix=reused_prefix,reused_prefix_sha256=reused_prefix_sha256,
        prefix_history_is_not_new_execution=True,planned_scoring_calls=80))
    def interrupted(signum, frame):
        raise InterruptedError('Controller interrupted by signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    phases, child, active, last_attempted = [], None, None, None
    try:
        for arm, phase in ORDER:
            assert all(sha(HERE / name) == expected for name, expected in sources.items())
            assert sha(protocol_path) == protocol['sha256']
            assert sha(args.binding)==binding_sha256
            destination = args.output / arm
            if phase == 'train':
                destination.mkdir(exist_ok=False)
            else:
                assert destination.is_dir() and (destination / 'train_COMPLETE.json').is_file()
            assert not (destination / (phase + '_STARTED.json')).exists()
            label, budget = arm + '_' + phase, LIMITS[phase]
            command = [str(args.python), '-u', str(HERE / 'worker.py'),
                '--root', str(args.root), '--run', str(destination), '--arm', arm,
                '--gpu', str(args.gpu), '--phase', phase,'--binding',str(args.binding)]
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
            assert complete['data_binding']['binding_sha256']==binding_sha256
            if phase=='train':assert complete['calibration_backwards']==calibration_backwards
            assert complete['new_generations'] == (20 if phase == 'train' else 25)
            assert complete['FIT_backwards'] == (69 if phase == 'train' else 0)
            assert complete['optimizer_updates'] == (5 if phase == 'train' else 0)
            # Do not read/hash the large checkpoint again here. The train
            # producer already hashed it; eval checks that exact file then
            # performs a strict full-state reload. Preserve both receipts.
            training = complete if phase == 'train' else read(destination / 'train_COMPLETE.json')
            checkpoint = training['checkpoint']
            assert Path(checkpoint['path']) == destination / 'complete.pt'
            assert (destination / 'complete.pt').stat().st_size == checkpoint['bytes']
            if phase == 'eval':
                assert complete['reloaded']['digest'] == checkpoint['digest']
                assert complete['reloaded']['layout'] == checkpoint['layout']
                assert complete['complete_state_and_buffers_unchanged']
            item.update(started_sha256=sha(started_path), complete_sha256=sha(complete_path),
                checkpoint={k: checkpoint[k] for k in ('path', 'bytes', 'sha256')})
            active, child = None, None
        write(args.output / 'SUPERVISOR_COMPLETE.json', dict(unix=time.time(),
            identity=owner, GPU=args.gpu, source_dir=str(HERE), source_sha256=sources,
            protocol=protocol,binding_path=str(args.binding),binding_sha256=binding_sha256, phases=phases, active_worker=None, no_retry=True,
            new_generations=90, new_scores=0, FIT_backwards=138,
            optimizer_updates=10, TEST_QA=0,calibration_backwards=2*calibration_backwards,
            model_phase_seconds=sum(p['seconds'] for p in phases),
            total_seconds=time.monotonic()-first_clock,
            cumulative_model_budget_seconds=5400,
            no_automatic_consumer=True,planned_scoring_calls=80,
            reused_prefix=reused_prefix,reused_prefix_sha256=reused_prefix_sha256,
            prefix_history_is_not_new_execution=True))
    except BaseException:
        failure = traceback.format_exc()
        if child is not None and active is not None:
            try:
                terminate_own(child, active)
            except BaseException:
                failure += '\nCleanup observation: ' + traceback.format_exc()
        write(args.output / 'SUPERVISOR_FAILURE.json', dict(unix=time.time(),
            identity=owner, GPU=args.gpu, source_sha256=sources, protocol=protocol,
            binding_path=str(args.binding),binding_sha256=binding_sha256,
            phases=phases, active_worker=identity(active['pid']) if active else None,
            last_attempted_worker=last_attempted, no_retry=True, traceback=failure,
            full_quality_missing=True, total_seconds=time.monotonic()-first_clock,
            reused_prefix=reused_prefix,reused_prefix_sha256=reused_prefix_sha256,
            prefix_history_is_not_new_execution=True))
        raise


if __name__ == '__main__':
    main()
