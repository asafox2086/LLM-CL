"""One supplied GPU, six new phases plus one sealed reused prefix; no retry.

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
ORDER = [(arm, phase) for arm in ('plain','true','shuffled') for phase in ('train','eval')]
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


def validate_reused_prefix(binding):
    """Read-only original S15 full graph/source/exit validation; no model imports.

    Returns deterministic provenance. It does not relabel the old failed parent
    experiment as success or create new prefix STARTED/EXIT records.
    """
    expected = binding['reused_prefix']
    assert expected['complete_sha256'] == 'ae5847e12d2773b54f5129dc59cd22921795c83556bc72a25d6b120e04af9de0'
    assert expected['checkpoint_sha256'] == 'efb578fe8bd65430284d5d5e305ad0eb55946d06ef623265f304db3ba2ab69fe'
    assert expected['binding_sha256'] == 'dd2969ea9ba8d20286fc5e5e5a76db74fd74808fddbc12b5bb77088908423ae0'
    run = Path(expected['run']); assert run.is_absolute() and run.name == 'prefix'
    complete_path = run/'prefix_COMPLETE.json'
    assert sha(complete_path) == expected['complete_sha256']
    complete = read(complete_path); started = read(run/'prefix_STARTED.json')
    assert expected['GPU'] == complete['GPU'] == started['GPU'] == 1
    assert complete['arm'] == complete['phase'] == started['arm'] == started['phase'] == 'prefix'
    assert complete['identity'] == started['identity'] and identity(started['identity']['pid']) != started['identity']
    assert complete['new_scores'] == complete['TEST_QA'] == 0
    assert (complete['new_generations'],complete['optimizer_updates'],complete['FIT_backwards']) == (120,69,959)
    assert complete['before_QA'] == complete['acquired_QA'] == 60
    assert complete['data_binding']['binding_sha256'] == expected['binding_sha256']
    assert read(run/'INPUT_BINDING.json') == complete['data_binding']
    assert read(run/'SOURCE_BINDING.json') == complete['source_sha256']
    parent = run.parent
    phase_start = read(parent/'PHASE_prefix_prefix_STARTED.json')
    assert phase_start['GPU'] == expected['GPU']
    exit_path = parent/'PHASE_prefix_prefix_EXIT.json'; exited = read(exit_path)
    assert phase_start['identity'] == exited['identity'] == started['identity']
    assert phase_start['run_dir'] == exited['run_dir'] == str(run)
    assert exited['returncode'] == 0 and exited['identity_absent'] and not exited['timeout']
    assert complete['seconds'] <= started['limit_seconds'] == phase_start['limit_seconds'] == 3600
    old_super = read(parent/'SUPERVISOR_STARTED.json')
    assert identity(old_super['identity']['pid']) != old_super['identity']
    command = phase_start['command']; assert command.count('--binding') == 1
    old_binding_path = Path(command[command.index('--binding')+1])
    assert sha(old_binding_path) == expected['binding_sha256']
    old_binding = read(old_binding_path)
    assert old_binding['schema'] == 'type-auxiliary-supervision-v1'
    assert set(old_binding['sources']) == {'worker.py','helpers.py','training.py'}
    for key,value in old_binding.items():
        if key not in ('sources','protocol'):
            assert key in binding and binding[key] == value, 'Inherited binding changed: '+key
    for name in ('helpers.py','training.py'):
        assert old_binding['sources'][name] == binding['sources'][name], 'Learning source changed: '+name
    assert old_super['source_sha256'] == dict(old_binding['sources'],
        **{'controller.py':'f7b7c20a8e8cd8a154fbb6a4c799fc915c23214aefce4122d6bf99b542375622'})
    assert old_super['binding_sha256'] == expected['binding_sha256']
    old_source = Path(old_super['source_dir'])
    for name,digest in old_super['source_sha256'].items():
        assert Path(name).name == name and sha(old_source/name) == digest
    for filename,digest in complete['source_sha256'].items():
        assert Path(filename).is_absolute() and sha(Path(filename)) == digest, filename
    # One file can be listed both as a source and as a run artifact. The CP
    # payload is streamed only once within this validation call, never loaded.
    checked = {}
    for relative,digest in complete['files'].items():
        relative = Path(relative); assert not relative.is_absolute() and '..' not in relative.parts
        path = (run/relative).resolve(); assert path.is_relative_to(run.resolve())
        if path not in checked: checked[path] = sha(path)
        assert checked[path] == digest, str(relative)
    prefix = read(run/'PREFIX_STATE.json'); cp = complete['checkpoint']
    assert prefix['checkpoint'] == cp == read(run/'checkpoints/latest.json') == read(run/'checkpoints/stage_15.json')
    assert cp['stage'] == 15 and cp['full_architecture_and_runtime'] and cp['method'] == 'SeqFT_LoRA'
    assert cp['path'] == 'checkpoints/stage_15.pt' and cp['sha256'] == expected['checkpoint_sha256']
    payload = (run/cp['path']).resolve(); assert payload.is_relative_to(run.resolve())
    assert payload.stat().st_size == cp['bytes']
    if payload not in checked: checked[payload] = sha(payload)
    assert checked[payload] == expected['checkpoint_sha256']
    assert complete['files']['before_predictions.json'] and complete['files']['acquired_predictions.json']
    assert complete['files']['PREFIX_STATE.json']
    assert not any((parent/name).exists() for name in ('SCORE_CLAIM.json','SCORES.json','VERIFIED.json'))
    # The old parent remains failed/closed, while this particular prefix had
    # already exited successfully. Keep both facts separate.
    failure = read(parent/'SUPERVISOR_FAILURE.json')
    assert failure['identity'] == old_super['identity'] and failure['no_retry'] and failure['full_quality_missing']
    assert failure['active_worker'] is None
    if failure['last_attempted_worker']:
        assert identity(failure['last_attempted_worker']['pid']) != failure['last_attempted_worker']
    return dict(run=str(run),GPU=expected['GPU'],complete_sha256=expected['complete_sha256'],
        checkpoint_sha256=expected['checkpoint_sha256'],binding_sha256=expected['binding_sha256'],
        checkpoint=cp,model_digest=prefix['model_digest'],buffers_digest=prefix['buffers_digest'],
        old_prefix_STARTED_sha256=sha(run/'prefix_STARTED.json'),old_prefix_EXIT_sha256=sha(exit_path),
        old_parent_FAILURE_sha256=sha(parent/'SUPERVISOR_FAILURE.json'),
        original_prefix_identity=started['identity'],original_controller_identity=old_super['identity'],
        original_exit0_and_identities_gone=True,original_parent_remains_failed_closed=True,
        original_source_files_verified=True,original_prefix_answers_unscored=True,
        source_binding_sha256=sha(run/'SOURCE_BINDING.json'),
        historical_cost=dict(generations=120,optimizer_updates=69,FIT_backwards=959,
                             worker_seconds=complete['seconds'],phase_seconds=exited['seconds']))


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
    parser.add_argument('--prefix-run', type=Path)
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
    prefix_check_clock = time.monotonic()
    reused = validate_reused_prefix(binding)
    prefix_check_seconds = time.monotonic() - prefix_check_clock
    assert args.gpu == reused['GPU'] == 1, 'Full prefix graph keeps original CUDA placement and RNG'
    prefix_run = Path(reused['run'])
    assert args.prefix_run is None or args.prefix_run == prefix_run
    protocol_path = Path(binding['protocol']['path'])
    assert protocol_path.is_absolute() and sha(protocol_path) == binding['protocol']['sha256']
    protocol = dict(path=str(protocol_path), sha256=sha(protocol_path))
    owner = identity(os.getpid())
    assert owner and owner['uid'] == os.getuid() and owner['pgid'] == owner['pid']
    args.output.mkdir(exist_ok=False)
    write(args.output / 'REUSED_PREFIX.json', reused)
    reused_sha256 = sha(args.output / 'REUSED_PREFIX.json')
    first_unix, first_clock = time.time(), time.monotonic()
    write(args.output / 'SUPERVISOR_STARTED.json', dict(unix=first_unix,
        identity=owner, GPU=args.gpu, source_dir=str(HERE), source_sha256=sources,
        binding_sha256=binding_sha256, protocol=protocol, order=[dict(arm=a, phase=p) for a, p in ORDER],
        phase_limits=LIMITS, cumulative_model_budget_seconds=9000,
        existing_allocator_supplied_GPU=True, no_retry=True, new_scores=0,
        reused_prefix_sha256=reused_sha256, reused_prefix_cost=reused['historical_cost'],
        prelaunch_prefix_validation_seconds=prefix_check_seconds))
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
            if phase == 'train':
                destination.mkdir(exist_ok=False)
            else:
                assert destination.is_dir() and (destination / 'train_COMPLETE.json').is_file()
            assert not (destination / (phase + '_STARTED.json')).exists()
            label, budget = arm + '_' + phase, LIMITS[phase]
            command = [str(args.python), '-u', str(HERE / 'worker.py'),
                '--root', str(args.root), '--run', str(destination), '--arm', arm,
                '--gpu', str(args.gpu), '--phase', phase, '--binding', str(args.binding)]
            if phase == 'train':
                command.extend(['--prefix-run',str(prefix_run)])
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
            counts = {'train':(80,254,18), 'eval':(105,0,0)}
            ng, nb, nu = counts[phase]
            assert complete['new_generations'] == ng
            assert complete['FIT_backwards'] == nb and complete['optimizer_updates'] == nu
            assert complete['data_binding']['binding_sha256'] == binding_sha256
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
            new_generations=555, new_scores=0, FIT_backwards=762,
            optimizer_updates=54, TEST_QA=0,
            model_phase_seconds=sum(p['seconds'] for p in phases),
            total_seconds=time.monotonic()-first_clock,
            cumulative_model_budget_seconds=9000,
            no_automatic_consumer=True, prelaunch_prefix_validation_seconds=prefix_check_seconds,
            reused_prefix_sha256=reused_sha256,
            reused_prefix_cost=reused['historical_cost']))
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
