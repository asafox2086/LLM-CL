"""Reuse the pct10 coordinator; route only FOREVER to its independent runner.

CPU handoff only after the prior coordinator stops. The ordinary allocator
owns the gate slot throughout preflight + reload; there is no side allocator.
Frozen dispatch.py, all model sources and native jobs are never rewritten.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(os.environ['LLMCL_ROOT']).resolve()
CONTROLLER = Path(__file__).resolve().parent
HERE = ROOT / 'exp/two_backbone_pct10_forever_dynamic_20261025'
REPORT = ROOT / 'summary/two_backbone_pct10_20261025'
READY_PATH = REPORT / 'FOREVER_DYNAMIC_DISPATCH_READY.json'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exclusive(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def prepare():
    ready = read(READY_PATH)
    original = ROOT / 'exp/two_backbone_pct10_20261025/dispatch.py'
    assert sha(original) == ready['original_dispatch_sha256']
    assert {p.name: sha(p) for p in HERE.glob('*.py')} == ready['dynamic_sources']
    assert {name: sha(CONTROLLER / name) for name in
            ('dispatch_dynamic.py', 'forever_gate_supervisor.py')} == ready['supervisor_sources']
    assert sha(HERE / 'execution_policy.json') == ready['dynamic_policy_sha256']
    assert sha(HERE / 'BUILD_RECEIPT.json') == ready['build_receipt_sha256']
    spec = importlib.util.spec_from_file_location('pct10_original_coordinator', original)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Original coordinator is stdlib-only.
    assert module.ROOT == ROOT
    module.FOLDERS = dict(module.FOLDERS, forever=HERE.name)
    # Its main launches Path(__file__): every new phase must re-enter this patch.
    module.__file__ = str(Path(__file__).resolve())
    return module, ready


def handoff(module, ready, machine):
    path = REPORT / ('queue_' + machine + '.json')
    record = REPORT / ('FOREVER_DYNAMIC_HANDOFF_' + machine + '.json')
    if record.exists():
        old = read(record)
        assert old['original_dispatch_sha256'] == ready['original_dispatch_sha256']
        assert old['dynamic_sources'] == ready['dynamic_sources']
        assert old['ready_sha256'] == sha(READY_PATH)
        return  # No permanent old-queue hash guard and no repeated unfreezing.
    assert machine == 'old', 'NEW requires its own pct01 allocation handoff first'
    inputs = REPORT / ('FOREVER_DYNAMIC_HANDOFF_INPUT_' + machine + '.json')
    handshake = read(inputs)
    assert handshake['ready_sha256'] == sha(READY_PATH)
    prior_controller = handshake['prior_controller']
    assert module.identity(prior_controller['pid']) != prior_controller, 'Stop only the old CPU coordinator first'
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == handshake['initial_queue_sha256']
    state = json.loads(raw)
    changed = {}
    protected = json.dumps({key: job for key, job in state['jobs'].items()
                            if job['method'] != 'FOREVER'}, sort_keys=True)
    for backbone in ('onevision', 'qwen'):
        key = backbone + ':FOREVER'
        job = state['jobs'][key]
        assert job['backbone'] == backbone and job['condition'] == 'pct10'
        assert job['status'] == 'waiting_forever_pct10_scheduler_review'
        assert not job.get('worker') and not job.get('pid'), 'Never adopt an unreviewed FOREVER model'
        changed[key] = dict(job)
        job.update(status='waiting', dynamic_runner=HERE.name,
                   deferred_status_before_dynamic_handoff=changed[key]['status'])
    assert protected == json.dumps({key: job for key, job in state['jobs'].items()
                                    if job['method'] != 'FOREVER'}, sort_keys=True)
    backup = REPORT / ('FOREVER_DYNAMIC_PRIOR_QUEUE_' + machine + '.json')
    with backup.open('xb') as stream:
        stream.write(raw)
    module.write(path, state)
    exclusive(record, dict(unix=time.time(), machine=machine,
        prior_controller=prior_controller, new_controller=module.identity(os.getpid()),
        prior_queue_sha256=sha(backup), adopted_queue_sha256=sha(path),
        changed_only_two_FOREVER=changed, native_job_records_preserved=True,
        GPU_signals_sent=0, original_dispatch_sha256=ready['original_dispatch_sha256'],
        ready_sha256=sha(READY_PATH), handoff_input_sha256=sha(inputs),
        dynamic_sources=ready['dynamic_sources'],
        first_gate_pending=True, formal_training_must_start_fresh=True))


def forever_phase(module, original_phase, request_path):
    request = read(request_path)
    job = request['job']
    if job['method'] != 'FOREVER' or request['phase'] == 'training':
        return original_phase(request_path)
    phase = request['phase']
    assert phase in ('preflight', 'reload')
    pipe = module.pipe('forever', request['machine'], job['backbone'])
    assert pipe.sources() == request['sources']
    assert request['manifest_sha256'] == read(HERE / 'execution_policy.json')['conditions']['pct10']['manifest_sha256']
    assert pipe.idle(request['gpu'], 8500 if request['machine'] == 'old' else 16000)
    if phase == 'reload':
        # The gate's second actual model process already did the independent
        # checkpoint reload within the shared 2400 seconds. Consume, do not redo.
        run = ROOT / job['run_dir']
        verified = read(run / 'GATE_VERIFIED.json')
        assert verified['passed'] and verified['sources'] == request['sources']
        assert verified['backbone'] == job['backbone']
        proof = read(run / 'checkpoint_reload_proof.json')
        assert proof['passed'] and proof['fresh_process'] and proof['finite_backward']
        assert proof['execution_source_sha256'] == request['sources']
        who = module.identity(os.getpid())
        module.write(request['status_path'], dict(unix=time.time(), job=job,
            wrapper=who, worker=who, phase='reload', run_dir=job['run_dir'],
            sources=request['sources'], GPU=request['gpu'],
            metadata_only_consumption_of_completed_gate_reload=True))
        module.write(request['exit_path'], dict(unix=time.time(), returncode=0,
            worker=who, wrapper=who, phase='reload', run_dir=job['run_dir'],
            gate_verified_sha256=sha(run / 'GATE_VERIFIED.json'), new_models=0))
        return
    run = HERE / 'runs' / job['backbone'] / 'pct10/FOREVER' / ('preflight_' + str(time.time_ns()))
    run.mkdir(parents=True, exist_ok=False)
    gate_request = Path(request_path).with_suffix('.gate.json')
    value = dict(request, run_dir=str(run.relative_to(ROOT)))
    exclusive(gate_request, value)
    env = dict(os.environ, LLMCL_ROOT=str(ROOT), PYTHONUNBUFFERED='1')
    with (run / 'gate_supervisor.log').open('a') as log:
        process = subprocess.Popen([sys.executable, '-u', str(CONTROLLER / 'forever_gate_supervisor.py'),
            '--request', str(gate_request)], cwd=ROOT, env=env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        child = module.identity(process.pid)
        assert child and child['uid'] == os.getuid() and child['pgid'] == child['pid']
        job.update(status='preflight', pid=process.pid, gpu=request['gpu'],
                   host=request['machine'] + '_pct10_' + job['backbone'],
                   run_dir=str(run.relative_to(ROOT)), completed_stage=0)
        module.write(request['status_path'], dict(unix=time.time(), job=job,
            wrapper=module.identity(os.getpid()), worker=child, phase='preflight',
            run_dir=job['run_dir'], sources=request['sources'], GPU=request['gpu']))
        code = process.wait()
    if code == 0:
        gate = read(run / 'GATE_VERIFIED.json')
        assert gate['passed'] and gate['sources'] == request['sources']
    module.write(request['exit_path'], dict(unix=time.time(), returncode=code,
        worker=child, wrapper=module.identity(os.getpid()), phase='preflight',
        run_dir=job['run_dir'], shared_preflight_reload_gate=True))
    if code:
        raise SystemExit(code)


def main():
    module, ready = prepare()
    original_phase = module.phase
    module.phase = lambda request: forever_phase(module, original_phase, request)
    if '--phase-request' not in sys.argv:
        machine = 'new' if '--machine' in sys.argv and sys.argv[sys.argv.index('--machine') + 1] == 'new' else 'old'
        handoff(module, ready, machine)
    module.main()


if __name__ == '__main__':
    main()
