"""Versioned pct01 capacity: adopt old workers, checkpoint all future launches.

The original queue/locks/GPU rules and SEALED exclusions remain unchanged.
--once is read-only. Pending old-version gates are preserved as history before
a new-version preflight; failed methods are never automatically retried.
"""
from pathlib import Path
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp/two_backbone_native_cl_20261017_v2'  # original shared locks and adopted workers
RUNNER = ROOT / 'exp/two_backbone_native_checkpoint_20261027'
REPORT = ROOT / 'summary/two_backbone_native_cl_20261017_v2'
HOSTS = {'new3090': 'qwen', 'new3090_onevision': 'onevision'}
ACTIVE = {'training', 'preflight', 'reload'}
ELIGIBLE = {'waiting', 'ready_reload', 'ready_training', 'ready_resume'}
SEALED = {('qwen', 'SeqFT_LoRA')} | {('onevision', m) for m in
          ('SeqFT_LoRA', 'CL_MoE', 'PASs_MoE', 'PaRSP', 'OSFT', 'SpaRTA', 'FOREVER')}
PIPELINE_SHA = '2b18ce445a01d76b2f065100752c05626bdaf800dd33f47aa8c6e75ff52f9d2d'
LEGACY_PIPELINE_SHA = 'a05c27c773dc69369cbef90f89a77c6224d1f5a5d1716c1f8576ac3ff823be17'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'


def read(path):
    return json.loads(path.read_bytes())


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        tail = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if tail[0] == 'Z':
            return None
        return dict(pid=int(pid), startticks=tail[19], uid=path.stat().st_uid,
                    pgid=int(tail[2]), argv=(path / 'cmdline').read_bytes().split(b'\0')[:-1])
    except (OSError, ValueError):
        return None


def public_identity(value):
    return {key: item for key, item in value.items() if key != 'argv'}


def runner_for(job):
    run = (ROOT / job['run_dir']).resolve()
    for source in (RUNNER, EXP):
        if run.is_relative_to((source / 'runs').resolve()): return source
    raise AssertionError('Run is outside both reviewed native runner directories')


def upgrade_pending(job):
    """Keep successful old gates as evidence; the next launch is a new preflight."""
    assert job['status'] in ELIGIBLE
    if job['status'] == 'waiting' or runner_for(job) == RUNNER: return
    assert job['status'] != 'ready_resume', 'Legacy resume requires its original runner'
    previous = dict(job)
    job.update(status='waiting', completed_stage=0,
        checkpoint_runner_upgrade=dict(unix=time.time(), from_runner=str(EXP),
            to_runner=str(RUNNER), reason='stage-checkpoint I/O version upgrade; new preflight required',
            not_a_scientific_failure_retry=True, prior_job=previous))
    job.pop('capacity_identity', None)
    job.pop('preflight_run', None)


def job_live(job):
    value = identity(job.get('pid'))
    if value is None or value['uid'] != os.getuid():
        return None
    entry = runner_for(job) / ('checkpoint_probe.py' if job['status'] == 'reload' else 'run.py')
    args = value['argv']
    if str(entry).encode() not in args or b'--run-dir' not in args:
        return None
    index = args.index(b'--run-dir')
    if index + 1 >= len(args) or Path(os.fsdecode(args[index + 1])).resolve() != (ROOT / job['run_dir']).resolve():
        return None
    saved = job.get('capacity_identity')
    assert saved is None or saved == public_identity(value), 'Worker identity changed'
    return value


def load_pipeline():
    assert hashlib.sha256((RUNNER / 'pipeline.py').read_bytes()).hexdigest() == PIPELINE_SHA
    assert hashlib.sha256((EXP / 'pipeline.py').read_bytes()).hexdigest() == LEGACY_PIPELINE_SHA
    previous = {key: os.environ.get(key) for key in ('LLMCL_HOST', 'LLMCL_BACKBONE')}
    os.environ.update(LLMCL_HOST='new3090', LLMCL_BACKBONE='qwen')
    try:
        spec = importlib.util.spec_from_file_location('checkpoint_native_capacity_pipeline', RUNNER / 'pipeline.py')
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    finally:
        for key, item in previous.items():
            if item is None: os.environ.pop(key, None)
            else: os.environ[key] = item
    return value


def policy():
    value = read(RUNNER / 'execution_policy.json')
    assert value['training_enabled'] and value['task_count'] == 19 and value['seed'] == 42
    assert value['conditions']['pct01']['manifest_sha256'] == MANIFEST_SHA
    for host, backbone in HOSTS.items():
        item = value['hosts'][host]
        assert item['backbone'] == backbone and item['reserved_gpus'] == [0]
        assert item['conditions']['pct01'] == [1, 2, 3] and item['conditions'].get('full', []) == []
    return value


def load_states():
    result = {}
    for host, backbone in HOSTS.items():
        path = REPORT / (host + '_run_status.json')
        if path.exists():
            result[host] = read(path)
        else:
            assert host == 'new3090_onevision'
            result[host] = dict(experiment_id=EXP.name, host=host, backbone=backbone,
                jobs={'pct01:SMoLoRA': dict(condition='pct01', method='SMoLoRA', host=host, status='waiting')})
    return result


def managed(host, job):
    return (job.get('host') == host and job.get('condition') == 'pct01'
            and job.get('method') != 'FOREVER'
            and (HOSTS[host], job['method']) not in SEALED
            and (host != 'new3090_onevision' or job['method'] == 'SMoLoRA'))


def valid_proof(pipe, run, name, preflight=False):
    path = run / name
    if not path.exists(): return False
    proof = read(path)
    source = runner_for(dict(run_dir=str(run.relative_to(ROOT))))
    expected = pipe.sources() if source == RUNNER else {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in EXP.glob('*.py')}
    return bool(proof.get('passed') and proof.get('execution_source_sha256') == expected
                and proof.get('manifest_sha256') == MANIFEST_SHA
                and (not preflight or proof.get('task_ids') == list(range(1, 20))))


def reconcile(pipe, host, key, job, processes):
    live = job_live(job)
    proc = processes.get((host, key))
    if live is not None and (proc is None or proc.poll() is None):
        job['capacity_identity'] = public_identity(live)
        path = ROOT / job['run_dir'] / 'run_status.json'
        if path.exists():
            job.update({k: v for k, v in read(path).items() if k in ('completed_stage', 'active_stage')})
        return True
    run = ROOT / job['run_dir']
    code = None if proc is None else proc.poll()
    job['capacity_last_exit_code'] = code  # adopted children have no retrievable wait status
    if job['status'] == 'preflight':
        ok = valid_proof(pipe, run, 'gpu_preflight.json', True)
        job['status'] = 'ready_reload' if ok and code in (None, 0) else 'failed_preflight'
    elif job['status'] == 'reload':
        ok = valid_proof(pipe, run, 'checkpoint_reload_proof.json')
        job['status'] = 'ready_training' if ok and code in (None, 0) else 'failed_checkpoint_reload'
        if job['status'] == 'ready_training': job['preflight_run'] = job['run_dir']
    else:
        path = run / 'run_status.json'; status = read(path) if path.exists() else {}
        if status.get('status') == 'completed' and status.get('completed_stage') == 19 and code in (None, 0):
            job.update(status='completed', completed_stage=19)
        elif status.get('status') == 'paused_checkpoint' and code in (None, 0):
            assert runner_for(job) == RUNNER
            pause = pipe.paused_checkpoint(run)
            job.update(status='paused_checkpoint', completed_stage=pause['completed_stage'],
                pause_receipt=pause['pause_receipt'], pause_receipt_sha256=pause['pause_receipt_sha256'])
        else:
            job['status'] = 'failed_training'  # no blind method retry or unsupported --resume
    if job['status'].startswith('failed'):
        job['failure_log'] = str((run / 'supervisor.log').relative_to(ROOT))
    processes.pop((host, key), None)
    job.pop('capacity_identity', None)
    return False


def prepare_ready(pipe, job):
    upgrade_pending(job)
    if job['status'] == 'ready_resume':
        assert runner_for(job) == RUNNER and pipe.resume_requested(ROOT / job['run_dir'])
        return
    if job['status'] == 'ready_reload':
        run = ROOT / job['run_dir']
        if not valid_proof(pipe, run, 'gpu_preflight.json', True):
            job['status'] = 'failed_preflight'; return
        # A completed probe deletes its temporary .pt; consume its existing
        # proof instead of blindly trying to reload the deleted checkpoint.
        if (run / 'checkpoint_reload_proof.json').exists():
            job['status'] = 'ready_training' if valid_proof(pipe, run, 'checkpoint_reload_proof.json') else 'failed_checkpoint_reload'
            if job['status'] == 'ready_training': job['preflight_run'] = job['run_dir']


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--once', action='store_true', help='Read-only metadata inspection; never launch/write queue')
    mode.add_argument('--watch', action='store_true', help='Run the CPU dispatcher')
    parser.add_argument('--expect-live', action='append', default=[], metavar='PID:STARTTICKS')
    args = parser.parse_args(); pipe = load_pipeline()
    if args.once:
        states = load_states()
        print(json.dumps(dict(inspection_only=True, no_launch=True,
            queues={host: {k: {n: j.get(n) for n in ('condition', 'method', 'status', 'pid', 'gpu', 'run_dir')}
                           for k, j in state['jobs'].items() if managed(host, j)} for host, state in states.items()},
            reserved_GPU=0, FOREVER_untouched=True)))
        return
    policy()  # root applies and records the device-only JSON amendment separately
    locks = []
    for path in [REPORT / 'baseline_pct01_capacity.lock', REPORT / 'hardware_queue.lock'] + [EXP / ('.pipeline_' + h + '.lock') for h in HOSTS]:
        handle = path.open('a'); fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB); locks.append(handle)
    for expected in args.expect_live:
        pid, start = expected.split(':'); value = identity(int(pid))
        assert value is not None and value['startticks'] == start and value['uid'] == os.getuid(), expected
    states = load_states(); processes = {}
    while True:
        rules = policy(); occupied = set(); active = {}
        for host, state in states.items():
            for key, job in state['jobs'].items():
                if not managed(host, job): continue
                if job['status'] in ACTIVE and reconcile(pipe, host, key, job, processes):
                    occupied.add(job['gpu']); active[host + ':' + key] = job['capacity_identity']
                if job['status'] == 'paused_checkpoint':
                    assert runner_for(job) == RUNNER
                    if job_live(job) is None and pipe.resume_requested(ROOT / job['run_dir']):
                        job['status'] = 'ready_resume'
                if job['status'] in ELIGIBLE: prepare_ready(pipe, job)
        # Fast existing reload/training admissions first; native BF16 does not
        # wait for every Qwen method, and precedes fresh Qwen preflights.
        candidates = [(host, key, job) for host, state in states.items() for key, job in state['jobs'].items()
                      if managed(host, job) and job['status'] in ELIGIBLE]
        rank = {'ready_reload': 0, 'ready_training': 1, 'ready_resume': 1, 'waiting': 2}
        candidates.sort(key=lambda item: (rank[item[2]['status']], item[0] != 'new3090_onevision', item[1]))
        for gpu in (1, 2, 3):
            if gpu in occupied or not pipe.idle(gpu, 16000): continue
            for item in list(candidates):
                host, key, job = item
                # FOREVER's unchanged independent waiter owns its existingGPU2
                # opportunity once Qwen has no pending admissions. Avoid racing
                # that waiter with the additional OneVision lane.
                qwen_pending = any(managed('new3090', j) and j['status'] in ELIGIBLE
                                   for j in states['new3090']['jobs'].values())
                if gpu == 2 and host == 'new3090_onevision' and not qwen_pending: continue
                if job['status'] == 'ready_resume':
                    assert runner_for(job) == RUNNER and pipe.resume_requested(ROOT / job['run_dir']), 'Resume needs this paused checkpoint and explicit authorization'
                phase = {'waiting': 'preflight', 'ready_reload': 'reload', 'ready_training': 'training', 'ready_resume': 'training'}[job['status']]
                pipe.HOST, pipe.BACKBONE = host, HOSTS[host]
                child = pipe.launch(job, phase, gpu, resume=job['status'] == 'ready_resume')
                processes[host, key] = child
                value = job_live(job)
                if value is not None: job['capacity_identity'] = public_identity(value)
                occupied.add(gpu); candidates.remove(item); break
        for host, state in states.items():
            state.update(updated_at=datetime.now(timezone.utc).isoformat(), reserved_gpus=[0], phase='running_or_waiting',
                         capacity_wrapper='analysis/baseline_pct01_checkpoint_capacity_20261027.py', capacity_pct01_only=True)
            pipe.write(REPORT / (host + '_run_status.json'), state)
        pipe.write(REPORT / 'run_status.json', states['new3090'])
        pipe.write(REPORT / 'baseline_pct01_capacity_status.json', dict(unix=time.time(), CPU_pid=os.getpid(), active=active,
            reserved_GPU_not_assigned=0, pct01_only=True, FOREVER_untouched=True,
            pending=[dict(host=h, key=k, status=j['status']) for h, k, j in candidates], no_GPU_signals=True))
        time.sleep(15)


if __name__ == '__main__':
    main()
