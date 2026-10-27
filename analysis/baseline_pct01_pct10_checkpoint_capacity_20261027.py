"""NEW's existing capacity owner, with stage-checkpoint routes for two pct10 SMo jobs.

--checkpoint-admission ABS --once validates metadata only; --watch runs the same
pct01 allocator and locks. No runner, policy, active pct01 job or slot is edited.
Copied build targets are matched by basename; actual destination bytes are bound.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LEGACY = ROOT/'analysis/baseline_pct01_pct10_capacity_20261026.py'
CAPACITY = ROOT/'analysis/baseline_pct01_checkpoint_capacity_20261027.py'
LEGACY_CAPACITY = ROOT/'analysis/baseline_pct01_capacity_20261025.py'
PCT01_FRAME = ROOT/'exp/two_backbone_native_checkpoint_20261027'
FRAME = ROOT/'exp/two_backbone_pct10_native_checkpoint_20261027'
DISPATCH = ROOT/'exp/stage_checkpoint_dispatch_20261027/dispatch.py'
REPORT = ROOT/'summary/two_backbone_pct10_20261025'
LEGACY_SHA = '28614141a9f75bb4fdc5c2762993004fe0792a6968776b2207e9fd5d69d732c9'
LEGACY_CAPACITY_SHA = 'c775274d0ccd3f0e342f224ee3a16923f5f826dd1237d2fd1565052fee9575d8'
CAPACITY_SHA = 'c463d1f641e30d8ba2a58d9d3f94e2f3b8d097936fcaa405fc572414a167b1e8'
DISPATCH_SHA = 'a60ac9d23bf4b3e047f0475dddcbecd4f6cacea2ce81b79644403f2785c3d625'
MANIFEST = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
KEYS = ('onevision:SMoLoRA', 'qwen:SMoLoRA')
ACTIVE = ('launching', 'preflight', 'reload', 'training')
ELIGIBLE = ('waiting', 'ready_reload', 'ready_training', 'ready_resume')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def validate(admission_path):
    assert admission_path.is_absolute()
    value = read(admission_path)
    assert Path(value['root']).resolve() == ROOT
    assert value['wrapper_sha256'] == sha(Path(__file__))
    assert value['legacy_wrapper_sha256'] == sha(LEGACY) == LEGACY_SHA
    assert value['legacy_capacity_sha256'] == sha(LEGACY_CAPACITY) == LEGACY_CAPACITY_SHA
    assert value['capacity_sha256'] == sha(CAPACITY) == CAPACITY_SHA
    assert value['jobs'] == list(KEYS) and value['allowed_pct10_gpus'] == [1, 3]
    assert value['reserved_gpu'] == 0 and value['pct01_priority'] is True
    assert value['manifest_sha256'] == MANIFEST
    ref = value['dispatch']
    assert Path(ref['path']).resolve() == DISPATCH
    assert ref['sha256'] == sha(DISPATCH) == DISPATCH_SHA
    build_ref = value['build']; build_path = Path(build_ref['path'])
    assert build_path.is_absolute() and sha(build_path) == build_ref['sha256']
    build = read(build_path); assert build['complete'] and build['inputs']['manifest_sha256'] == MANIFEST
    targets = [r for r in build['runners'] if Path(r['target']).name == FRAME.name]
    assert len(targets) == 1
    native = value['native_runner']
    assert Path(native['path']).resolve() == FRAME
    sources = {p.name: sha(p) for p in FRAME.glob('*.py')}
    assert sources == native['sources'] == targets[0]['destination_py_sha256']
    assert sha(FRAME/'execution_policy.json') == native['policy_sha256'] == targets[0]['destination_sha256']['execution_policy.json']
    assert read(FRAME/'BUILD_RECEIPT.json') == targets[0]
    policy = read(FRAME/'execution_policy.json')
    assert policy['training_enabled'] and policy['seed'] == 42 and policy['task_count'] == 19
    assert policy['conditions']['pct10']['manifest_sha256'] == MANIFEST
    assert sha(ROOT/policy['conditions']['pct10']['manifest']) == MANIFEST
    for backbone in ('onevision', 'qwen'):
        host = policy['hosts']['new_pct10_'+backbone]
        assert host['reserved_gpus'] == [0] and host['conditions']['pct10'] == [1, 2, 3]
    pct01 = value['pct01_runner']
    assert Path(pct01['path']).resolve() == PCT01_FRAME
    assert {p.name: sha(p) for p in PCT01_FRAME.glob('*.py')} == pct01['sources']
    assert sha(PCT01_FRAME/'execution_policy.json') == pct01['policy_sha256']
    pct01_policy = read(PCT01_FRAME/'execution_policy.json')
    assert pct01_policy['conditions']['pct01']['manifest_sha256'] == '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
    for host_name in ('new3090', 'new3090_onevision'):
        host = pct01_policy['hosts'][host_name]
        assert host['reserved_gpus'] == [0] and host['conditions']['pct01'] == [1, 2, 3]
        assert host['conditions'].get('full', []) == []
    return value, sources, targets[0]['target']


def addition_class(legacy, admission_path, admission, sources, copied_target):
    # Inherited save/held/patch use these globals. Only this private imported
    # module is rebound; its original source and the pct01 module stay unchanged.
    legacy.ROOT, legacy.FRAME, legacy.REPORT = ROOT, FRAME, REPORT
    legacy.DISPATCH, legacy.ELIGIBLE = DISPATCH, ELIGIBLE

    class Addition(legacy.Addition):
        def __init__(self, cap, inspection):
            self.cap, self.inspection, self.states01 = cap, inspection, None
            self.processes, self.locks, self.sources = {}, [], sources
            environment = {k: os.environ.get(k) for k in ('LLMCL_ROOT', 'LLMCL_HOST', 'LLMCL_BACKBONE')}
            os.environ['LLMCL_ROOT'] = str(ROOT)
            try:
                self.dispatch = load('checkpoint_pct10_phase_dispatch', DISPATCH)
                self.pipes = {b: self.dispatch.pipe('native', 'new', b) for b in ('onevision', 'qwen')}
                assert all(p.sources() == sources for p in self.pipes.values())
            finally:
                for key, value in environment.items():
                    if value is None: os.environ.pop(key, None)
                    else: os.environ[key] = value
            self.path = REPORT/'queue_new.json'
            self.state = read(self.path) if self.path.exists() else dict(created=time.time(), machine='new', jobs={
                k: dict(key=k, condition='pct10', method='SMoLoRA', backbone=k.split(':')[0], status='waiting') for k in KEYS})
            assert set(self.state['jobs']) == set(KEYS)
            for key, job in self.state['jobs'].items():
                assert job['key'] == key and job['method'] == 'SMoLoRA' and job['condition'] == 'pct10'
                assert job['backbone'] == key.split(':')[0]
                if job['status'] in ACTIVE + ('ready_reload', 'ready_training', 'ready_resume', 'paused_checkpoint'):
                    assert (ROOT/job['run_dir']).resolve().is_relative_to((FRAME/'runs').resolve()), 'Legacy pct10 phase must be reconciled before this handoff'
            self.state.update(checkpoint_admission_sha256=sha(admission_path),
                checkpoint_runner=str(FRAME), copied_build_target=copied_target)

        def reconcile(self):
            for key, job in self.state['jobs'].items():
                if job['status'] not in ACTIVE: continue
                status, end = Path(job['phase_status']), Path(job['phase_exit'])
                if status.exists():
                    info = read(status)
                    assert info['wrapper'] == job['wrapper'] and info['GPU'] == job['gpu'] and info['phase'] == job['phase']
                    job.update(status=info['phase'], pid=info['worker']['pid'], worker=info['worker'], run_dir=info['run_dir'])
                proc = self.processes.get(key)
                if legacy.identity(job['wrapper']['pid']) == job['wrapper'] and (proc is None or proc.poll() is None): continue
                if not end.exists(): job['status'] = 'orphaned_missing_exit'; continue
                exit = read(end)
                assert exit['wrapper'] == job['wrapper'] and exit['phase'] == job['phase']
                assert job.get('worker') == exit['worker'] and job['run_dir'] == exit['run_dir']
                if legacy.identity(exit['worker']['pid']) is not None: job['status'] = 'orphaned_live'; continue
                if proc is not None: assert proc.poll() == exit['returncode']
                job['last_exit'] = exit; self.processes.pop(key, None)
                run = ROOT/exit['run_dir']; phase = exit['phase']
                if exit['returncode'] != 0: job['status'] = 'failed_'+phase; continue
                if phase in ('preflight', 'reload'):
                    proof = read(run/('gpu_preflight.json' if phase == 'preflight' else 'checkpoint_reload_proof.json'))
                    assert proof['passed'] and proof['manifest_sha256'] == MANIFEST and proof['execution_source_sha256'] == self.sources
                    if phase == 'preflight': assert proof['task_ids'] == list(range(1, 20))
                    job['status'] = 'ready_reload' if phase == 'preflight' else 'ready_training'
                    if phase == 'reload': job['preflight_run'] = job['run_dir']
                else:
                    status = read(run/'run_status.json')
                    if status['status'] == 'paused_checkpoint':
                        pause = self.dispatch.paused_checkpoint(run, job, self.pipes[job['backbone']])
                        job.update(status='paused_checkpoint', completed_stage=pause['completed_stage'],
                            pause_receipt=pause['pause_receipt'], pause_receipt_sha256=pause['pause_receipt_sha256'])
                    else:
                        assert status['status'] == 'completed' and status['completed_stage'] == 19
                        job.update(status='completed_unverified', completed_stage=19)
            for job in self.state['jobs'].values():
                if job['status'] != 'paused_checkpoint': continue
                if any(job.get(k) and legacy.identity(job[k]['pid']) == job[k] for k in ('worker', 'wrapper')): continue
                if self.dispatch.resume_requested(ROOT/job['run_dir'], job, self.pipes[job['backbone']]):
                    job['status'] = 'ready_resume'
            self.save()

        def offer(self, gpu, minimum, original_idle):
            if gpu in self.held(): return False
            free = original_idle(gpu, minimum)
            if not free or gpu not in (1, 3): return free
            assert self.states01 is not None
            if any(self.cap.managed(host, job) and job['status'] in self.cap.ELIGIBLE
                   for host, state in self.states01.items() for job in state['jobs'].values()): return True
            jobs = [(k, j) for k, j in self.state['jobs'].items() if j['status'] in ELIGIBLE]
            if not jobs or shutil.disk_usage(ROOT).free < 12*1024**3: return free
            jobs.sort(key=lambda x: ({'ready_reload': 0, 'ready_training': 1, 'ready_resume': 1, 'waiting': 2}[x[1]['status']], x[0]))
            key, job = jobs[0]; resume = job['status'] == 'ready_resume'
            phase = {'waiting': 'preflight', 'ready_reload': 'reload', 'ready_training': 'training', 'ready_resume': 'training'}[job['status']]
            digest = None
            if resume:
                assert self.dispatch.resume_requested(ROOT/job['run_dir'], job, self.pipes[job['backbone']])
                digest = self.dispatch.paused_checkpoint(ROOT/job['run_dir'], job, self.pipes[job['backbone']])['pause_receipt_sha256']
            request = REPORT/'requests'/(key.replace(':', '_')+'_'+phase+'_'+str(time.time_ns())+'.json')
            status, end = request.with_suffix('.status.json'), request.with_suffix('.exit.json')
            self.dispatch.write(request, dict(job=dict(job), machine='new', gpu=gpu, phase=phase,
                manifest_sha256=MANIFEST, sources=self.sources, status_path=str(status), exit_path=str(end),
                resume=resume, resume_pause_receipt_sha256=digest))
            with request.with_suffix('.wrapper.log').open('a') as log:
                child = subprocess.Popen([sys.executable, '-u', str(DISPATCH), '--phase-request', str(request)],
                    env=dict(os.environ, LLMCL_ROOT=str(ROOT)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            who = legacy.identity(child.pid)
            assert who and who['uid'] == os.getuid() and who['pgid'] == who['pid']
            job.update(status='launching', wrapper=who, gpu=gpu, phase=phase,
                phase_status=str(status), phase_exit=str(end), request=str(request))
            self.processes[key] = child; self.save()
            return False
    return Addition


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--checkpoint-admission', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    admission, sources, copied_target = validate(args.checkpoint_admission)
    legacy = load('unchanged_pct01_pct10_capacity', LEGACY)
    cap = legacy.module('unchanged_pct01_capacity', CAPACITY)
    addition = addition_class(legacy, args.checkpoint_admission, admission, sources, copied_target)(cap, '--once' in remaining)
    if addition.inspection:
        print(json.dumps(dict(inspection_only=True, no_launch=True, jobs=addition.state['jobs'],
            admission_sha256=sha(args.checkpoint_admission), copied_build_target=copied_target,
            allowed_GPU=[1, 3], checkpoint_sources=sources)))
    addition.patch()
    sys.argv = [str(Path(__file__).resolve()), *remaining]
    cap.main()


if __name__ == '__main__': main()
