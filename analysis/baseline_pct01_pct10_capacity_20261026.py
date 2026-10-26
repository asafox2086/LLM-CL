"""One NEW capacity owner: unchanged pct01 first, then only two pct10 SMo jobs.

This is a CPU scheduling wrapper, not a second allocator. It wraps the original
capacity main/locks/idle opportunity and calls the unchanged pct10 phase entry.
GPU0 is never allocated; pct10 uses only GPU1/3, leaving FOREVER's GPU2 rule.
--once is read-only. --watch keeps the original capacity CLI and live adoption.
"""
from pathlib import Path
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CAPACITY = ROOT/'analysis/baseline_pct01_capacity_20261025.py'
DISPATCH = ROOT/'exp/two_backbone_pct10_20261025/dispatch.py'
FRAME = ROOT/'exp/two_backbone_pct10_native_20261025'
REPORT = ROOT/'summary/two_backbone_pct10_20261025'
BUILD = REPORT/'RUNNER_BUILD_RECEIPT.json'
MANIFEST = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
CAPACITY_SHA = 'c775274d0ccd3f0e342f224ee3a16923f5f826dd1237d2fd1565052fee9575d8'
DISPATCH_SHA = '0d7a4c849b3f22978b70d6cb77e14db892f97b84ffbea594edc817b5dca4146f'
BUILD_SHA = '9d3a2c76b0229a25be6ecbf2e17b5ab732d6dd0ac451797cec8c518786e1a573'
POLICY_SHA = '8dd765f08e586b3fee00274d5122a9ccf87b9327843bdfe602340884ddf3487b'
KEYS = ('onevision:SMoLoRA', 'qwen:SMoLoRA')
ACTIVE = ('launching', 'preflight', 'reload', 'training')
ELIGIBLE = ('waiting', 'ready_reload', 'ready_training')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def identity(pid):
    try:
        p = Path('/proc')/str(pid); s = (p/'stat').read_text().rsplit(') ', 1)[1].split()
        if s[0] == 'Z': return None
        return dict(pid=int(pid), startticks=s[19], uid=p.stat().st_uid, pgid=int(s[2]))
    except (OSError, ValueError):
        return None


class Addition:
    def __init__(self, cap, inspection):
        self.cap = cap; self.inspection = inspection; self.states01 = None
        self.processes = {}; self.locks = []
        assert sha(DISPATCH) == DISPATCH_SHA and sha(BUILD) == BUILD_SHA
        assert sha(FRAME/'execution_policy.json') == POLICY_SHA
        built = read(BUILD)
        target = next(x for x in built['runners'] if Path(x['target']) == FRAME)
        self.sources = {p.name: sha(p) for p in FRAME.glob('*.py')}
        expected = {k: v for k, v in target['destination_sha256'].items() if k.endswith('.py')}
        assert self.sources == expected
        policy = read(FRAME/'execution_policy.json')
        assert policy['training_enabled'] and policy['seed'] == 42 and policy['task_count'] == 19
        assert policy['conditions']['pct10']['manifest_sha256'] == MANIFEST
        for b in ('onevision', 'qwen'):
            p = policy['hosts']['new_pct10_'+b]
            assert p['reserved_gpus'] == [0] and p['conditions']['pct10'] == [1, 2, 3]
        oldenv = {k: os.environ.get(k) for k in ('LLMCL_ROOT', 'LLMCL_HOST', 'LLMCL_BACKBONE')}
        os.environ['LLMCL_ROOT'] = str(ROOT)
        try:
            self.dispatch = module('bound_pct10_phase_dispatch', DISPATCH)
            self.pipes = {b: self.dispatch.pipe('native', 'new', b) for b in ('onevision', 'qwen')}
        finally:
            for k, v in oldenv.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
        self.path = REPORT/'queue_new.json'
        self.state = read(self.path) if self.path.exists() else dict(created=time.time(), machine='new', jobs={
            k: dict(key=k, condition='pct10', method='SMoLoRA', backbone=k.split(':')[0], status='waiting') for k in KEYS})
        assert set(self.state['jobs']) == set(KEYS), 'Only the two pct10 SMoLoRA jobs are admitted'
        for k, j in self.state['jobs'].items():
            assert j['key'] == k and j['condition'] == 'pct10' and j['method'] == 'SMoLoRA'
            assert j['backbone'] == k.split(':')[0]

    def save(self):
        self.state.update(updated=time.time(), controller=identity(os.getpid()), manifest_sha256=MANIFEST,
            reserved_new_GPU=0, allowed_pct10_GPUs=[1, 3], pct01_priority=True,
            all_training_from_pretrained=True, full_test_QA=21454,
            original_pct01_sources_and_jobs_untouched=True)
        self.dispatch.write(self.path, self.state)

    def held(self):
        return {j['gpu'] for j in self.state['jobs'].values()
                if j['status'] in ACTIVE or j['status'].startswith('orphaned')}

    def reconcile(self):
        for key, j in self.state['jobs'].items():
            if j['status'] not in ACTIVE: continue
            status = Path(j['phase_status']); end = Path(j['phase_exit'])
            if status.exists():
                x = read(status)
                assert x['wrapper'] == j['wrapper'] and x['GPU'] == j['gpu'] and x['phase'] == j['phase']
                j.update(status=x['phase'], pid=x['worker']['pid'], worker=x['worker'], run_dir=x['run_dir'])
            proc = self.processes.get(key)
            if identity(j['wrapper']['pid']) == j['wrapper'] and (proc is None or proc.poll() is None): continue
            if not end.exists():
                # No exit proof means no automatic release/retry, even if the
                # wrapper disappeared before publishing its child identity.
                j['status'] = 'orphaned_missing_exit'; continue
            ex = read(end)
            assert ex['wrapper'] == j['wrapper'] and ex['phase'] == j['phase']
            assert j.get('worker') == ex['worker'] and j['run_dir'] == ex['run_dir']
            if identity(ex['worker']['pid']) is not None:
                j['status'] = 'orphaned_live'; continue
            if proc is not None: assert proc.poll() == ex['returncode']
            j['last_exit'] = ex; self.processes.pop(key, None)
            run = ROOT/ex['run_dir']; phase = ex['phase']
            if ex['returncode'] != 0:
                j['status'] = 'failed_'+phase; continue
            if phase in ('preflight', 'reload'):
                name = 'gpu_preflight.json' if phase == 'preflight' else 'checkpoint_reload_proof.json'
                p = read(run/name)
                assert p['passed'] and p['manifest_sha256'] == MANIFEST and p['execution_source_sha256'] == self.sources
                if phase == 'preflight': assert p['task_ids'] == list(range(1, 20))
                j['status'] = 'ready_reload' if phase == 'preflight' else 'ready_training'
                if phase == 'reload': j['preflight_run'] = j['run_dir']
            else:
                p = read(run/'run_status.json')
                assert p['status'] == 'completed' and p['completed_stage'] == 19
                j.update(status='completed_unverified', completed_stage=19)
        self.save()

    def offer(self, gpu, minimum, original_idle):
        if gpu in self.held(): return False
        free = original_idle(gpu, minimum)
        if not free or gpu not in (1, 3): return free
        # The original loop has reconciled/validated pct01 admissions already.
        # Its eligible candidates retain first use of every available slot.
        if any(self.cap.managed(h, j) and j['status'] in self.cap.ELIGIBLE
               for h, s in self.states01.items() for j in s['jobs'].values()): return True
        jobs = [(k, j) for k, j in self.state['jobs'].items() if j['status'] in ELIGIBLE]
        if not jobs or shutil.disk_usage(ROOT).free < 12*1024**3: return free
        jobs.sort(key=lambda x: ({'ready_reload': 0, 'ready_training': 1, 'waiting': 2}[x[1]['status']], x[0]))
        key, job = jobs[0]; phase = {'waiting': 'preflight', 'ready_reload': 'reload', 'ready_training': 'training'}[job['status']]
        request = REPORT/'requests'/(key.replace(':', '_')+'_'+phase+'_'+str(time.time_ns())+'.json')
        status, end = request.with_suffix('.status.json'), request.with_suffix('.exit.json')
        self.dispatch.write(request, dict(job=dict(job), machine='new', gpu=gpu, phase=phase,
            manifest_sha256=MANIFEST, sources=self.sources, status_path=str(status), exit_path=str(end)))
        with request.with_suffix('.wrapper.log').open('a') as log:
            p = subprocess.Popen([sys.executable, '-u', str(DISPATCH), '--phase-request', str(request)],
                env=dict(os.environ, LLMCL_ROOT=str(ROOT)), cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        who = identity(p.pid)
        assert who and who['uid'] == os.getuid() and who['pgid'] == who['pid']
        job.update(status='launching', wrapper=who, gpu=gpu, phase=phase,
            phase_status=str(status), phase_exit=str(end), request=str(request))
        self.processes[key] = p; self.save()
        return False  # the same allocator gave this slot to pct10

    def patch(self):
        original_load = self.cap.load_pipeline; original_states = self.cap.load_states
        original_policy = self.cap.policy
        def load_pipeline():
            p = original_load(); idle = p.idle
            if not self.inspection: p.idle = lambda g, m: self.offer(g, m, idle)
            return p
        def states():
            self.states01 = original_states(); return self.states01
        def policy():
            result = original_policy()
            if not self.locks:
                for path in [REPORT/'dispatch_new.lock']+[FRAME/('.pipeline_new_pct10_'+b+'.lock') for b in ('onevision', 'qwen')]:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    f = path.open('a'); fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB); self.locks.append(f)
            self.reconcile(); return result
        self.cap.load_pipeline = load_pipeline; self.cap.load_states = states
        self.cap.policy = policy


def main():
    assert sha(CAPACITY) == CAPACITY_SHA
    cap = module('original_pct01_capacity', CAPACITY)
    addition = Addition(cap, '--once' in sys.argv[1:])
    if addition.inspection:
        print(json.dumps(dict(pct10_inspection_only=True, no_launch=True,
            jobs=addition.state['jobs'], allowed_GPU=[1, 3], source_sha256=addition.sources)))
    addition.patch(); cap.main()


if __name__ == '__main__': main()
