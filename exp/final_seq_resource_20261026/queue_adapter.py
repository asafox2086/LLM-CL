"""Append two same-card resource workers after the existing conditional slot.

No allocator/service/scanner is created. Forward all existing conditional CLI
arguments, adding --resource-root ABS --resource-output NEW_ABS
--resource-python ABS --resource-onevision-binding ABS --resource-qwen-binding ABS.
Only the original idle chain supplies a GPU. This file's --resource-request
subprocess is a small serial supervisor, never another allocator.
"""
import argparse
import fcntl
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

CONDITIONAL_SHA = '1eaea047c6b6399b922387a80fc47b00a90728c021e50e006d11ee6de1f7910c'
ORDER = ('onevision', 'qwen')
LIMIT = 600


def read(path): return json.loads(Path(path).read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def apps(gpu):
    text = subprocess.check_output(['nvidia-smi', f'--id={gpu}', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True, timeout=5)
    return {int(x.strip()) for x in text.splitlines() if x.strip()}


def check_request(request):
    for item in request['sources'].values(): assert sha(item['path']) == item['sha256']
    assert Path(request['sources']['queue_adapter']['path']).resolve() == Path(__file__).resolve()
    for backbone in ORDER:
        item = request['bindings'][backbone]
        assert sha(item['path']) == item['sha256']
        b = read(item['path'])
        assert b['backbone'] == backbone and b['worker_sha256'] == request['sources']['worker']['sha256']
        assert b['GPU_uuids'][str(request['GPU'])] == request['GPU_uuid']


def supervised(request_path):
    request = read(request_path); check_request(request)
    helper = module('resource_existing_controller_helpers', request['sources']['controller_helper']['path'])
    root, output = Path(request['root']), Path(request['output'])
    assert output.is_absolute() and not output.exists()
    owner = helper.identity(os.getpid()); assert owner and owner['pid'] == owner['pgid'] and owner['uid'] == os.getuid()
    gpu = request['GPU']; started = time.monotonic()
    output.mkdir(exist_ok=False)
    helper.write(output/'SUPERVISOR_STARTED.json', dict(unix=time.time(), identity=owner, GPU=gpu,
        GPU_uuid=request['GPU_uuid'], request_sha256=sha(request_path), sources=request['sources'],
        bindings=request['bindings'], order=list(ORDER), per_phase_seconds=LIMIT, no_retry=True,
        new_scores=0, same_card_for_both_workers=True))
    def interrupted(signum, frame): raise InterruptedError('Resource supervisor signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    child = active = last_attempted = None; phases = []
    try:
        for backbone in ORDER:
            check_request(request)
            phase_clock, phase_unix = time.monotonic(), time.time()
            deadline = phase_clock+LIMIT
            # Retain the same supplied slot between workers. Wait only for its
            # preceding CUDA context/utilization to settle, within this phase's
            # existing600 cap; never search another card or retry a worker.
            while True:
                line = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
                    '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True, timeout=5).strip().split(',')
                assert line[0].strip() == request['GPU_uuid']
                if not apps(gpu) and int(line[1]) < 500 and int(line[2]) < 10: break
                assert time.monotonic() < min(deadline, phase_clock+30), 'Supplied slot did not settle; no model retry'
                time.sleep(1)
            destination = output/backbone
            assert not destination.exists()
            ref = request['bindings'][backbone]
            command = [request['python'], '-u', request['sources']['worker']['path'],
                '--root', str(root), '--output', str(destination), '--binding', ref['path'], '--gpu', str(gpu)]
            timed_out = False
            with (output/(backbone+'.log')).open('xb') as log:
                child = subprocess.Popen(command, cwd=root, env=dict(os.environ, LLMCL_ROOT=str(root),
                    OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', PYTHONUNBUFFERED='1'),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                active = helper.identity(child.pid)
                assert active and active['uid'] == owner['uid'] and active['pgid'] == child.pid
                last_attempted = active
                helper.write(output/('PHASE_'+backbone+'_STARTED.json'), dict(unix=phase_unix,
                    identity=active, GPU=gpu, backbone=backbone, binding_sha256=ref['sha256'],
                    worker_sha256=request['sources']['worker']['sha256'], output=str(destination),
                    deadline_unix=phase_unix+LIMIT, phase_limit_seconds=LIMIT,
                    includes_python_startup_and_slot_settling=True, command=command))
                try: code = child.wait(timeout=max(.001, deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    timed_out = True; helper.terminate_own(child, active); code = child.returncode
            destination.mkdir(exist_ok=True)
            item = dict(returncode=code, worker=active, GPU=gpu, backbone=backbone,
                binding_sha256=ref['sha256'], worker_sha256=request['sources']['worker']['sha256'],
                identity_absent=helper.identity(child.pid) is None, timeout=timed_out,
                unix=time.time(), phase_started_unix=phase_unix, seconds=time.monotonic()-phase_clock)
            helper.write(destination/'EXIT.json', item)
            phases.append(item)
            assert code == 0 and not timed_out and item['identity_absent'], backbone+' resource worker failed'
            completed = read(destination/'COMPLETE.json'); begun = read(destination/'STARTED.json')
            assert completed['identity'] == begun['identity'] == active
            assert completed['binding_sha256'] == begun['binding_sha256'] == ref['sha256']
            assert completed['source_sha256']['worker'] == request['sources']['worker']['sha256']
            assert completed['physical_GPU'] == begun['physical_GPU'] == gpu
            assert completed['GPU_uuid'] == request['GPU_uuid'] and completed['seconds'] <= LIMIT
            assert completed['new_generations'] == 5 and completed['new_scores'] == completed['backwards'] == completed['optimizer_updates'] == 0
            assert sha(destination/'windows.json') == completed['windows_sha256']
            item.update(complete_sha256=sha(destination/'COMPLETE.json'), started_sha256=sha(destination/'STARTED.json'))
            child = active = None
        helper.write(output/'SUPERVISOR_COMPLETE.json', dict(unix=time.time(), identity=owner,
            GPU=gpu, GPU_uuid=request['GPU_uuid'], request_sha256=sha(request_path), sources=request['sources'],
            phases=phases, active_worker=None, no_retry=True, new_generations=10, new_scores=0,
            total_seconds=time.monotonic()-started, model_phase_budget_seconds=1200,
            no_automatic_verifier=True))
    except BaseException:
        failure = traceback.format_exc()
        if child is not None and active is not None:
            try: helper.terminate_own(child, active)
            except BaseException: failure += '\nCleanup: '+traceback.format_exc()
        helper.write(output/'SUPERVISOR_FAILURE.json', dict(unix=time.time(), identity=owner,
            GPU=gpu, GPU_uuid=request['GPU_uuid'], request_sha256=sha(request_path), sources=request['sources'],
            phases=phases, active_worker=helper.identity(active['pid']) if active else None,
            last_attempted_worker=last_attempted, no_retry=True, traceback=failure,
            total_seconds=time.monotonic()-started, no_automatic_verifier=True))
        raise


class ResourceSlot:
    def __init__(self, conditional_slot, args, conditional_path):
        self.parent = conditional_slot; self.ids = conditional_slot.position
        self.root, self.output, self.python = args.resource_root, args.resource_output, args.resource_python
        self.source_dir = Path(__file__).resolve().parent
        # Reviewed source may live in the independent metadata work tree;
        # asset/data ROOT is not the source location. Bind its actual files.
        assert (self.source_dir/'worker.py').is_file()
        assert self.root.is_absolute() and self.root.is_dir()
        assert self.output.is_absolute() and self.output.parent.is_dir() and str(self.output).startswith('/data2/')
        assert self.python.is_absolute() and self.python.is_file()
        self.path = self.output.with_name(self.output.name+'.slot.json')
        self.request_path = self.output.with_name(self.output.name+'.request.json')
        self.sources = {k: dict(path=str(p), sha256=sha(p)) for k, p in {
            'queue_adapter': Path(__file__).resolve(), 'worker': self.source_dir/'worker.py',
            'conditional_wrapper': conditional_path,
            'controller_helper': conditional_path.parent/'controller.py'}.items()}
        self.bindings = {}
        for backbone in ORDER:
            p = getattr(args, 'resource_'+backbone+'_binding')
            assert p.is_absolute() and p.is_file()
            self.bindings[backbone] = dict(path=str(p), sha256=sha(p))
            b = read(p); assert b['backbone'] == backbone and b['worker_sha256'] == self.sources['worker']['sha256']
        self.child, self.pipelines = None, []

    def record(self):
        value = read(self.path)
        assert value['sources'] == self.sources and value['bindings'] == self.bindings
        assert value['root'] == str(self.root) and value['output'] == str(self.output) and value['no_retry']
        return value

    def status(self, value, status, **fields):
        updated = dict(value, status=status, **fields)
        if updated != value:
            updated['status_changed_unix'] = time.time(); self.ids.write(self.path, updated)

    def held_gpu(self):
        if not self.path.exists(): return None
        value = self.record()
        if value.get('released'): return None
        controller = value.get('controller')
        if controller and self.ids.same_identity(self.ids.identity(controller['pid']), controller): return value['GPU']
        if self.child is not None: self.child.poll()
        try:
            terminal = [p for p in (self.output/'SUPERVISOR_COMPLETE.json', self.output/'SUPERVISOR_FAILURE.json') if p.exists()]
            assert controller and len(terminal) == 1
            p = terminal[0]; proof = read(p)
            assert self.ids.same_identity(proof['identity'], controller) and proof['GPU'] == value['GPU']
            assert proof['sources'] == self.sources and proof['request_sha256'] == value['request_sha256']
            assert proof['no_retry'] and proof['active_worker'] is None
            phases = proof['phases']; assert [x['backbone'] for x in phases] == list(ORDER[:len(phases)])
            assert len(phases) <= 2
            if p.name == 'SUPERVISOR_COMPLETE.json': assert len(phases) == 2 and all(x['returncode'] == 0 for x in phases)
            for x in phases:
                assert x['identity_absent'] and not self.ids.same_identity(self.ids.identity(x['worker']['pid']), x['worker'])
            if p.name == 'SUPERVISOR_FAILURE.json':
                last = proof['last_attempted_worker']
                assert last is None or not self.ids.same_identity(self.ids.identity(last['pid']), last)
            assert not apps(value['GPU']), 'Compute context still present; keep slot'
            self.status(value, 'complete' if p.name == 'SUPERVISOR_COMPLETE.json' else 'failed_closed',
                released=True, terminal_path=str(p), terminal_sha256=sha(p), actual_worker_and_supervisor_absent=True, NVML_empty=True)
            return None
        except Exception as error:
            self.status(value, 'orphaned_hold', reason=repr(error)); return value['GPU']

    def launch(self, gpu):
        assert not self.path.exists() and not self.output.exists() and not self.request_path.exists()
        assert self.parent.path.exists() and self.parent.record().get('released') is True
        uuids = {read(v['path'])['GPU_uuids'][str(gpu)] for v in self.bindings.values()}; assert len(uuids) == 1
        request = dict(root=str(self.root), output=str(self.output), python=str(self.python), GPU=int(gpu),
            GPU_uuid=uuids.pop(), sources=self.sources, bindings=self.bindings)
        check_request(request); self.ids.write(self.request_path, request, exclusive=True)
        value = dict(request, status='claimed', unix=time.time(), released=False, no_retry=True,
            request_path=str(self.request_path), request_sha256=sha(self.request_path), controller=None,
            coordinator=self.ids.identity(os.getpid()), conditional_slot_path=str(self.parent.path),
            conditional_release_sha256=sha(self.parent.path))
        self.ids.write(self.path, value, exclusive=True)
        try:
            command = [str(self.python), '-u', str(Path(__file__).resolve()), '--resource-request', str(self.request_path)]
            log_path = self.output.with_name(self.output.name+'.supervisor.log')
            with log_path.open('xb') as log:
                self.child = subprocess.Popen(command, cwd=self.root, env=dict(os.environ, LLMCL_ROOT=str(self.root)),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            who = self.ids.identity(self.child.pid)
            assert who and who['uid'] == os.getuid() and who['pgid'] == who['pid']
            self.status(value, 'running', controller=who, command=command, log_path=str(log_path))
        except BaseException as error:
            self.status(value, 'orphaned_hold', reason='Launch exception: '+repr(error), attempted_PID=self.child.pid if self.child else None)

    def wrap(self, pipeline):
        if getattr(pipeline, '_final_resource_wrapped', False): return pipeline
        original_idle = pipeline.idle  # original position and conditional hooks already installed
        def idle(gpu, minimum):
            if self.held_gpu() == gpu: return False
            available = original_idle(gpu, minimum)
            if available and not self.path.exists() and self.parent.path.exists() and self.parent.record().get('released') is True:
                self.launch(gpu); return False
            return available
        pipeline.idle = idle; pipeline._final_resource_wrapped = True; self.pipelines.append(pipeline)
        return pipeline

    def drain(self):
        assert self.pipelines
        lock_path = self.parent.baseline_root/'summary/two_backbone_pct10_20261025/dispatch_old.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                self.held_gpu()
                if self.path.exists() and self.record().get('released'): return
                for gpu in (0, 1, 2, 3): self.pipelines[0].idle(gpu, 8500)
                time.sleep(15)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--resource-request', type=Path)
    args, _ = parser.parse_known_args()
    if args.resource_request: supervised(args.resource_request); return
    for name in ('root', 'output', 'python', 'onevision-binding', 'qwen-binding'):
        parser.add_argument('--resource-'+name, type=Path, required=True)
    args, remaining = parser.parse_known_args()
    assert '--phase-request' not in remaining
    conditional_root = Path(remaining[remaining.index('--conditional-root')+1])
    assert conditional_root.is_absolute()
    path = conditional_root/'exp/conditional_shared_budget_20261026/v1/queue_adapter.py'
    assert sha(path) == CONDITIONAL_SHA
    conditional = module('unchanged_conditional_capacity_owner', path)
    original_slot = conditional.ConditionalSlot
    class AppendedConditionalSlot(original_slot):
        def __init__(self, *values, **keywords):
            super().__init__(*values, **keywords)
            self.resource = ResourceSlot(self, args, path)
        def wrap(self, pipeline, position_slot):
            return self.resource.wrap(super().wrap(pipeline, position_slot))
        def drain_after_normal_return(self):
            super().drain_after_normal_return()
            self.resource.drain()
    conditional.ConditionalSlot = AppendedConditionalSlot
    sys.argv = [str(path), *remaining]
    conditional.main()  # Original coordinator, children, source/queue and normal-return drain.


if __name__ == '__main__': main()
