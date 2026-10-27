"""One formal candidate slot on the unchanged OLD allocator/idle chain.

Only the child controller receives the study ROOT. Keep every downstream argv,
baseline job and lock unchanged; never send a signal or retry a closed slot.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PRIOR_SHA256 = 'e3ad3c231b0fcad025331ac438043eb3cf95767018335e20a2cb445781e112a3'
POSITION_SHA256 = '23a7a6fff3f33766a22d636c632476008560023c62da1b0c504f7b9913a9f833'
ORDER = ['preflight', 'reload', 'training']


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value, exclusive=False):
    path = Path(path)
    temporary = path if exclusive else path.with_suffix(path.suffix + '.partial')
    with temporary.open('x' if exclusive else 'w', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if not exclusive:
        temporary.replace(path)


class FormalSlot:
    def __init__(self, position, baseline_root, science_root, python, binding):
        self.position = position
        self.baseline_root, self.root = baseline_root, science_root
        self.python, self.binding = python, binding
        self.source_dir = Path(__file__).resolve().parent
        assert self.root.is_absolute() and self.root.is_dir() and self.root != self.baseline_root
        assert self.source_dir == self.root / 'exp/task_gradient_formal_20261028'
        assert self.python.is_absolute() and self.python.is_file()
        assert self.binding.is_absolute() and self.binding.is_file()
        self.output = self.root / 'run_v1'
        self.path = self.root / 'summary/task_gradient_formal_20261028/SLOT.json'
        assert self.path.parent.is_dir()
        self.binding_sha256 = sha(self.binding)
        data = read(self.binding)
        self.sources = {p.name: sha(p) for p in sorted(self.source_dir.glob('*.py'))}
        assert {'queue_adapter.py', 'controller.py', 'run.py', 'task_gradient_adapter.py'} <= set(self.sources)
        assert data['sources'] == self.sources
        self.protocol = Path(data['protocol']['path'])
        assert self.protocol.is_absolute() and sha(self.protocol) == data['protocol']['sha256']
        self.protocol_sha256 = data['protocol']['sha256']
        self.GPU = 1
        self.child, self.pipelines = None, []

    def record(self):
        value = read(self.path)
        assert value['baseline_root'] == str(self.baseline_root)
        assert value['science_root'] == str(self.root) and value['output'] == str(self.output)
        assert value['sources'] == self.sources and value['protocol_sha256'] == self.protocol_sha256
        assert value['binding_sha256'] == self.binding_sha256 and value['no_retry']
        assert value['GPU'] == self.GPU
        return value

    def status(self, value, status, **fields):
        updated = dict(value, status=status, **fields)
        if updated != value:
            updated['status_changed_unix'] = time.time()
            write(self.path, updated)
        return updated

    def held_gpu(self):
        if not self.path.exists():
            return None
        value = self.record()
        if value.get('released'):
            return None
        identity = self.position.identity
        controller = value.get('controller')
        # A long initial evaluation or formal training is not a dead process.
        if controller and self.position.same_identity(identity(controller['pid']), controller):
            return value['GPU']
        if self.child is not None:
            self.child.poll()
        terminals = [p for p in (self.output / 'FORMAL_COMPLETE.json', self.output / 'FORMAL_FAILURE.json') if p.exists()]
        try:
            assert controller is not None and len(terminals) == 1
            terminal = terminals[0]; proof = read(terminal)
            assert self.position.same_identity(proof['identity'], controller)
            assert proof['GPU'] == self.GPU and proof['no_retry']
            assert proof['active_worker'] is None
            assert proof['source_sha256'] == self.sources
            assert proof['binding_sha256'] == self.binding_sha256
            assert proof['protocol'] == {'path': str(self.protocol), 'sha256': self.protocol_sha256}
            phases = proof['phases']
            assert [p['phase'] for p in phases] == ORDER[:len(phases)] and len(phases) <= len(ORDER)
            if terminal.name == 'FORMAL_COMPLETE.json':
                assert len(phases) == len(ORDER)
                assert all(p['returncode'] == 0 and not p['timeout'] and p['identity_absent'] for p in phases)
            pids = {int(controller['pid'])}
            for phase in phases:
                worker = phase['identity']
                assert phase['identity_absent']
                assert not self.position.same_identity(identity(worker['pid']), worker)
                pids.add(int(worker['pid']))
            worker = proof['last_attempted_worker']
            if worker is not None:
                assert not self.position.same_identity(identity(worker['pid']), worker)
                pids.add(int(worker['pid']))
            output = subprocess.check_output(['nvidia-smi', f'--id={self.GPU}',
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
            present = {int(line.strip()) for line in output.splitlines() if line.strip()}
            assert not present.intersection(pids)
            self.status(value, 'complete' if terminal.name == 'FORMAL_COMPLETE.json' else 'failed_closed',
                released=True, terminal_path=str(terminal), terminal_sha256=sha(terminal),
                original_workers_and_controller_absent=True, own_compute_PIDs_absent=True)
            return None
        except Exception as error:
            self.status(value, 'orphaned_hold', reason=repr(error))
            return value['GPU']

    def launch(self, gpu, prior_slot):
        assert gpu == self.GPU and not self.path.exists() and not self.output.exists()
        prior = prior_slot.record()
        assert prior.get('released') is True and prior['status'] == 'complete'
        assert Path(prior['terminal_path']).name == 'SUPERVISOR_COMPLETE.json'
        assert sha(prior['terminal_path']) == prior['terminal_sha256']
        assert all(sha(self.source_dir / n) == h for n, h in self.sources.items())
        assert sha(self.protocol) == self.protocol_sha256 and sha(self.binding) == self.binding_sha256
        value = dict(status='claimed', unix=time.time(), GPU=int(gpu),
            baseline_root=str(self.baseline_root), science_root=str(self.root), output=str(self.output),
            sources=self.sources, protocol_sha256=self.protocol_sha256, binding_sha256=self.binding_sha256,
            controller=None, no_retry=True, released=False, coordinator=self.position.identity(os.getpid()),
            prior_slot_path=str(prior_slot.path), prior_release_sha256=sha(prior_slot.path))
        write(self.path, value, exclusive=True)
        try:
            command = [str(self.python), '-u', str(self.source_dir / 'controller.py'),
                '--root', str(self.root), '--binding', str(self.binding), '--gpu', str(gpu), '--python', str(self.python)]
            log_path = self.path.with_suffix('.controller.log')
            with log_path.open('xb') as log:
                self.child = subprocess.Popen(command, cwd=self.root,
                    env=dict(os.environ, LLMCL_ROOT=str(self.root)), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            who = self.position.identity(self.child.pid)
            assert who and who['uid'] == os.getuid() and who['pgid'] == who['pid']
            self.status(value, 'running', controller=who, command=command, log_path=str(log_path))
        except BaseException as error:
            self.status(value, 'orphaned_hold', reason='Launch exception: ' + repr(error),
                attempted_controller_PID=self.child.pid if self.child else None)

    def wrap(self, pipeline, prior_slot):
        if getattr(pipeline, '_task_gradient_formal_slot_wrapped', False):
            return pipeline
        original_idle = pipeline.idle

        def idle(gpu, minimum):
            if self.held_gpu() == gpu:
                return False
            available = original_idle(gpu, minimum)
            if (available and gpu == self.GPU and not self.path.exists() and prior_slot.path.exists()
                    and prior_slot.record().get('released') is True and prior_slot.record()['status'] == 'complete'):
                self.launch(gpu, prior_slot)
                return False
            return available

        pipeline.idle = idle
        pipeline._task_gradient_formal_slot_wrapped = True
        self.pipelines.append(pipeline)
        return pipeline

    def drain_after_normal_return(self):
        assert self.pipelines
        lock_path = self.baseline_root / 'summary/two_backbone_pct10_20261025/dispatch_old.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                self.held_gpu()
                if self.path.exists() and self.record().get('released'):
                    return
                for gpu in (0, 1, 2, 3):
                    self.pipelines[0].idle(gpu, 8500)
                time.sleep(15)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--task-gradient-root', type=Path, required=True)
    parser.add_argument('--task-gradient-binding', type=Path, required=True)
    parser.add_argument('--task-gradient-python', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    baseline = Path(os.environ['LLMCL_ROOT']).resolve()
    assert remaining.count('--direction-local-root') == 1
    prior_root = Path(remaining[remaining.index('--direction-local-root') + 1])
    assert prior_root.is_absolute()
    prior_path = prior_root / 'exp/task_direction_local_20261027/v1/queue_adapter.py'
    assert sha(prior_path) == PRIOR_SHA256
    position_path = baseline / 'exp/finance_position_diagnostic_20261026/queue_adapter.py'
    assert sha(position_path) == POSITION_SHA256

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
        return value

    position = load('task_gradient_formal_identity', position_path)
    slot = FormalSlot(position, baseline, args.task_gradient_root, args.task_gradient_python, args.task_gradient_binding)
    prior = load('task_gradient_formal_preserved_local', prior_path)
    original = prior.LocalDirectionSlot

    class WithFormal(original):
        def wrap(self, pipeline, prior_slot):
            return slot.wrap(super().wrap(pipeline, prior_slot), self)

        def drain_after_normal_return(self):
            super().drain_after_normal_return()
            slot.drain_after_normal_return()

    prior.LocalDirectionSlot = WithFormal
    sys.argv = [str(prior_path), *remaining]
    prior.main()


if __name__ == '__main__':
    main()
