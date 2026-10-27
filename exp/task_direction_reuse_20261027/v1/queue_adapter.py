"""Reserve the current-task direction four-phase experiment through the existing allocator.

Only the existing OLD coordinator's pipeline.idle chain supplies a card. The
coordinator ROOT stays unchanged; only the research child receives its own ROOT.
No model imports, GPU signals, baseline queue edits, or retries.
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

POSITION_SHA256 = '23a7a6fff3f33766a22d636c632476008560023c62da1b0c504f7b9913a9f833'
ORDER = [(a,p) for a in ('aligned','random') for p in ('train','eval')]


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


class DirectionSlot:
    def __init__(self, position, baseline_root, science_root, output, python, binding):
        self.position = position
        self.baseline_root, self.root = baseline_root, science_root
        self.output, self.python = output, python
        self.source_dir = Path(__file__).resolve().parent
        assert self.source_dir == self.root / 'exp/task_direction_reuse_20261027/v1'
        assert self.root != self.baseline_root and self.root.is_dir()
        assert self.output.is_absolute() and self.output.parent.is_dir()
        assert self.python.is_absolute() and self.python.is_file()
        self.path = self.root / 'summary/task_direction_reuse_20261027/SLOT.json'
        assert self.path.parent.is_dir()
        self.sources = {n: sha(self.source_dir / n) for n in
                        ('queue_adapter.py','controller.py','verify.py','worker.py','adaptation.py','calibration.py')}
        self.binding = binding
        assert binding.is_absolute()
        self.binding_sha256 = sha(binding)
        data=read(binding)
        assert data['schema']=='task-direction-reuse-v1'
        assert data['sources']=={n:h for n,h in self.sources.items() if n!='queue_adapter.py'}
        self.GPU=1
        self.protocol = Path(read(binding)['protocol']['path'])
        assert self.protocol.is_absolute()
        self.protocol_sha256 = sha(self.protocol)
        self.child, self.pipelines = None, []

    def record(self):
        value = read(self.path)
        assert value['baseline_root'] == str(self.baseline_root)
        assert value['science_root'] == str(self.root) and value['output'] == str(self.output)
        assert value['sources'] == self.sources and value['protocol_sha256'] == self.protocol_sha256
        assert value['no_retry'] and value['binding_sha256'] == self.binding_sha256
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
        controller = value.get('controller')
        identity = self.position.identity
        if controller and self.position.same_identity(identity(controller['pid']), controller):
            return value['GPU']
        if self.child is not None:
            self.child.poll()
        terminals = [p for p in (self.output / 'SUPERVISOR_COMPLETE.json',
                     self.output / 'SUPERVISOR_FAILURE.json') if p.exists()]
        try:
            assert controller is not None and len(terminals) == 1
            terminal = terminals[0]; proof = read(terminal)
            assert self.position.same_identity(proof['identity'], controller)
            assert proof['GPU'] == value['GPU'] and proof['no_retry']
            assert proof['active_worker'] is None
            assert proof['source_sha256'] == {n: h for n, h in self.sources.items() if n != 'queue_adapter.py'}
            assert proof['protocol'] == {'path': str(self.protocol), 'sha256': self.protocol_sha256}
            phases = proof['phases']
            assert [(p['arm'], p['phase']) for p in phases] == ORDER[:len(phases)]
            assert len(phases) <= len(ORDER)
            if terminal.name == 'SUPERVISOR_COMPLETE.json':
                assert len(phases) == len(ORDER) and all(p['returncode'] == 0 for p in phases)
            pids = {int(controller['pid'])}
            for phase in phases:
                worker = phase['identity']
                assert not self.position.same_identity(identity(worker['pid']), worker)
                pids.add(int(worker['pid']))
            if terminal.name == 'SUPERVISOR_FAILURE.json':
                worker = proof['last_attempted_worker']
                if worker is not None:
                    assert not self.position.same_identity(identity(worker['pid']), worker)
                    pids.add(int(worker['pid']))
            output = subprocess.check_output(['nvidia-smi', f"--id={value['GPU']}",
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=5)
            present = {int(line.strip()) for line in output.splitlines() if line.strip()}
            assert not present.intersection(pids)
            self.status(value, 'complete' if terminal.name == 'SUPERVISOR_COMPLETE.json'
                        else 'failed_closed', released=True, terminal_path=str(terminal),
                        terminal_sha256=sha(terminal), original_workers_and_controller_absent=True,
                        own_compute_PIDs_absent=True)
            return None
        except Exception as error:
            self.status(value, 'orphaned_hold', reason=repr(error))
            return value['GPU']

    def launch(self, gpu, prior_slot):
        assert not self.path.exists() and not self.output.exists()
        assert prior_slot.path.exists() and prior_slot.record().get('released') is True
        assert all(sha(self.source_dir / n) == h for n, h in self.sources.items())
        assert sha(self.protocol) == self.protocol_sha256
        assert sha(self.binding) == self.binding_sha256
        value = dict(status='claimed', unix=time.time(), GPU=int(gpu),
            baseline_root=str(self.baseline_root), science_root=str(self.root), output=str(self.output),
            sources=self.sources, protocol_sha256=self.protocol_sha256, binding_sha256=self.binding_sha256, controller=None,
            no_retry=True, released=False, coordinator=self.position.identity(os.getpid()),
            prior_slot_path=str(prior_slot.path), prior_release_sha256=sha(prior_slot.path))
        write(self.path, value, exclusive=True)
        try:
            command = [str(self.python), '-u', str(self.source_dir / 'controller.py'),
                '--root', str(self.root), '--output', str(self.output), '--gpu', str(gpu),
                '--python', str(self.python), '--binding', str(self.binding)]
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
        if getattr(pipeline, '_task_direction_slot_wrapped', False):
            return pipeline
        original_idle = pipeline.idle  # Already includes the unchanged position reservation.

        def idle(gpu, minimum):
            if self.held_gpu() == gpu:
                return False
            available = original_idle(gpu, minimum)
            if (available and gpu == self.GPU and not self.path.exists() and prior_slot.path.exists()
                    and prior_slot.record().get('released') is True):
                self.launch(gpu, prior_slot)
                return False
            return available

        pipeline.idle = idle
        pipeline._task_direction_slot_wrapped = True
        self.pipelines.append(pipeline)
        return pipeline

    def drain_after_normal_return(self):
        """The original main can return with no baseline job left to call idle."""
        assert self.pipelines
        lock_path = self.baseline_root / 'summary/two_backbone_pct10_20261025/dispatch_old.lock'
        with lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                self.held_gpu()
                if self.path.exists() and self.record().get('released'):
                    return
                # Same pipeline objects, OLD card order and original idle threshold;
                # never a second live allocator or a second GPU search policy.
                for gpu in (0, 1, 2, 3):
                    self.pipelines[0].idle(gpu, 8500)
                time.sleep(15)


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--direction-root',type=Path,required=True)
    parser.add_argument('--direction-binding',type=Path,required=True)
    parser.add_argument('--direction-python',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    baseline=Path(os.environ['LLMCL_ROOT']).resolve()
    # Preserve every existing outer/downstream argument, including the old v2
    # slot root and binding; use that wrapper as the one allocator entry.
    assert remaining.count('--type-aux-v2-root')==1
    prior_root=Path(remaining[remaining.index('--type-aux-v2-root')+1])
    assert prior_root.is_absolute()
    prior_path=prior_root/'exp/type_auxiliary_supervision_20261027/v2/queue_adapter.py'
    assert sha(prior_path)=='6bc952b98c65980812e14da0d124b43264cb2034832d5e900e262627ba0ffa47'
    position_path=baseline/'exp/finance_position_diagnostic_20261026/queue_adapter.py'
    assert sha(position_path)==POSITION_SHA256
    def load(name,path):
        spec=importlib.util.spec_from_file_location(name,path)
        value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
        return value
    position=load('task_direction_identity',position_path)
    slot=DirectionSlot(position,baseline,args.direction_root,args.direction_root/'run_v1',
        args.direction_python,args.direction_binding)
    prior=load('task_direction_preserved_type_aux_v2',prior_path)
    original=prior.TypeAuxSlot
    class WithDirection(original):
        def wrap(self,pipeline,prior_slot):
            return slot.wrap(super().wrap(pipeline,prior_slot),self)
        def drain_after_normal_return(self):
            super().drain_after_normal_return()
            slot.drain_after_normal_return()
    prior.TypeAuxSlot=WithDirection
    sys.argv=[str(prior_path),*remaining]
    prior.main()


if __name__=='__main__':main()
