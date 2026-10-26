"""Reserve one existing pct10 allocator slot for one fixed position diagnostic.

Stdlib only. No independent GPU search, baseline job mutation, model import,
signals or retries. All normal phase requests still use the original routes.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

DYNAMIC_SHA256 = '6c5006c6ece0df9550b30eabbe1949ac50a89df4ca7ff3ff0aec656bb4cdc690'
ORDER = [('pct01', 15), ('pct01', 16), ('pct10', 15), ('pct10', 16)]


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        path = Path('/proc') / str(pid)
        fields = (path / 'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return dict(pid=int(pid), startticks=fields[19], uid=path.stat().st_uid,
                    pgid=int(fields[2]))
    except (OSError, ValueError, TypeError):
        return None


def same_identity(left, right):
    return bool(left and right and all(str(left[k]) == str(right[k])
                for k in ('pid', 'startticks', 'uid', 'pgid')))


def write(path, value, exclusive=False):
    path = Path(path)
    target = path if exclusive else path.with_suffix(path.suffix + '.partial')
    with target.open('x' if exclusive else 'w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    if not exclusive:
        target.replace(path)


def validate(admission_path):
    admission = read(admission_path)
    assert Path(admission['root']).is_absolute()
    root = Path(admission['root']).resolve()
    assert root == Path(os.environ['LLMCL_ROOT']).resolve()
    source_dir = Path(admission['source_dir'])
    assert source_dir.is_absolute() and source_dir.resolve() == Path(__file__).resolve().parent
    assert admission['dynamic_wrapper_sha256'] == DYNAMIC_SHA256
    assert {'queue_adapter.py', 'controller.py', 'worker.py'} <= set(admission['sources'])
    for name, expected in admission['sources'].items():
        assert Path(name).name == name
        assert sha(source_dir / name) == expected, name
    assert sha(Path(__file__)) == admission['sources']['queue_adapter.py']
    for key in ('binding', 'protocol'):
        ref = admission[key]
        assert Path(ref['path']).is_absolute() and sha(ref['path']) == ref['sha256']
    for key in ('slot_path', 'output_path', 'python'):
        assert Path(admission[key]).is_absolute()
    assert Path(admission['python']).is_file()
    assert Path(admission['slot_path']).parent.is_dir()
    assert Path(admission['output_path']).parent.is_dir()
    return root, admission


class Slot:
    def __init__(self, admission_path, admission):
        self.path = Path(admission['slot_path'])
        self.output = Path(admission['output_path'])
        self.admission_path, self.admission = admission_path, admission
        self.admission_sha256 = sha(admission_path)
        self.child = None

    def record(self):
        value = read(self.path)
        assert value['admission_sha256'] == self.admission_sha256
        assert value['output_path'] == str(self.output) and value['no_retry']
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
            return None  # Once ledger remains; never launch again.
        controller = value.get('controller')
        if same_identity(identity(controller['pid']) if controller else None, controller):
            return value['GPU']  # Includes CUDA-free gaps between all four phases.
        if self.child is not None:
            self.child.poll()  # Reap this coordinator's exited Popen child.
        terminal_paths = [p for p in (self.output / 'SUPERVISOR_COMPLETE.json',
            self.output / 'SUPERVISOR_FAILURE.json') if p.exists()]
        if controller is None or len(terminal_paths) != 1:
            self.status(value, 'orphaned_hold', reason='Missing/ambiguous terminal or launch identity')
            return value['GPU']
        try:
            terminal_path = terminal_paths[0]
            proof = read(terminal_path)
            assert same_identity(proof['identity'], controller)
            assert proof['admission_sha256'] == self.admission_sha256
            assert proof['GPU'] == value['GPU'] and proof['no_retry']
            assert proof['active_worker'] is None
            phases = proof['phases']
            assert [(p['condition'], p['stage']) for p in phases] == ORDER[:len(phases)]
            assert len(phases) <= 4
            if terminal_path.name == 'SUPERVISOR_COMPLETE.json':
                assert len(phases) == 4 and all(p['returncode'] == 0 for p in phases)
            worker_pids = set()
            for phase in phases:
                worker = phase['identity']
                assert phase['identity_absent'] is True
                assert not same_identity(identity(worker['pid']), worker)
                worker_pids.add(int(worker['pid']))
            if terminal_path.name == 'SUPERVISOR_FAILURE.json':
                # The controller can fail before appending a PHASE_EXIT item.
                # Cleanup's active_worker=None alone does not prove this last
                # actual Popen worker has gone; check its original identity too.
                worker = proof['last_attempted_worker']
                if worker is not None:
                    assert not same_identity(identity(worker['pid']), worker)
                    worker_pids.add(int(worker['pid']))
            # A reused PID or uncertain NVML result conservatively keeps the slot.
            output = subprocess.check_output(['nvidia-smi', f"--id={value['GPU']}",
                '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                text=True, timeout=5)
            present = {int(line.strip()) for line in output.splitlines() if line.strip()}
            assert not present.intersection(worker_pids | {int(controller['pid'])})
            self.status(value, 'complete' if terminal_path.name == 'SUPERVISOR_COMPLETE.json'
                        else 'failed_closed', released=True, terminal_path=str(terminal_path),
                        terminal_sha256=sha(terminal_path), actual_workers_absent=True,
                        original_controller_absent=True, own_compute_PIDs_absent=True)
            return None
        except Exception as error:
            self.status(value, 'orphaned_hold', reason='Terminal/identity/NVML check pending: ' + repr(error))
            return value['GPU']

    def launch(self, gpu):
        assert not self.path.exists(), 'One attempt only, including startup failures'
        assert not self.output.exists(), 'Never reuse model output or a prior controller'
        # Claim before Popen: even a startup exception cannot cause a new attempt.
        value = dict(status='claimed', unix=time.time(), GPU=int(gpu),
            admission_sha256=self.admission_sha256, admission_path=str(self.admission_path),
            output_path=str(self.output), controller=None, released=False, no_retry=True,
            coordinator=identity(os.getpid()))
        write(self.path, value, exclusive=True)
        try:
            controller = Path(self.admission['source_dir']) / 'controller.py'
            command = [self.admission['python'], '-u', str(controller), '--gpu', str(gpu),
                       '--admission', str(self.admission_path), '--output', str(self.output)]
            log_path = self.path.with_suffix(self.path.suffix + '.controller.log')
            with log_path.open('xb') as log:
                self.child = subprocess.Popen(command, cwd=self.admission['root'],
                    env=dict(os.environ), stdin=subprocess.DEVNULL, stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
            who = identity(self.child.pid)
            assert who and who['uid'] == os.getuid() and who['pgid'] == who['pid']
            self.status(value, 'running', controller=who, command=command,
                        controller_log=str(log_path))
        except BaseException as error:
            # No release without a trustworthy terminal; no baseline/model signal.
            self.status(value, 'orphaned_hold', reason='Launch exception: ' + repr(error),
                        attempted_controller_PID=self.child.pid if self.child else None)

    def wrap(self, pipeline):
        if getattr(pipeline, '_position_slot_wrapped', False):
            return pipeline
        original_idle = pipeline.idle

        def idle(gpu, minimum):
            held = self.held_gpu()
            if held == gpu:
                return False
            result = original_idle(gpu, minimum)
            if result and not self.path.exists():
                self.launch(gpu)
                return False
            return result

        pipeline.idle = idle
        pipeline._position_slot_wrapped = True
        return pipeline


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--admission', type=Path)
    known, remaining = parser.parse_known_args()
    supplied = known.admission or os.environ.get('LLMCL_POSITION_ADMISSION')
    assert supplied and Path(supplied).is_absolute(), 'Absolute --admission or inherited LLMCL_POSITION_ADMISSION required'
    admission_path = Path(supplied).resolve()
    root, admission = validate(admission_path)
    os.environ['LLMCL_POSITION_ADMISSION'] = str(admission_path)
    dynamic_path = root / 'exp/two_backbone_pct10_20261025/dispatch_dynamic.py'
    assert sha(dynamic_path) == DYNAMIC_SHA256
    spec = importlib.util.spec_from_file_location('position_original_dynamic_dispatch', dynamic_path)
    dynamic = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dynamic)  # Only the original stdlib coordinator.
    if '--phase-request' not in remaining:
        assert '--machine' not in remaining or remaining[remaining.index('--machine') + 1] == 'old'
        original_prepare = dynamic.prepare
        slot = Slot(admission_path, admission)

        def prepare():
            module, ready = original_prepare()
            original_pipe = module.pipe
            module.pipe = lambda *args, **kwargs: slot.wrap(original_pipe(*args, **kwargs))
            # Original main spawns its __file__; children inherit admission and
            # re-enter here without adding the coordinator-only idle hook.
            module.__file__ = str(Path(__file__).resolve())
            return module, ready

        dynamic.prepare = prepare
    sys.argv = [str(Path(__file__).resolve()), *remaining]
    dynamic.main()


if __name__ == '__main__':
    main()
