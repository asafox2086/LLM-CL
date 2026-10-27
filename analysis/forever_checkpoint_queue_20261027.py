"""Independent pct01 Qwen FOREVER waiter for the retained-stage runner.

No old source, queue or GPU process is modified. The old CPU waiter must already
be gone and have no GPU child. Preserve its GPU2/established-native-priority idle
guard. Run the existing source-matched preflight/reload once, then a fresh formal
run. A recorded boundary pause requires an explicit matching resume request.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
FRAME_NAME = 'two_backbone_pct01_forever_checkpoint_20261027'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
IO_CHANGES = {'run.py', 'rolling_io.py', 'pipeline.py', 'execution_policy.json'}
HELPERS = ('analysis/forever_multimodal_core_20261019.py',
           'analysis/forever_multimodal_replay_v2_20261019.py')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    with temporary.open('w', encoding='utf8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def identity(pid):
    try:
        p = Path('/proc')/str(pid); fields = (p/'stat').read_text().rsplit(') ', 1)[1].split()
        if fields[0] == 'Z': return None
        return dict(pid=int(pid),startticks=int(fields[19]),uid=p.stat().st_uid,pgid=int(fields[2]),
                    cmdline=(p/'cmdline').read_bytes().replace(b'\0', b' ').decode())
    except (OSError, ValueError): return None


def available(backbone):
    # Exact original waiter guard: only GPU2 for Qwen and established native
    # pct01 jobs keep priority. This is not a second multi-GPU allocator.
    candidates=[0] if backbone=='onevision' else [2]
    if backbone=='qwen':
        registry=ROOT/'summary/two_backbone_native_cl_20261017_v2/new3090_run_status.json'
        state=json.loads(registry.read_text())
        jobs=[j for j in state['jobs'].values()
              if j.get('host')=='new3090' and j['condition']=='pct01']
        if any(j['status'] in ['waiting','ready_reload','ready_training','ready_resume'] for j in jobs):
            return None
        if any(j['status'] in ['training','preflight','reload'] and j.get('gpu')==2 for j in jobs):
            return None
    minimum=8500 if backbone=='onevision' else 16000
    raw=subprocess.check_output(['nvidia-smi',
        '--query-gpu=index,uuid,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    proc=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader,nounits'],text=True)
    busy={line.split(',')[0].strip() for line in proc.splitlines() if line.strip()}
    for line in raw.splitlines():
        number,uuid,free,used,util=[x.strip() for x in line.split(',')]
        if (int(number) in candidates and uuid not in busy and int(free)>=minimum
                and int(used)<100 and int(util)==0):return int(number)
    return None


def source_check(binding, frame):
    assert sha(__file__) == binding['launcher_sha256']
    expected = binding['training_source_sha256']
    frame_relative = frame.relative_to(ROOT).as_posix()
    required = {frame_relative+'/'+p.name for p in frame.glob('*.py')}
    required.add(frame_relative+'/execution_policy.json'); required.update(HELPERS)
    assert set(expected) == required
    for name, digest in expected.items():
        path = Path(name); assert not path.is_absolute() and '..' not in path.parts
        assert sha(ROOT/path) == digest, name
    descriptor = binding['inherited_formal_admission']
    inherited_path = Path(descriptor['path'])
    if not inherited_path.is_absolute(): inherited_path = ROOT/inherited_path
    assert sha(inherited_path) == descriptor['sha256']
    old = read(inherited_path)
    assert old['both_backbone_GPU_and_reload_passed'] and old['rolling_latest_and_branch_recovery_CPU_passed']
    assert old['manifest_sha256'] == MANIFEST_SHA
    # The source revision is I/O only. Keep every original method/kernel and
    # author helper byte exact; do not substitute the pct10 dynamic algorithm.
    for name, digest in old['training_source_sha256'].items():
        path = Path(name)
        if name in HELPERS:
            assert sha(ROOT/path) == digest
        elif path.suffix == '.py' or path.name == 'execution_policy.json':
            if path.name not in IO_CHANGES: assert sha(frame/path.name) == digest, name
    policy = read(frame/'execution_policy.json')
    assert policy['seed'] == 42 and policy['epochs'] == 1 and policy['methods'] == ['FOREVER']
    assert policy['conditions']['pct01']['manifest_sha256'] == MANIFEST_SHA
    assert 'FOREVER' in policy['checkpoint_all_stages_methods']
    return {p.name: sha(p) for p in frame.glob('*.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=['qwen'], default='qwen')
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args()
    assert os.getuid() == 1004 and str(ROOT).startswith('/data1/liyapeng/')
    assert args.binding.is_absolute()
    binding = read(args.binding); frame = ROOT/'exp'/FRAME_NAME
    source_map = source_check(binding, frame)
    previous = binding['prior_waiter']; current = identity(previous['pid'])
    assert current is None or current['startticks'] != int(previous['startticks']), 'Old CPU waiter still owns its queue'
    old_state_path = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v3/formal_queue/qwen/state.json'
    old_state = read(old_state_path)
    assert old_state['phase'] == 'healthy_wait_for_free_baseline_slot' and old_state.get('worker') is None
    folder = ROOT/'summary/stage_checkpoint_dispatch_20261027/pct01_forever/formal_queue/qwen'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder/'launcher.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (folder/'state.json').exists(), 'Inspect the existing child; never automatically restart this waiter'
    write(folder/'OWNER.json', dict(worker=identity(os.getpid()),binding_sha256=sha(args.binding),
        prior_waiter=previous,old_queue_state_sha256=sha(old_state_path),GPU_signals=0))

    def wait_gpu(phase):
        while True:
            gpu = available('qwen')
            if gpu is not None: return gpu
            write(folder/'state.json',dict(phase='healthy_wait_for_free_baseline_slot',next_phase=phase,
                backbone='qwen',worker=None,GPU_candidates=[2],binding_sha256=sha(args.binding)))
            time.sleep(15)

    def execute(phase, run, script, options):
        gpu = wait_gpu(phase); assert gpu == 2
        assert source_check(binding, frame) == source_map
        env = dict(os.environ,CUDA_VISIBLE_DEVICES='2',LLMCL_HOST='new3090',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
        command = [str(ROOT/'.vision-env/bin/python'),'-u',str(frame/script),*options,'--run-dir',str(run)]
        log = folder/f'{phase}_{time.time_ns()}.log'
        with log.open('xb') as stream:
            child = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            write(folder/'state.json',dict(phase=phase,backbone='qwen',GPU=gpu,worker=identity(child.pid),
                run=str(run),binding_sha256=sha(args.binding),command=command))
            code = child.wait()
        write(folder/f'{phase}_EXIT_{child.pid}.json',dict(returncode=code,pid=child.pid,run=str(run),unix=time.time()))
        if code:
            write(folder/'NEEDS_ATTENTION.json',dict(phase=phase,exit_code=code,run=str(run),no_auto_retry=True))
            raise RuntimeError(f'{phase} failed {code}; preserve original evidence')

    gate = frame/'runs/qwen/pct01/FOREVER'/time.strftime('preflight_checkpoint_%Y%m%d_%H%M%S')
    gate.mkdir(parents=True,exist_ok=False)
    execute('preflight',gate,'run.py',['--method','FOREVER','--condition','pct01','--preflight-only'])
    execute('reload',gate,'checkpoint_probe.py',[])
    preflight = read(gate/'gpu_preflight.json'); reload = read(gate/'checkpoint_reload_proof.json')
    assert preflight['passed'] and preflight['task_ids'] == list(range(1,20))
    assert reload['passed'] and reload['fresh_process'] and reload['finite_backward']
    for proof in (preflight,reload):
        assert proof['execution_source_sha256'] == source_map and proof['manifest_sha256'] == MANIFEST_SHA
    names = ('FOREVER_REPLAY_GPU_GATE.json','gpu_preflight.json','checkpoint_reload_proof.json')
    assert read(gate/names[0])['not_formal_results']
    write(folder/'SOURCE_MATCHED_QWEN_GATE.json',dict(passed=True,run=str(gate),
        proof_sha256={name:sha(gate/name) for name in names},execution_source_sha256=source_map,
        fresh_formal_model_required=True,inherited_admission_sha256=binding['inherited_formal_admission']['sha256'],
        only_existing_Qwen_preflight_and_reload=True))
    run = frame/'runs/qwen/pct01/FOREVER'/time.strftime('training_%Y%m%d_%H%M%S')
    assert not run.exists(); resume = False
    while True:
        options = ['--method','FOREVER','--condition','pct01']+(['--resume'] if resume else [])
        execute('resume' if resume else 'training',run,'run.py',options)
        status = read(run/'run_status.json')
        if status['status'] == 'completed':
            assert status['completed_stage'] == 19
            write(folder/'TRAINING_PROCESS_COMPLETE.json',dict(run=str(run),
                requires_independent_19_74_21454_verification_before_publication=True,not_yet_verified_result=True))
            return
        assert status['status'] == 'paused_checkpoint'
        write(folder/'state.json',dict(phase='paused_checkpoint',run=str(run),worker=None,
            completed_stage=status['completed_stage'],pause_receipt_sha256=status['pause_receipt_sha256']))
        while not (run/'RESUME_REQUEST.json').exists(): time.sleep(15)
        assert read(run/'RESUME_REQUEST.json')['pause_receipt_sha256'] == status['pause_receipt_sha256']
        # The runner revalidates full pause/source/helper metadata and consumes
        # the request itself. Nothing automatically resumes a failed child.
        resume = True


if __name__ == '__main__': main()
