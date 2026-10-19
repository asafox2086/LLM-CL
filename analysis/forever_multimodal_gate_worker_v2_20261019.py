"""No-retry, sequential real-backbone FOREVER gates on an otherwise idle GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PORT = ROOT/'exp/two_backbone_forever_20261019_v2'
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v2'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def identity(pid):
    p = Path(f'/proc/{pid}')
    try:
        stat = (p/'stat').read_text()
        return {'pid': pid, 'startticks': int(stat[stat.rfind(')')+2:].split()[19]),
                'cmdline': (p/'cmdline').read_bytes().replace(b'\0',b' ').decode()}
    except FileNotFoundError:
        return None


def idle(gpu):
    rows = subprocess.check_output([
        'nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True)
    row = next([int(x.strip()) for x in line.split(',')]
               for line in rows.splitlines() if int(line.split(',')[0]) == gpu)
    assert row[1] >= 8500 and row[2] < 100 and row[3] == 0, row
    assignments = subprocess.check_output([
        'nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True)
    uuids = subprocess.check_output([
        'nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True)
    uuid = next(line.split(',')[1].strip() for line in uuids.splitlines()
                if int(line.split(',')[0]) == gpu)
    assert not any(line.split(',')[0].strip() == uuid for line in assignments.splitlines())
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0], required=True)
    args = parser.parse_args()
    assert str(ROOT).startswith('/data2/liyapeng_grp/'), 'This admission is OLD GPU0 only'
    plan = json.loads((REPORT/'GPU_GATE_PLAN.json').read_text())
    assert plan['status'] == 'FROZEN_FOR_GPU_GATE_ONLY_NOT_FORMAL_ADMISSION'
    for name, digest in plan['source_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
    attempt = REPORT/'gate_attempt'
    attempt.mkdir(exist_ok=False)
    protected = {str(pid): identity(pid) for pid in [341019,342152,826894]}
    assert all(protected.values()), 'Recheck baseline identities before admitting work'
    write(attempt/'PROTECTED_BEFORE.json', protected)
    write(attempt/'OWNER.json', identity(os.getpid()))
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(args.gpu),
           'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
    try:
        for backbone, host in [('onevision','old2080'), ('qwen','old2080_qwen_gate')]:
            free = idle(args.gpu)
            run = PORT/'runs/gates'/backbone/time.strftime('%Y%m%d_%H%M%S')
            env['LLMCL_HOST'] = host
            for phase, script, options in [
                ('preflight','run.py',['--method','FOREVER','--condition','pct01','--preflight-only']),
                ('reload','checkpoint_probe.py',[]),
            ]:
                for name, digest in plan['source_sha256'].items():
                    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest, name
                # Previous child has exited; never compete with a new foreign job.
                idle(args.gpu)
                command = [str(ROOT/'.vision-env/bin/python'),'-u',str(PORT/script),
                           *options,'--run-dir',str(run)]
                log = attempt/f'{backbone}_{phase}.log'
                with log.open('xb') as stream:
                    child = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                                             stderr=subprocess.STDOUT)
                    time.sleep(.2)
                    write(attempt/'state.json', {
                        'phase': phase, 'backbone': backbone, 'GPU': args.gpu,
                        'worker': identity(child.pid), 'run': str(run),
                        'source_plan_sha256': hashlib.sha256((REPORT/'GPU_GATE_PLAN.json').read_bytes()).hexdigest(),
                    })
                    code = child.wait()
                if code:
                    raise RuntimeError(f'{backbone}/{phase} exited {code}; preserve {log}')
            for name in ['FOREVER_REPLAY_GPU_GATE.json','gpu_preflight.json',
                         'checkpoint_reload_proof.json']:
                obj = json.loads((run/name).read_text())
                assert obj.get('passed', obj.get('not_formal_results')) is True
            write(attempt/f'{backbone}_COMPLETE.json', {
                'run': str(run), 'preflight_and_fresh_reload': True,
                'not_formal_results': True,
                'proof_sha256': {name: hashlib.sha256((run/name).read_bytes()).hexdigest()
                                  for name in ['FOREVER_REPLAY_GPU_GATE.json','gpu_preflight.json',
                                               'checkpoint_reload_proof.json']},
            })
        write(attempt/'BOTH_GPU_GATES_COMPLETE.json', {'passed':True,
              'formal_training_launched':False,'formal_publication_verifier_still_required':True})
    except Exception as error:
        write(attempt/'NEEDS_ATTENTION.json', {'error':repr(error),'no_auto_retry':True,
                                              'not_performance_evidence':True})
        raise
    finally:
        after = {pid: identity(int(pid)) for pid in protected}
        write(attempt/'PROTECTED_AFTER.json', after)
        # A baseline may complete naturally; never restart or signal it here.
        write(attempt/'BASELINE_IDENTITY_CHECK.json', {
            pid: {'before': protected[pid], 'after': after[pid],
                  'unchanged_if_alive': after[pid] is None or after[pid] == protected[pid]}
            for pid in protected})


if __name__ == '__main__':
    main()
