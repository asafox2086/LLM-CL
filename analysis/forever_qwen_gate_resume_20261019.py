"""CPU-only operational continuation after a verified OneVision v2 gate.

Retain the exact original idle guard; wait for idle telemetry rather than
declaring a GPU failure. Never rerun the completed OneVision gate or change
frozen v2 training/optimizer/checkpoint sources.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT/'summary/two_backbone_pct01_20261017/forever_multimodal_20261019/v2'
PORT = ROOT/'exp/two_backbone_forever_20261019_v2'
spec = importlib.util.spec_from_file_location('frozen_forever_v2_guard',
    ROOT/'analysis/forever_multimodal_gate_worker_v2_20261019.py')
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def wait_idle(gpu, publish, idle=old.idle, sleep=time.sleep):
    while True:
        try:
            return idle(gpu)
        except AssertionError as error:
            publish({'phase':'wait_for_original_idle_guard', 'GPU':gpu,
                     'reason':repr(error), 'no_GPU_child_started':True})
            sleep(5)


def cpu_check():
    calls, waits, messages = [], [], []
    def fake(gpu):
        calls.append(gpu)
        if len(calls) < 3:
            raise AssertionError([0,11002,8,19])
        return [0,11002,8,0]
    result = wait_idle(0,messages.append,idle=fake,sleep=waits.append)
    assert result == [0,11002,8,0] and len(messages) == len(waits) == 2
    assert all(row['no_GPU_child_started'] for row in messages)
    return {'status':'CPU_WAIT_CONTROL_PASS','waits':2,
            'original_idle_guard_not_relaxed':True,'GPU_or_model':False,
            'OneVision_not_rerun':True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-check-only',action='store_true')
    args = parser.parse_args()
    if args.cpu_check_only:
        print(json.dumps(cpu_check()));return
    assert str(ROOT).startswith('/data2/liyapeng_grp/')
    plan = json.loads((REPORT/'GPU_GATE_PLAN.json').read_text())
    def sources():
        for name,digest in plan['source_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == digest,name
    sources()
    inherited = json.loads((REPORT/'gate_attempt/onevision_COMPLETE.json').read_text())
    assert inherited['preflight_and_fresh_reload'] and inherited['not_formal_results']
    inherited_run = Path(inherited['run'])
    for name,digest in inherited['proof_sha256'].items():
        assert hashlib.sha256((inherited_run/name).read_bytes()).hexdigest() == digest
    attempt = REPORT/'qwen_cpu_resume'
    attempt.mkdir(exist_ok=False)
    old.write(attempt/'OWNER.json',old.identity(os.getpid()))
    old.write(attempt/'INHERITED_ONEVISION_COMPLETE.json',inherited)
    protected = {str(pid):old.identity(pid) for pid in [341019,342152,826894]}
    assert all(protected.values())
    old.write(attempt/'PROTECTED_BEFORE.json',protected)
    run = PORT/'runs/gates/qwen'/time.strftime('%Y%m%d_%H%M%S')
    env = {**os.environ,'CUDA_VISIBLE_DEVICES':'0','LLMCL_HOST':'old2080_qwen_gate',
           'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1'}
    try:
        for phase,script,options in [
            ('preflight','run.py',['--method','FOREVER','--condition','pct01','--preflight-only']),
            ('reload','checkpoint_probe.py',[]),
        ]:
            wait_idle(0,lambda state:old.write(attempt/'state.json',state))
            sources()
            command=[str(ROOT/'.vision-env/bin/python'),'-u',str(PORT/script),
                     *options,'--run-dir',str(run)]
            with (attempt/f'qwen_{phase}.log').open('xb') as stream:
                child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,
                                       stderr=subprocess.STDOUT)
                time.sleep(.2)
                old.write(attempt/'state.json',{'phase':phase,'backbone':'qwen','GPU':0,
                          'worker':old.identity(child.pid),'run':str(run),
                          'inherited_OneVision_verified':True})
                code=child.wait()
            assert code == 0,f'Qwen {phase} failed {code}; preserve log, no retry'
        names=['FOREVER_REPLAY_GPU_GATE.json','gpu_preflight.json','checkpoint_reload_proof.json']
        for name in names:
            proof=json.loads((run/name).read_text())
            assert proof.get('passed',proof.get('not_formal_results')) is True
        old.write(attempt/'QWEN_COMPLETE.json',{'run':str(run),'not_formal_results':True,
                  'proof_sha256':{name:hashlib.sha256((run/name).read_bytes()).hexdigest()
                                  for name in names}})
        old.write(attempt/'BOTH_GPU_GATES_COMPLETE.json',{'passed':True,
                  'OneVision_inherited_same_frozen_v2_sources':True,
                  'Qwen_fresh_v2_completed':True,'formal_training_launched':False,
                  'formal_admission_still_required':True})
    except Exception as error:
        old.write(attempt/'NEEDS_ATTENTION.json',{'error':repr(error),'no_auto_retry':True})
        raise
    finally:
        old.write(attempt/'PROTECTED_AFTER.json',
                  {pid:old.identity(int(pid)) for pid in protected})


if __name__ == '__main__':
    main()
