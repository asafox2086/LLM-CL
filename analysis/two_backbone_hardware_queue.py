"""Defer the native-BF16 OneVision method until prioritized Qwen jobs yield.

CPU-only coordinator. It never changes running Python sources or interrupts a GPU.
"""
import fcntl
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / 'exp/two_backbone_native_cl_20261017'
REPORT = ROOT / 'summary/two_backbone_native_cl_20261017'


def read(p):
    return json.loads(p.read_text())


def write(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)


def main():
    lock = (REPORT / 'hardware_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = REPORT / 'hardware_queue.json'
    if status.exists() and read(status).get('phase') == 'completed':
        return
    while True:
        qwen = read(REPORT / 'new3090_run_status.json')['jobs']
        pending = {k: v for k, v in qwen.items() if v['condition'] == 'pct01' and
                   v['status'] != 'completed' and not v['status'].startswith('failed')}
        if not pending:
            break
        live = [v['pid'] for v in pending.values() if v.get('pid') and
                (Path('/proc') / str(v['pid']) / 'cmdline').exists()]
        write(status, {'phase': 'waiting_Qwen_pct01_priority', 'pending': list(pending),
                       'actual_live_worker_pids': live, 'no_GPU_allocated_by_coordinator': True})
        time.sleep(30)
    service = 'llmcl-onevision-smo-lane.service'
    state_path = REPORT / 'new3090_onevision_run_status.json'
    if not state_path.exists():
        write(state_path, {'experiment_id': EXP.name, 'host': 'new3090_onevision',
              'backbone': 'onevision', 'jobs': {'pct01:SMoLoRA': {'condition': 'pct01',
              'method': 'SMoLoRA', 'host': 'new3090_onevision', 'status': 'waiting'}}})
    unit = Path.home() / '.config/systemd/user' / service
    unit.write_text('[Unit]\nDescription=NativeBF16 OneVision SMoLoRA lane\nAfter=default.target\n'
        '[Service]\nType=simple\nWorkingDirectory=' + str(ROOT) + '\n'
        'Environment=LLMCL_HOST=new3090_onevision LLMCL_BACKBONE=onevision '
        'PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync\n'
        'ExecStart=/usr/bin/nohup ' + str(ROOT / '.vision-env/bin/python') + ' -u ' + str(EXP / 'pipeline.py') + '\n'
        'StandardInput=null\nStandardOutput=append:' + str(REPORT / 'onevision_smo_lane.log') + '\n'
        'StandardError=inherit\nKillMode=process\nRestart=on-failure\nRestartSec=30\n')
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'start', service], check=True)
    while True:
        job = read(state_path)['jobs']['pct01:SMoLoRA']
        if job['status'].startswith('failed'):
            raise RuntimeError('OneVision SMoLoRA admission/run failed; preserve evidence, no automatic retry')
        if job['status'] == 'completed':
            actual = read(ROOT / job['run_dir'] / 'run_status.json')
            assert actual['status'] == 'completed' and actual['completed_stage'] == 19
            for stage in range(1, 20):
                assert read(ROOT / job['run_dir'] / 'validity' / f'stage_{stage:02d}.json')['passed']
            break
        write(status, {'phase': 'nativeBF16_OneVision_SMoloRA_lane', 'job': job})
        time.sleep(30)
    # No more jobs exist in the dedicated lane. Preserve the method graph/result.
    subprocess.run(['systemctl', '--user', 'stop', service], check=True)
    failed = [k for k, value in read(REPORT / 'new3090_run_status.json')['jobs'].items()
              if value['condition'] == 'pct01' and value['status'].startswith('failed')]
    if failed:
        raise RuntimeError('Review failed Qwen1% methods before admitting full queue: ' + repr(failed))
    policy_path = EXP / 'execution_policy.json'
    policy = read(policy_path)
    policy['hosts']['new3090']['conditions']['full'] = [3]
    primary = 'llmcl-two-backbone-native.service'
    assert subprocess.check_output(['systemctl', '--user', 'show', primary,
                                    '-p', 'KillMode', '--value'], text=True).strip() == 'process'
    subprocess.run(['systemctl', '--user', 'stop', primary], check=True)
    try:
        write(policy_path, policy)
        ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', '-i',
               str(ROOT / '.deployment/old_pct01_id_ed25519'), '-p', '8666',
               'liyapeng_grp@10.254.29.8', '/data2/liyapeng_grp/program/LLMCL/.vision-env/bin/python -']
        code = "from pathlib import Path;Path('/data2/liyapeng_grp/program/LLMCL/exp/two_backbone_native_cl_20261017/execution_policy.json').write_text(" + repr(json.dumps(policy, ensure_ascii=False, indent=2) + '\n') + ")"
        subprocess.run(ssh, input=code, text=True, capture_output=True, check=True)
    finally:
        subprocess.run(['systemctl', '--user', 'start', primary], check=True)
    write(status, {'phase': 'completed', 'OneVision_SMoloRA': job,
                   'newGPU3_full_Qwen_enabled': True, 'newGPU0_GPU1_unchanged': True})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        write(REPORT / 'hardware_queue.json', {'phase': 'needs_attention', 'error': repr(exc)})
        raise
