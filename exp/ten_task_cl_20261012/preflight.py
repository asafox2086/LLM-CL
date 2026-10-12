"""Record actual launch readiness without allocating GPUs or starting training."""
import csv
import io
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def main():
    root = Path(__file__).resolve().parents[2]
    now = datetime.now(ZoneInfo('Asia/Shanghai'))
    run = root/'exp/ten_task_cl_20261012/preflight'/now.strftime('%Y%m%d_%H%M%S')
    run.mkdir(parents=True, exist_ok=False)
    reports = root/'summary/ten_task_cl_20261012'
    selection = json.loads((root/'code/method_selection.json').read_text())
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'], text=True, capture_output=True)
    gpus = []
    if query.returncode == 0:
        for row in csv.reader(io.StringIO(query.stdout)):
            if not row:
                continue
            index, name, total, used, utilization = [field.strip() for field in row]
            total, used, utilization = int(total), int(used), int(utilization)
            gpus.append({'index': int(index), 'name': name, 'total_mib': total,
                         'used_mib': used, 'utilization_percent': utilization,
                         'available_for_preflight': used < 1024 and utilization < 10})
    available = [gpu['index'] for gpu in gpus if gpu['available_for_preflight']]
    final_manifest = root/'data/TenTaskCL/final_task_manifest.json'
    reasons = []
    if query.returncode:
        reasons.append('nvidia-smi failed: '+query.stderr.strip())
    elif not available:
        reasons.append('No idle GPU: do not interrupt or contend with other users training.')
    if not final_manifest.is_file():
        reasons.append('Final ten-task sample manifest absent; legal-image and art input protocols and fixed selections remain unresolved.')
    else:
        tasks = json.loads(final_manifest.read_text()).get('tasks', [])
        if len(tasks) != 10 or any(not task.get('ready_for_formal_training') for task in tasks):
            reasons.append('Final manifest does not certify all ten tasks ready for formal training.')
    adapter_status_path = root/'exp/ten_task_cl_20261012/adapter_status.json'
    adapter_status = json.loads(adapter_status_path.read_text()) if adapter_status_path.is_file() else {}
    methods = []
    for name in selection['first_adaptation_batch']:
        source = root/'code'/name/'source_manifest.json'
        manifest = json.loads(source.read_text())
        adapter = adapter_status.get(name, {})
        verified = bool(adapter.get('ten_task_adaptation_verified') and adapter.get('resume_verified'))
        methods.append({'method': name, 'source_status': manifest['status'],
                        'source_revision': manifest.get('revision'), 'training_started': False,
                        'ten_task_adaptation_verified': verified, 'stage_completed': 0,
                        'status': 'ready_for_launch' if verified else 'awaiting_ten_task_adapter_and_resume_verification'})
    if any(not method['ten_task_adaptation_verified'] for method in methods):
        reasons.append('Author methods have source snapshots, but some ten-task training adapters or per-method resume checks are unverified.')
    status = {'experiment_id': 'ten_task_cl_20261012', 'updated_at': now.isoformat(),
              'status': 'preflight_blocked' if reasons else 'ready_to_launch', 'training_started': False,
              'preflight_log': str((run/'preflight.log').relative_to(root)),
              'gpu_snapshot': gpus, 'available_gpus': available,
              'methods': methods, 'blockers': reasons,
              'stopped_other_processes': False, 'formal_task_manifest_exists': final_manifest.is_file()}
    rendered = json.dumps(status, ensure_ascii=False, indent=2)+'\n'
    (run/'preflight.json').write_text(rendered)
    (run/'preflight.log').write_text('Launch readiness check; no training started.\n'+rendered)
    (reports/'run_status.json').write_text(rendered)
    lines = ['# 本轮执行状态', '', f'更新时间：{now.isoformat()}', '',
             '已执行启动预检，未启动正式训练。', '', '## 阻塞项', '']
    lines.extend('- '+reason for reason in reasons)
    lines += ['', '日志：`'+status['preflight_log']+'`。', '',
              '四卡并行优先用于不同方法/种子；只有已锁定数据、通过适配和恢复验证的运行才可调度。',
              '不会停止其他用户进程，也不会用历史分类实验或简化方法替代本轮方法。', '']
    (reports/'report.md').write_text('\n'.join(lines))
    print(rendered)


if __name__ == '__main__':
    main()
