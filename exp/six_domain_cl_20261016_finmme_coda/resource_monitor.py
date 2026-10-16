"""Byte-valued process RAM / GPU memory measurements, independent of score logging."""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import torch


class ResourceMonitor:
    def __init__(self, run, interval=5):
        self.run = Path(run)
        self.run.mkdir(parents=True, exist_ok=True)
        self.path = self.run / 'resources.jsonl'
        self.lock = threading.Lock()
        self.done = threading.Event()
        self.stage = 0
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.done.wait(5):
            self.snapshot('periodic', self.stage, synchronize=False)

    def snapshot(self, phase, stage=0, model=None, synchronize=True):
        if synchronize and torch.cuda.is_initialized():
            torch.cuda.synchronize()
        values = {}
        for file in ['status', 'smaps_rollup']:
            try:
                for line in Path(f'/proc/{os.getpid()}/{file}').read_text().splitlines():
                    key, _, value = line.partition(':')
                    if key in ['VmRSS', 'VmHWM', 'VmSwap', 'Pss', 'Private_Clean', 'Private_Dirty']:
                        values[key + '_bytes'] = int(value.split()[0]) * 1024
            except OSError:
                pass
        record = {'phase': phase, 'stage': stage, 'unix': time.time(), 'pid': os.getpid(),
                  'visible_gpu': os.environ.get('CUDA_VISIBLE_DEVICES'), **values}
        record['cpu_uss_bytes'] = values.get('Private_Clean_bytes', 0) + values.get('Private_Dirty_bytes', 0)
        if torch.cuda.is_initialized():
            record.update(gpu_allocated_bytes=torch.cuda.memory_allocated(),
                          gpu_reserved_bytes=torch.cuda.memory_reserved(),
                          gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
        try:
            query = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_memory',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=3)
            matches = [int(line.split(',')[1]) * 1024**2 for line in query.stdout.splitlines()
                       if line.split(',')[0].strip() == str(os.getpid())]
            record['gpu_process_nvml_bytes'] = sum(matches) if matches else None
        except (OSError, ValueError, subprocess.TimeoutExpired):
            record['gpu_process_nvml_bytes'] = None
        if model is not None:
            record['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            record['trainable_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters() if p.requires_grad)
            record['model_parameter_bytes'] = sum(p.numel()*p.element_size() for p in model.parameters())
        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
        return record

    def finish(self, model):
        self.done.set()
        self.thread.join(timeout=6)
        self.snapshot('run_finished', self.stage, model)
        rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        before = next(x for x in rows if x['phase'] == 'training_before')
        after = next(x for x in rows if x['phase'] == 'training_after')
        keys = ['VmRSS_bytes', 'Pss_bytes', 'cpu_uss_bytes', 'gpu_allocated_bytes',
                'gpu_reserved_bytes', 'gpu_process_nvml_bytes']
        summary = {'unit': 'bytes', 'before': before, 'after': after,
                   'growth_bytes': {k: after[k]-before[k] if after.get(k) is not None and before.get(k) is not None else None for k in keys},
                   'observed_peak_bytes': {k: max(x[k] for x in rows if x.get(k) is not None) if any(x.get(k) is not None for x in rows) else None for k in keys},
                   'note': 'Process memory; CUDA allocator, reserved cache and NVML process allocation are separate. Before/after retain method state and exclude optimizer after cleanup.'}
        (self.run / 'memory_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
        return summary
