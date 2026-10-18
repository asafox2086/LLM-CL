"""Actual worker imports/plans, no GPU, optimizer or moving TEST outputs."""
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import onevision_osft_budget_trial as worker


def main():
    assert not torch.cuda.is_initialized()
    spec, values = worker.plan()
    assert spec == worker.read(worker.REPORT/'CPU_PLAN.json')['spec']
    manifest, data, domains, tests, full, native, proof = values
    assert sum(len(data[t, 'train']) for t in range(1, 20)) == 1313
    assert sum(map(len, full.values())) == 21454 and sum(map(len, tests.values())) == 500
    assert sum(len(ids) for ids in spec['generation_ids'].values()) == 155
    assert set(spec['train_ids']) == {str(t) for t in range(1, 20)}
    for t in range(1, 20):
        assert spec['train_ids'][str(t)] == [row['id'] for row in data[t, 'train']]
    reference = worker.read(native/'protocol.json')
    assert spec['generation_ids'] == reference['generation_ids']
    assert sum((len(data[t, 'train'])+15)//16 for t in range(1, 20)) == 93
    assert spec['trainable_parameters'] == 540672 and spec['S_low_frozen']
    for gpu in ['0', '1']:
        source = ("import os,sys; os.environ['CUDA_VISIBLE_DEVICES']="+repr(gpu)+"; "
                  "sys.path.insert(0,"+repr(str(worker.ROOT/'analysis'))+"); "
                  "import onevision_osft_budget_trial as w; "
                  "w.plan(); assert os.environ['CUDA_VISIBLE_DEVICES']=="+repr(gpu)+"; "
                  "assert not w.torch.cuda.is_initialized()")
        subprocess.run([sys.executable, '-c', source], check=True)
    mechanics = worker.read(worker.REPORT/'CPU_MECHANICS.json')
    for rel, expected in mechanics['source_sha256'].items():
        assert worker.digest(worker.ROOT/rel) == expected
    names = ['onevision_osft_budget_core.py', 'onevision_osft_budget_cpu.py',
             'onevision_osft_budget_trial.py', 'onevision_osft_budget_entry.py',
             'onevision_osft_budget_worker_check.py']
    result = dict(CPU_only=True, checks=5, source_sha256={
        'analysis/'+n: worker.digest(worker.ROOT/'analysis'/n) for n in names},
        actual_worker_import_preserves_GPU_env=['0', '1'], initial_and_final_frozen_ids_exact=True,
        actual_expected_positive_updates=93, CUDA_initialized=False,
        GPU_forward_backward_or_QA_performance=False)
    worker.write(worker.REPORT/'CPU_WORKER_CHECK.json', result)
    assert not torch.cuda.is_initialized()
    print(json.dumps(result))


if __name__ == '__main__':
    main()
