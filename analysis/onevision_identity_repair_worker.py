"""New GPU entry point for the aux-ID repair; original workers stay immutable."""
import argparse
import importlib
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback

import onevision_feature_identity_repair as repair

ROOT, BASE = repair.ROOT, repair.REPAIR
FUNCTIONAL = BASE/'functional_router'
ALLOWED = {'onevision_functional_router_probe.py', 'onevision_warm_start_trial.py',
           'onevision_learned_reuse_trial.py', 'onevision_warm_quality_probe.py',
           'onevision_cold_data_factorial.py'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    entry = Path(args.worker)
    assert entry.parent == Path('analysis') and entry.name in ALLOWED
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '1'
    original_args = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    auth = json.loads((BASE/'AUTHORIZATION.json').read_text())
    for name, expected in auth['new_source_sha256'].items():
        assert repair.sha(ROOT/name) == expected
    repair.install(gpu_gate=True)
    if entry.name == 'onevision_functional_router_probe.py':
        options = argparse.ArgumentParser()
        options.add_argument('--scope', choices=['full'], required=True)
        options.add_argument('--view', choices=['quality_hard', 'quality_soft'], required=True)
        options = options.parse_args(original_args)
        worker = importlib.import_module(entry.stem)
        worker.REPORT = FUNCTIONAL
        spec, *_ = worker.plan()
        assert json.loads((FUNCTIONAL/'CPU_PLAN_CHECK.json').read_text())['spec'] == spec
        name = 'full_'+options.view
        pointer = FUNCTIONAL/(name+'_attempt.json')
        assert not pointer.exists(), 'One attempt only; never reuse failed partial predictions'
        run = ROOT/'exp/onevision_functional_router_20261018/identity_repair_v1'/name/time.strftime('%Y%m%d_%H%M%S')
        run.mkdir(parents=True)
        worker.write(pointer, {'spec': spec, 'run': str(run.relative_to(ROOT))})
        worker.write(run/'protocol.json', spec)
        worker.run_probe('full', options.view)
    else:
        sys.argv = [str(ROOT/entry), *original_args]
        runpy.run_path(str(ROOT/entry), run_name='__main__')
    assert repair.RUNS, 'Worker did not run a repaired experiment'
    receipts = []
    for run in repair.RUNS:
        path = run/'feature_identity_repair_execution.json'
        record = json.loads(path.read_text())
        assert record['GPU_gate']['passed']
        receipts.append({'run': str(run.relative_to(ROOT)), 'execution_receipt_sha256': repair.sha(path)})
    folder = BASE/'worker_success_receipts'
    folder.mkdir(exist_ok=True)
    (folder/(str(os.getpid())+'.json')).write_text(json.dumps({'worker': args.worker, 'arguments': original_args,
        'GPU1_only': True, 'runs': receipts, 'repair_sources': auth['new_source_sha256']}, indent=2)+'\n')


if __name__ == '__main__':
    try:
        main()
    except Exception:
        BASE.mkdir(parents=True, exist_ok=True)
        (BASE/('worker_'+str(os.getpid())+'_needs_attention.json')).write_text(json.dumps({
            'pid': os.getpid(), 'traceback': traceback.format_exc(), 'no_blind_retry': True}, indent=2)+'\n')
        raise
