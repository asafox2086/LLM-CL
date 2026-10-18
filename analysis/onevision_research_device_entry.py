"""Versioned physical GPU0/1 entry; immutable algorithms differ only in guard.

Set the physical device before importing torch. No image, QA, optimizer, seed,
allocation, score, or weight logic is altered. Parent source bytes stay frozen.
"""
import argparse
import ast
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def inventory():
    raw = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    cards = {}
    for line in raw.splitlines():
        index, uuid, memory, util = map(str.strip, line.split(','))
        cards[int(index)] = {'uuid': uuid, 'memory_MiB': int(memory), 'utilization': int(util), 'owners': []}
    raw = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid',
                                   '--format=csv,noheader,nounits'], text=True)
    for line in raw.splitlines():
        pid, uuid = map(str.strip, line.split(','));proc = Path('/proc')/pid
        try:
            uid = proc.stat().st_uid
        except FileNotFoundError:
            continue
        for info in cards.values():
            if info['uuid'] == uuid:
                info['owners'].append({'pid': int(pid), 'uid': uid})
    return cards


def select(cards, uid):
    ours = {g for g, info in cards.items() if any(p['uid'] == uid for p in info['owners'])}
    if ours & {0, 1}:
        return None  # One research stream; never duplicate it on the other card.
    for gpu in (1, 0):
        c = cards[gpu]
        if not c['owners'] and c['memory_MiB'] < 500 and c['utilization'] < 10 and len(ours|{gpu}) <= 3:
            return gpu
    return None


def compiled_function(module, name, gpu):
    tree = ast.parse(Path(module.__file__).read_text())
    old = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    node = copy.deepcopy(old)
    first = node.body[0]
    assert isinstance(first, ast.Assert) and isinstance(first.test, ast.Compare)
    assert len(first.test.comparators) == 1 and first.test.comparators[0].value == '1'
    first.test.comparators[0] = ast.Constant(str(gpu))
    reverted = copy.deepcopy(node);reverted.body[0].test.comparators[0] = ast.Constant('1')
    assert ast.dump(old, include_attributes=False) == ast.dump(reverted, include_attributes=False)
    namespace = dict(vars(module))
    return ast.fix_missing_locations(node), namespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    p.add_argument('--kind', choices=['cache', 'factorial'], required=True)
    p.add_argument('--phase', choices=['build', 'full'])
    p.add_argument('--view')
    p.add_argument('--case-index', type=int)
    args = p.parse_args()
    auth = json.loads((ROOT/'summary/onevision_diagnostics_20261016/functional_state_cache_20261018/v1/flex_queue/AUTHORIZATION.json').read_text())
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == auth['new_device_sources']['analysis/onevision_research_device_entry.py']
    assert auth['research_allowed_GPUs'] == [0, 1] and auth['max_our_GPU_cards'] == 3
    # Check actual occupancy immediately before importing/initializing CUDA.
    while True:
        cards = inventory()
        selected = select(cards, os.getuid())
        ours = {g for g, c in cards.items() if any(o['uid'] == os.getuid() for o in c['owners'])}
        c = cards[args.gpu]
        if not (ours & {0, 1}) and not c['owners'] and c['memory_MiB'] < 500 and c['utilization'] < 10 and len(ours|{args.gpu}) <= 3:
            break
        print(json.dumps({'phase': 'waiting_device_entry', 'requested_GPU': args.gpu,
                          'our_GPU_cards': sorted(ours), 'no_CUDA_model_started': True}), flush=True)
        time.sleep(15)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_NUM_THREADS'] = '6'
    module_name = 'onevision_state_cache_trial' if args.kind == 'cache' else 'onevision_cold_data_factorial'
    module = importlib.import_module(module_name)
    name = 'run_trial' if args.kind == 'cache' else 'run_case'
    node, namespace = compiled_function(module, name, args.gpu)
    parent_class = namespace['CacheExperiment' if args.kind == 'cache' else 'RepairedExperiment']

    class DeviceExperiment(parent_class):
        def __init__(self, manifest, run):
            receipt = {'physical_GPU': args.gpu, 'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                'local_CUDA_device': 0, 'our_GPU_cards_before': sorted(ours), 'max_our_GPU_cards': 3,
                'admission_inventory': cards, 'only_guard_literal_changed': True,
                'algorithm_source_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
                'device_entry_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'no_source_hot_edit': True, 'QA_and_training_and_scoring_unchanged': True}
            (run/'physical_device_execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
            super().__init__(manifest, run)
            assert module.torch.cuda.current_device() == 0
            receipt['actual_device_name'] = module.torch.cuda.get_device_name(0)
            receipt['actual_CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
            (run/'physical_device_execution.json').write_text(json.dumps(receipt, indent=2)+'\n')

    namespace['CacheExperiment' if args.kind == 'cache' else 'RepairedExperiment'] = DeviceExperiment
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(Path(__file__)), 'exec'), namespace)
    try:
        if args.kind == 'cache':
            assert args.phase in ('build', 'full')
            namespace[name](args.phase, args.view)
        else:
            assert args.case_index is not None and 0 <= args.case_index < 18
            spec, values = module.plan()
            assert json.loads((module.REPORT/'CPU_PLAN_CHECK.json').read_text())['spec'] == spec
            namespace[name](args.case_index, spec, values)
    except Exception:
        import traceback
        label = ('build' if args.phase == 'build' else 'full_'+str(args.view)) if args.kind == 'cache' else f'case_{args.case_index}'
        (module.REPORT/(label+'_needs_attention.json')).write_text(json.dumps({
            'traceback': traceback.format_exc(), 'physical_GPU': args.gpu, 'no_blind_retry': True}, indent=2)+'\n')
        raise


if __name__ == '__main__':
    main()
