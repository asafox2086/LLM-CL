"""Complete a fixed 2x2 structural intervention using two sealed CPU packages."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/width_component_split_20261021/v1'
PRIVATE = ROOT / 'exp/edge_width_component_split_20261021/v1'
PARENT = ROOT / 'summary/edge_deployment_20261019/structured_width_20261020/v1'
OLD_PRIVATE = ROOT / 'exp/edge_structured_width_20261020/v1'
CASES = ['vision_only', 'text_only']
SOURCES = ['edge_width_split_20261021.py', 'edge_width_split_report_20261021.py']
import edge_width_pilot_20261020 as inherited
from edge_width_core_20261020 import sites, parameter_digest
from edge_width_runtime_20261020 import WidthOneVision


sha, write = inherited.sha, inherited.write


def chosen(key, prefixes):
    return any(key.startswith(p + '.') for p in prefixes)


def replace_branch(core, kind, joint):
    """Reader supplies exact frozen joint tensors; never rerank or compensate."""
    import torch
    from torch import nn
    found = sites(core)
    prefixes = [name for name, (k, _, _) in found.items() if k == kind]
    assert len(prefixes) == (24 if kind == 'text' else 26) or len(prefixes) == 2
    for name in prefixes:
        module = core.get_submodule(name)
        members = ['gate_proj', 'up_proj', 'down_proj'] if kind == 'text' else ['fc1', 'fc2']
        for part in members:
            prefix = name + '.' + part
            weight = joint(prefix + '.weight')
            bias = joint(prefix + '.bias', optional=True)
            new = nn.Linear(weight.shape[1], weight.shape[0], bias=bias is not None,
                            device=weight.device, dtype=weight.dtype)
            new.weight = nn.Parameter(weight.clone(), requires_grad=False)
            if bias is not None:
                new.bias = nn.Parameter(bias.clone(), requires_grad=False)
            setattr(module, part, new)
        if kind == 'text':
            module.intermediate_size = module.down_proj.in_features
    if kind == 'text':
        core.config.text_config.intermediate_size //= 2
        core.config.llmcl_compensated_text_down_bias = True
    else:
        core.config.vision_config.intermediate_size //= 2
    return prefixes


def verify_origins(core, dense, joint, prefixes):
    import torch
    count = {'dense': 0, 'joint': 0}
    for name, parameter in core.named_parameters():
        origin = 'joint' if chosen(name, prefixes) else 'dense'
        expected = (joint if origin == 'joint' else dense)(name)
        assert parameter.dtype == expected.dtype and torch.equal(parameter, expected), name
        count[origin] += 1
    assert all(count.values())
    return count


def cpu_gate():
    import copy
    import torch
    from transformers import LlavaOnevisionConfig, Qwen2Config, SiglipVisionConfig
    from edge_width_core_20261020 import prune
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    config = LlavaOnevisionConfig(
        text_config=Qwen2Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                               num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                               tie_word_embeddings=True),
        vision_config=SiglipVisionConfig(hidden_size=16, intermediate_size=32,
                                        num_hidden_layers=2, num_attention_heads=2,
                                        image_size=28, patch_size=14, vision_use_head=False),
        image_token_index=63, image_grid_pinpoints=[[28, 28]])
    dense = WidthOneVision(copy.deepcopy(config)).eval()
    joint = WidthOneVision(copy.deepcopy(config)).eval()
    joint.load_state_dict(dense.state_dict(), strict=True)
    moments = {n: {'mean': torch.linspace(-.5, .5, o.in_features).double(),
                    'variance': torch.linspace(.01, 2, o.in_features).double(), 'positions': [3]}
               for n, (_, _, o) in sites(joint).items()}
    prune(joint, moments, 'fluctuation')
    states = [dense.state_dict(), joint.state_dict()]
    def reader(state):
        def get(name, optional=False):
            if optional:
                return state.get(name)
            return state[name]
        return get
    results = {}
    for kind in ['vision', 'text']:
        model = WidthOneVision(copy.deepcopy(config)).eval()
        model.load_state_dict(states[0], strict=True)
        prefixes = replace_branch(model, kind, reader(states[1]))
        origins = verify_origins(model, reader(states[0]), reader(states[1]), prefixes)
        ids, pixels = torch.tensor([[1, 2, 3]]), torch.randn(1, 3, 28, 28)
        with torch.inference_mode():
            text = model.language_model(ids).logits
            vision = model.vision_tower(pixels).last_hidden_state
        assert torch.isfinite(text).all() and torch.isfinite(vision).all()
        path = PRIVATE / 'gate_packages' / kind
        assert not path.exists()
        model.save_pretrained(path, safe_serialization=True)
        loaded = WidthOneVision.from_pretrained(path, local_files_only=True).eval()
        assert parameter_digest(model) == parameter_digest(loaded)
        with torch.inference_mode():
            assert torch.equal(text, loaded.language_model(ids).logits)
            assert torch.equal(vision, loaded.vision_tower(pixels).last_hidden_state)
        results[kind] = {'origin_tensors': origins, 'fresh_state_and_text_vision_exact': True}
    assert not torch.cuda.is_initialized()
    write(REPORT / 'CPU_GATE.json', {'conditions': results, 'CPU_only': True,
                                    'CUDA_initialized': False, 'not_full_model_or_quality_result': True})


def parent_inputs():
    inherited.verify_ready()
    prepared = json.loads((PARENT / 'prepare_COMPLETE.json').read_text())
    assert sha(OLD_PRIVATE / 'fluctuation_structural_masks.json') == prepared['packages']['fluctuation']['private_masks_sha256']
    for name in ['dense', 'fluctuation']:
        complete = json.loads((PARENT / (name + '_COMPLETE.json')).read_text())
        assert complete['complete'] and complete['fresh_parameter_digest_exact']
        assert complete['ready_sha256'] == sha(PARENT / 'READY.json')
        assert sha(OLD_PRIVATE / (name + '_predictions.json')) == complete['raw_predictions_sha256']
    paths = [PARENT / n for n in ['READY.json', 'PROTOCOL.md', 'prepare_COMPLETE.json',
             'dense_COMPLETE.json', 'fluctuation_COMPLETE.json', 'VERIFIED_RESULTS.json', 'OUTCOME.json']]
    paths += [OLD_PRIVATE / n for n in ['fluctuation_structural_masks.json',
                                      'dense_predictions.json', 'fluctuation_predictions.json']]
    for name in ['dense', 'fluctuation']:
        spec = prepared['packages'][name]
        for rel, info in spec['files'].items():
            path = ROOT / spec['package'] / rel
            assert sha(path) == info['sha256']
            paths.append(path)
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def admit():
    assert not (REPORT / 'READY.json').exists()
    gate = json.loads((REPORT / 'CPU_GATE.json').read_text())
    assert all(c['fresh_state_and_text_vision_exact'] for c in gate['conditions'].values())
    assert gate['CUDA_initialized'] is False
    write(REPORT / 'READY.json', {
        'sources': {n: sha(ROOT / 'analysis' / n) for n in SOURCES},
        'protocol_sha256': sha(REPORT / 'PROTOCOL.md'), 'gate_sha256': sha(REPORT / 'CPU_GATE.json'),
        'admission_review_sha256': sha(REPORT / 'ADMISSION_REVIEW.json'),
        'parent_files': parent_inputs(), 'cases': CASES, 'CPU_only': True,
        'case_budget_seconds': 7200, 'prepare_budget_seconds': 7200,
        'official_TEST_used': False, 'DEV_selection': False, 'new_optimizer_updates': 0})


def verify_ready():
    ready = json.loads((REPORT / 'READY.json').read_text())
    assert ready['sources'] == {n: sha(ROOT / 'analysis' / n) for n in SOURCES}
    assert ready['protocol_sha256'] == sha(REPORT / 'PROTOCOL.md')
    assert ready['gate_sha256'] == sha(REPORT / 'CPU_GATE.json')
    assert ready['admission_review_sha256'] == sha(REPORT / 'ADMISSION_REVIEW.json')
    for rel, expected in ready['parent_files'].items():
        assert sha(ROOT / rel) == expected, rel
    inherited.verify_ready()
    return ready


def safereader(handle):
    keys = set(handle.keys())
    def read(name, optional=False):
        if optional and name not in keys:
            return None
        return handle.get_tensor(name)
    return read


def prepare():
    torch = inherited.imports()
    from safetensors import safe_open
    from transformers import AutoProcessor
    ready = verify_ready()
    assert not (REPORT / 'prepare_STARTED.json').exists()
    write(REPORT / 'prepare_STARTED.json', {'pid': os.getpid(), 'startticks': Path('/proc/self/stat').read_text().split()[21]})
    tick = time.monotonic()
    parent = json.loads((PARENT / 'prepare_COMPLETE.json').read_text())
    dense_path = ROOT / parent['packages']['dense']['package']
    joint_path = ROOT / parent['packages']['fluctuation']['package']
    processor = AutoProcessor.from_pretrained(dense_path, local_files_only=True)
    core = WidthOneVision.from_pretrained(dense_path, torch_dtype='auto', attn_implementation='sdpa', local_files_only=True).eval()
    assert parameter_digest(core) == parent['packages']['dense']['parameter_digest']
    selected, rows = inherited.data()
    first = next(x for x in selected if x['role'] == 'calibration' and x['domain'] == 'medical')
    task, row = rows[first['id']]
    model = inherited.wrap(core)
    batch, _ = inherited.batch(model, processor, task, row)
    import runtime_utils
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        loss, _ = runtime_utils.answer_loss(model, batch, row['answer'], processor)
    assert float(loss) == parent['reference_NLL'], 'Inherited dense CPU reference changed'
    write(REPORT / 'REFERENCE_GATE.json', {'NLL': float(loss), 'expected': parent['reference_NLL'], 'exact': True})
    import onevision_bridge, weakref
    reference = weakref.ref(core)
    onevision_bridge.bind(None)
    model._llmcl_visual_cache.clear()
    del model, core, batch
    gc.collect()
    assert reference() is None, 'Reference model retained before package construction'
    packages = {}
    for name in CASES:
        assert time.monotonic() - tick < ready['prepare_budget_seconds']
        kind = 'vision' if name == 'vision_only' else 'text'
        core = WidthOneVision.from_pretrained(dense_path, torch_dtype='auto', attn_implementation='sdpa', local_files_only=True).eval()
        with safe_open(dense_path / 'model.safetensors', framework='pt', device='cpu') as dense, safe_open(joint_path / 'model.safetensors', framework='pt', device='cpu') as joint:
            prefixes = replace_branch(core, kind, safereader(joint))
            origins = verify_origins(core, safereader(dense), safereader(joint), prefixes)
        assert {p.dtype for p in core.parameters()} == {torch.float16}
        destination = PRIVATE / 'packages' / name
        assert not destination.exists()
        digest = parameter_digest(core)
        core.save_pretrained(destination, safe_serialization=True, max_shard_size='2GB')
        processor.save_pretrained(destination)
        shutil.copyfile(ROOT / 'analysis/edge_width_runtime_20261020.py', destination / 'runtime.py')
        write(destination / 'runtime_spec.json', {'entry': 'from runtime import WidthOneVision; WidthOneVision.from_pretrained(path, torch_dtype="auto", attn_implementation="sdpa", local_files_only=True)',
              'storage_dtype': 'float16', 'compute_autocast': 'CPU float16', 'server_proxy_not_phone_or_GPU': True})
        files = {str(p.relative_to(destination)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                 for p in sorted(destination.rglob('*')) if p.is_file()}
        packages[name] = {'package': str(destination.relative_to(ROOT)), 'files': files,
                         'actual_package_bytes': sum(x['bytes'] for x in files.values()),
                         'parameters': sum(p.numel() for p in core.parameters()), 'parameter_digest': digest,
                         'origin_tensor_counts': origins, 'modified_prefixes': prefixes,
                         'text_intermediate_size': core.config.text_config.intermediate_size,
                         'vision_intermediate_size': core.config.vision_config.intermediate_size}
        assert parent['packages']['fluctuation']['parameters'] < packages[name]['parameters'] < parent['packages']['dense']['parameters']
        del core
        gc.collect()
    assert not torch.cuda.is_initialized()
    write(REPORT / 'prepare_COMPLETE.json', {'packages': packages, 'reference_NLL': parent['reference_NLL'],
          'source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json'),
          'CPU_only': True, 'new_calibration_forwards': 0, 'reference_forward': 1,
          'seconds': time.monotonic() - tick})


def case(name):
    torch = inherited.imports()
    import psutil, runtime_utils
    from transformers import AutoProcessor
    ready = verify_ready()
    assert not (REPORT / (name + '_STARTED.json')).exists()
    write(REPORT / (name + '_STARTED.json'), {'pid': os.getpid(), 'startticks': Path('/proc/self/stat').read_text().split()[21]})
    tick = time.monotonic()
    prepared = json.loads((REPORT / 'prepare_COMPLETE.json').read_text())
    assert prepared['ready_sha256'] == sha(REPORT / 'READY.json')
    spec = prepared['packages'][name]
    folder = ROOT / spec['package']
    for rel, info in spec['files'].items():
        assert sha(folder / rel) == info['sha256']
    process = psutil.Process()
    before = inherited.memory(process)
    load_start = time.monotonic()
    core = WidthOneVision.from_pretrained(folder, torch_dtype='auto', attn_implementation='sdpa', local_files_only=True).eval()
    assert parameter_digest(core) == spec['parameter_digest']
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = inherited.wrap(core)
    gc.collect()
    loaded = inherited.memory(process)
    load_seconds = time.monotonic() - load_start
    selected, rows = inherited.data()
    warm = next(x for x in selected if x['role'] == 'calibration' and x['domain'] == 'medical')
    task, row = rows[warm['id']]
    batch, _ = inherited.batch(model, processor, task, row)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        loss, _ = runtime_utils.answer_loss(model, batch, row['answer'], processor)
    assert torch.isfinite(loss)
    del batch
    model._llmcl_visual_cache.clear()
    model._llmcl_visual_cache_bytes = 0
    gc.collect()
    records = []
    for item in [x for x in selected if x['role'] == 'probe']:
        assert time.monotonic() - tick < ready['case_budget_seconds']
        task, row = rows[item['id']]
        for speed in [True, False]:
            assert time.monotonic() - tick < ready['case_budget_seconds']
            records.append(inherited.measure(model, processor, task, row, speed))
        write(PRIVATE / (name + '_partial_predictions.json'), records)
        write(REPORT / (name + '_PROGRESS.json'), {'completed_QA': len(records) // 2, 'required': 5})
        print('PROBE_COMPLETE', name, item['id'], flush=True)
    assert parameter_digest(core) == spec['parameter_digest']
    assert not torch.cuda.is_initialized()
    path = PRIVATE / (name + '_predictions.json')
    write(path, records)
    public = [{k: v for k, v in x.items() if k not in ['prediction', 'references']} for x in records]
    write(REPORT / (name + '_COMPLETE.json'), {'complete': True, 'case': name,
          'source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json'),
          'prepare_sha256': sha(REPORT / 'prepare_COMPLETE.json'), 'package': spec,
          'fresh_parameter_digest_exact': True, 'probe_QA': 5, 'requests': public,
          'raw_predictions_sha256': sha(path), 'memory_before_load': before,
          'memory_after_load': loaded, 'load_seconds': load_seconds,
          'warmup_calibration_NLL': float(loss), 'seconds': time.monotonic() - tick,
          'CPU_only': True, 'teacher_saw_probes': True, 'not_independent_DEV': True,
          'official_TEST_QA': 0, 'new_optimizer_updates': 0,
          'single_pass_latency_not_controlled_speedup': True})


def controller():
    import fcntl
    PRIVATE.mkdir(parents=True, exist_ok=True)
    lock = (PRIVATE / '.controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_ready()
    assert not (REPORT / 'CONTROLLER_STARTED.json').exists()
    write(REPORT / 'CONTROLLER_STARTED.json', {'pid': os.getpid(), 'startticks': Path('/proc/self/stat').read_text().split()[21]})
    for stage in ['prepare', *CASES, 'report']:
        args = ['--prepare'] if stage == 'prepare' else ['--case', stage]
        entry = Path(__file__)
        if stage == 'report':
            entry, args = ROOT / 'analysis/edge_width_split_report_20261021.py', []
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   OPENBLAS_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')
        with (REPORT / (stage + '.log')).open('xb') as log:
            worker = subprocess.Popen([sys.executable, '-u', str(entry), *args], cwd=ROOT,
                                      stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=env)
            write(REPORT / 'CONTROLLER_STATE.json', {'phase': stage, 'worker_pid': worker.pid,
                  'worker_startticks': Path(f'/proc/{worker.pid}/stat').read_text().split()[21], 'CPU_only': True})
            code = worker.wait()
        terminal = REPORT / ('VERIFIED_RESULTS.json' if stage == 'report' else stage + '_COMPLETE.json')
        if code or not terminal.exists():
            write(REPORT / 'NEEDS_ATTENTION.json', {'phase': stage, 'returncode': code, 'no_blind_retry': True})
            return
    write(REPORT / 'CONTROLLER_STATE.json', {'phase': 'complete_verified', 'CPU_only': True,
          'new_optimizer_updates': 0, 'goal_achieved': False})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gate', action='store_true')
    p.add_argument('--admit', action='store_true')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--case', choices=CASES)
    p.add_argument('--all', action='store_true')
    args = p.parse_args()
    try:
        if args.gate:
            cpu_gate()
        elif args.admit:
            admit()
        elif args.prepare:
            prepare()
        elif args.case:
            case(args.case)
        elif args.all:
            controller()
        else:
            p.error('Choose an action')
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        (REPORT / ('FAILURE_' + str(os.getpid()) + '.txt')).write_text(traceback.format_exc())
        raise
