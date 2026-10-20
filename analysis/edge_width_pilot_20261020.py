"""CPU-only structural FFN pilot; all original baselines/GPU queues untouched."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/structured_width_20261020/v1'
PRIVATE = ROOT / 'exp/edge_structured_width_20261020/v1'
TEACHER = ROOT / 'exp/edge_teacher_transfer_20261019/onevision_osft_final'
PARENT = ROOT / 'summary/edge_deployment_20261019/cpu_function_space_20261020/v2'
CASES = ['dense', 'weight', 'fluctuation']
SOURCES = ['edge_width_pilot_20261020.py', 'edge_width_core_20261020.py',
           'edge_width_runtime_20261020.py', 'edge_width_report_20261020.py',
           'cpu_function_space_20261020_v2.py']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def imports():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    sys.path.insert(0, str(TEACHER / 'runner_source'))
    import frozen_vision, runtime_utils, retrieval
    frozen_vision.ROOT = runtime_utils.ROOT = retrieval.ROOT = ROOT
    return torch


def verify_ready():
    r = json.loads((REPORT / 'READY.json').read_text())
    assert r['sources'] == {n: sha(ROOT / 'analysis' / n) for n in SOURCES}
    assert r['protocol_sha256'] == sha(REPORT / 'PROTOCOL.md')
    assert r['CPU_gate_sha256'] == sha(REPORT / 'CPU_GATE.json')
    assert r['parent_evidence_sha256'] == sha(PARENT / 'EVIDENCE.json')
    assert r['upstream_manifest_sha256'] == sha(ROOT / 'code/FLAP/source_manifest.json')
    for rel, h in r['input_source_sha256'].items():
        assert sha(ROOT / rel) == h
    for rel, h in r['teacher_runner_sources'].items():
        assert sha(TEACHER / 'runner_source' / rel) == h
    return r


def data():
    from cpu_function_space_20261020_v2 import selection
    selected, sources, rows = selection()
    r = json.loads((REPORT / 'READY.json').read_text())
    assert selected == r['selected'] and sources == r['input_source_sha256']
    return selected, rows


def wrap(core):
    import onevision_bridge, frozen_vision
    for p in core.parameters():
        p.requires_grad_(False)
    model = onevision_bridge.OneVisionBridge(core).eval()
    onevision_bridge.bind(model)
    frozen_vision.install(model)
    return model


def batch(model, processor, task, row):
    import runtime_utils
    # Clear between every request: each cost includes fresh visual encoding.
    model._llmcl_visual_cache.clear()
    model._llmcl_visual_cache_bytes = 0
    return runtime_utils.input_batch(row, task, processor, 'cpu')


def package(core, processor, name):
    import torch, transformers
    from edge_width_core_20261020 import parameter_digest
    destination = PRIVATE / 'packages' / name
    assert not destination.exists()
    destination.mkdir(parents=True)
    assert {p.dtype for p in core.parameters()} == {torch.float16}
    digest = parameter_digest(core)
    core.save_pretrained(destination, safe_serialization=True, max_shard_size='2GB')
    processor.save_pretrained(destination)
    shutil.copyfile(ROOT / 'analysis/edge_width_runtime_20261020.py', destination / 'runtime.py')
    write(destination / 'runtime_spec.json', {
        'entry': 'from runtime import WidthOneVision; WidthOneVision.from_pretrained(path, torch_dtype="auto", attn_implementation="sdpa", local_files_only=True)',
        'storage_dtype': 'float16', 'compute_autocast': 'CPU float16',
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'structurally_reduced_not_zero_mask': name != 'dense',
        'not_phone_or_GPU_measurement': True})
    files = {str(p.relative_to(destination)): {'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(destination.rglob('*')) if p.is_file()}
    return {'package': str(destination.relative_to(ROOT)), 'files': files,
            'actual_package_bytes': sum(x['bytes'] for x in files.values()),
            'parameters': sum(p.numel() for p in core.parameters()),
            'parameter_digest': digest,
            'text_intermediate_size': core.config.text_config.intermediate_size,
            'vision_intermediate_size': core.config.vision_config.intermediate_size}


def prepare_models():
    torch = imports()
    import frozen_vision, onevision_bridge, runtime_utils
    from transformers import AutoProcessor
    from edge_width_runtime_20261020 import WidthOneVision
    from edge_width_core_20261020 import sites, QAMoments, prune
    ready = verify_ready()
    assert not (REPORT / 'prepare_STARTED.json').exists()
    write(REPORT / 'prepare_STARTED.json', {'pid': os.getpid(), 'startticks': Path('/proc/self/stat').read_text().split()[21]})
    tick = time.monotonic()
    selected, rows = data()
    checkpoint = TEACHER / 'checkpoints/stage_19.pt'
    assert sha(checkpoint) == ready['teacher_checkpoint_sha256']
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=pickle, weights_only=False)
    original = payload['method'].model.eval()
    processor = payload['processor']
    onevision_bridge.bind(original)
    frozen_vision.install(original)
    first = next(x for x in selected if x['role'] == 'calibration' and x['domain'] == 'medical')
    task, row = rows[first['id']]
    b, _ = batch(original, processor, task, row)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        expected, _ = runtime_utils.answer_loss(original, b, row['answer'], processor)
    assert torch.isfinite(expected)
    expected = float(expected)
    core = original.core
    replaced = 0
    with torch.inference_mode():
        for name, module in list(core.named_modules()):
            if type(module).__name__ != 'OSFTLinear':
                continue
            with torch.autocast('cpu', dtype=torch.float16):
                w = module.kernel['reconstruct_weight_matrix'](module.svd(), torch.float32, module.output_dtype).detach()
            linear = torch.nn.Linear(w.shape[1], w.shape[0], bias=module.bias is not None, dtype=w.dtype)
            linear.weight = torch.nn.Parameter(w, requires_grad=False)
            if module.bias is not None:
                linear.bias = torch.nn.Parameter(module.bias.detach(), requires_grad=False)
            parent, child = name.rsplit('.', 1)
            setattr(core.get_submodule(parent), child, linear)
            replaced += 1
    assert replaced == 168
    model = wrap(core)
    del payload, original, b, module
    gc.collect()
    b, _ = batch(model, processor, task, row)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        actual, _ = runtime_utils.answer_loss(model, b, row['answer'], processor)
    assert float(actual) == expected, 'Original/new static CPU reference differs'
    write(REPORT / 'REFERENCE_GATE.json', {'id': first['id'], 'original_NLL': expected,
          'folded_NLL': float(actual), 'exact': True, 'CPU_only': True})
    found = sites(core)
    assert sum(k == 'text' for k, _, _ in found.values()) == 24
    assert sum(k == 'vision' for k, _, _ in found.values()) == 26
    assert {o.in_features for k, _, o in found.values() if k == 'text'} == {4864}
    assert {o.in_features for k, _, o in found.values() if k == 'vision'} == {4304}
    moments = {n: QAMoments() for n in found}
    context = {'id': None, 'text_mask': None}
    handles = []
    def hook_for(name, kind):
        def hook(module, args):
            if context['id'] is None:
                return
            x = args[0]
            if kind == 'text':
                assert x.shape[0] == 1 and len(context['text_mask']) == x.shape[1]
                x = x[0, context['text_mask']]
            else:
                x = x.reshape(-1, x.shape[-1])
            moments[name].add(x, context['id'])
        return hook
    for name, (kind, _, output) in found.items():
        handles.append(output.register_forward_pre_hook(hook_for(name, kind)))
    calibration = [first] + [x for x in selected if x['role'] == 'calibration' and x['id'] != first['id']]
    for i, item in enumerate(calibration):
        assert time.monotonic() - tick < 7200, 'CPU preparation budget exceeded'
        task, row = rows[item['id']]
        context['id'] = item['id']
        b, _ = batch(model, processor, task, row)
        target = processor.tokenizer(row['answer'], add_special_tokens=False, truncation=True, max_length=512)['input_ids']
        target += [processor.tokenizer.eos_token_id]
        ids = torch.cat([b['input_ids'][0], torch.tensor(target)], dim=0)
        context['text_mask'] = ids != model.config.image_token_id
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
            loss, _ = runtime_utils.answer_loss(model, b, row['answer'], processor)
        assert torch.isfinite(loss)
        context['id'] = None
        write(REPORT / 'prepare_PROGRESS.json', {'calibration_QA': i + 1, 'required': 5, 'id': item['id']})
        print('CALIBRATION_COMPLETE', i + 1, item['id'], flush=True)
    for h in handles:
        h.remove()
    stats = {n: m.finish() for n, m in moments.items()}
    assert all(len(x['ids']) == 5 for x in stats.values())
    stats_path = PRIVATE / 'calibration_moments.pt'
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stats, stats_path)
    packages = {'dense': package(core, processor, 'dense')}
    assert packages['dense']['parameters'] == 893675552
    del handles, found, moments, model, core, b
    gc.collect()
    dense_path = ROOT / packages['dense']['package']
    for rule in ['weight', 'fluctuation']:
        assert time.monotonic() - tick < 7200, 'CPU preparation budget exceeded'
        candidate = WidthOneVision.from_pretrained(dense_path, torch_dtype='auto', attn_implementation='sdpa', local_files_only=True).eval()
        changes = prune(candidate, stats, rule)
        masks = PRIVATE / (rule + '_structural_masks.json')
        write(masks, changes)
        packages[rule] = package(candidate, processor, rule)
        assert packages[rule]['parameters'] < .70 * packages['dense']['parameters']
        assert packages[rule]['actual_package_bytes'] < .75 * packages['dense']['actual_package_bytes']
        packages[rule]['private_masks_sha256'] = sha(masks)
        del candidate
        gc.collect()
    assert packages['weight']['parameters'] == packages['fluctuation']['parameters']
    assert not torch.cuda.is_initialized()
    write(REPORT / 'prepare_COMPLETE.json', {
        'source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json'),
        'CPU_only': True, 'original_fold_exact': True, 'reference_NLL': expected,
        'calibration_QA': 5, 'calibration_moments_sha256': sha(stats_path),
        'packages': packages, 'seconds': time.monotonic() - tick,
        'new_optimizer_updates': 0, 'not_quality_or_speed_proof': True})


def memory(process):
    return process.memory_full_info()._asdict()


def measure(model, processor, task, row, speed):
    import torch, psutil, runtime_utils
    from transformers.generation.streamers import BaseStreamer
    process = psutil.Process()
    peak = [process.memory_info().rss]
    stop = threading.Event()
    def observe():
        while not stop.wait(.02):
            peak[0] = max(peak[0], process.memory_info().rss)
    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    before = memory(process)
    start = time.perf_counter()
    class Clock(BaseStreamer):
        def __init__(self):
            self.first = True
            self.tokens = []
            self.times = []
        def put(self, value):
            if self.first:
                self.first = False
                return
            self.times.append(time.perf_counter())
            self.tokens.extend(value.reshape(-1).tolist())
        def end(self):
            pass
    clock = Clock()
    try:
        with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
            prepared, details = batch(model, processor, task, row)
            model.generate(**prepared, max_new_tokens=8 if speed else 64,
                           min_new_tokens=8 if speed else 0, do_sample=False, use_cache=True,
                           streamer=clock, pad_token_id=processor.tokenizer.pad_token_id,
                           eos_token_id=processor.tokenizer.eos_token_id)
        elapsed = time.perf_counter() - start
    finally:
        stop.set()
        monitor.join()
    peak[0] = max(peak[0], process.memory_info().rss)
    assert clock.times and (not speed or len(clock.tokens) == len(clock.times) == 8)
    prediction = processor.tokenizer.decode(clock.tokens, skip_special_tokens=True)
    references = row['answers']
    assert references and isinstance(references, list)
    return {'id': row['id'], 'domain': task['domain'], 'speed_only': speed,
            'TTFT_seconds': clock.times[0] - start,
            'decode_tokens_per_second': ((len(clock.times) - 1) / (clock.times[-1] - clock.times[0])) if len(clock.times) > 1 else None,
            'seconds': elapsed, 'output_token_count': len(clock.tokens), 'input': details,
            'memory_before': before, 'memory_after': memory(process),
            'generation_peak_RSS_bytes': peak[0], 'system_loadavg': Path('/proc/loadavg').read_text(),
            'prediction': None if speed else prediction, 'references': None if speed else references,
            'raw_AnswerF1': None if speed else runtime_utils.score(prediction, references)['token_f1']}


def case(name):
    torch = imports()
    import psutil, runtime_utils
    from transformers import AutoProcessor
    from edge_width_runtime_20261020 import WidthOneVision
    from edge_width_core_20261020 import parameter_digest
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
    before_load = memory(process)
    load_tick = time.perf_counter()
    core = WidthOneVision.from_pretrained(folder, torch_dtype='auto', attn_implementation='sdpa', local_files_only=True).eval()
    assert parameter_digest(core) == spec['parameter_digest'], 'Fresh package parameters differ'
    processor = AutoProcessor.from_pretrained(folder, local_files_only=True)
    model = wrap(core)
    gc.collect()
    after_load = memory(process)
    load_seconds = time.perf_counter() - load_tick
    selected, rows = data()
    warm = next(x for x in selected if x['role'] == 'calibration' and x['domain'] == 'medical')
    task, row = rows[warm['id']]
    b, _ = batch(model, processor, task, row)
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        loss, _ = runtime_utils.answer_loss(model, b, row['answer'], processor)
    assert torch.isfinite(loss)
    if name == 'dense':
        assert float(loss) == prepared['reference_NLL'], 'Fresh dense whole-model reference differs'
    del b
    model._llmcl_visual_cache.clear()
    model._llmcl_visual_cache_bytes = 0
    gc.collect()
    records = []
    for item in [x for x in selected if x['role'] == 'probe']:
        assert time.monotonic() - tick < 5400, 'CPU case budget exceeded'
        task, row = rows[item['id']]
        for speed in [True, False]:
            records.append(measure(model, processor, task, row, speed))
        write(PRIVATE / (name + '_partial_predictions.json'), records)
        write(REPORT / (name + '_PROGRESS.json'), {'complete': False, 'completed_probe_QA': len(records) // 2, 'required': 5})
        print('PROBE_COMPLETE', name, item['id'], flush=True)
    assert parameter_digest(core) == spec['parameter_digest']
    assert not torch.cuda.is_initialized()
    predictions = PRIVATE / (name + '_predictions.json')
    write(predictions, records)
    public = [{k: v for k, v in x.items() if k not in ['prediction', 'references']} for x in records]
    write(REPORT / (name + '_COMPLETE.json'), {
        'complete': True, 'case': name, 'source_sha256': sha(__file__),
        'ready_sha256': sha(REPORT / 'READY.json'), 'prepare_sha256': sha(REPORT / 'prepare_COMPLETE.json'),
        'package': spec, 'fresh_parameter_digest_exact': True, 'probe_QA': 5,
        'TRAIN_probe_not_unseen_generalization': True, 'teacher_trained_on_probe_QA': True,
        'official_TEST_QA': 0, 'DEV_selection': False, 'new_optimizer_updates': 0,
        'requests': public, 'raw_predictions_sha256': sha(predictions),
        'load_seconds': load_seconds, 'memory_before_load': before_load, 'memory_after_load': after_load,
        'warmup_calibration_NLL': float(loss), 'seconds': time.monotonic() - tick,
        'CPU_only': True, 'server_proxy_not_phone_or_GPU': True,
        'single_pass_latency_not_final_speedup_proof': True})


def admit():
    from cpu_function_space_20261020_v2 import selection
    selected, inputs, _ = selection()
    assert not (REPORT / 'READY.json').exists()
    gate = json.loads((REPORT / 'CPU_GATE.json').read_text())
    assert gate['fresh_reload_state_and_text_vision_exact'] and not gate['CUDA_initialized']
    inherited = json.loads((PARENT / 'EVIDENCE.json').read_text())
    assert sha(ROOT / 'analysis/cpu_function_space_20261020_v2.py') == inherited['source_sha256']
    # The exact original helper folder is frozen by content, not a mutable environment default.
    helpers = {str(p.relative_to(TEACHER / 'runner_source')): sha(p)
               for p in sorted((TEACHER / 'runner_source').glob('*.py'))}
    write(REPORT / 'READY.json', {
        'sources': {n: sha(ROOT / 'analysis' / n) for n in SOURCES},
        'protocol_sha256': sha(REPORT / 'PROTOCOL.md'), 'CPU_gate_sha256': sha(REPORT / 'CPU_GATE.json'),
        'parent_evidence_sha256': sha(PARENT / 'EVIDENCE.json'),
        'upstream_manifest_sha256': sha(ROOT / 'code/FLAP/source_manifest.json'),
        'teacher_checkpoint_sha256': inherited['checkpoint_sha256'],
        'teacher_runner_sources': helpers, 'selected': selected, 'input_source_sha256': inputs,
        'cases': CASES, 'calibration_QA': 5, 'probe_QA': 5,
        'new_optimizer_updates': 0, 'case_model_forwards': 'one warmup NLL + five forced8 + five free max64',
        'CPU_threads': 2, 'CPU_interop_threads': 1, 'seed': 42,
        'base_storage': 'FP16', 'autocast': 'CPU FP16', 'statistical_analysis': 'FP64',
        'GPU_started': False, 'official_TEST_used': False,
        'not_a_formal_comparison_or_independent_DEV_screen': True})


def all_cases():
    import fcntl
    PRIVATE.mkdir(parents=True, exist_ok=True)
    lock = (PRIVATE / '.cpu_controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_ready()
    assert not (REPORT / 'CONTROLLER_STARTED.json').exists()
    write(REPORT / 'CONTROLLER_STARTED.json', {'pid': os.getpid(), 'startticks': Path('/proc/self/stat').read_text().split()[21]})
    for stage in ['prepare', *CASES]:
        args = ['--prepare-models'] if stage == 'prepare' else ['--case', stage]
        log = (REPORT / (stage + '.log')).open('ab')
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', TOKENIZERS_PARALLELISM='false')
        worker = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()), *args], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        write(REPORT / 'CONTROLLER_STATE.json', {'phase': stage, 'worker_pid': worker.pid, 'CPU_only': True})
        code = worker.wait()
        log.close()
        terminal = REPORT / (stage + '_COMPLETE.json')
        if code or not terminal.exists():
            write(REPORT / 'NEEDS_ATTENTION.json', {'phase': stage, 'returncode': code, 'no_blind_retry': True})
            return
    write(REPORT / 'ALL_COMPLETE.json', {'cases': CASES, 'source_sha256': sha(__file__),
          'ready_sha256': sha(REPORT / 'READY.json'), 'CPU_only': True, 'not_full_quality_or_goal_complete': True})
    report = subprocess.run([sys.executable, str(ROOT / 'analysis/edge_width_report_20261020.py')], cwd=ROOT)
    if report.returncode:
        write(REPORT / 'NEEDS_ATTENTION.json', {'phase': 'independent_report', 'no_blind_retry': True})
        return
    write(REPORT / 'CONTROLLER_STATE.json', {'phase': 'complete_verified', 'CPU_only': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--admit', action='store_true')
    parser.add_argument('--prepare-models', action='store_true')
    parser.add_argument('--case', choices=CASES)
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    try:
        if args.admit:
            admit()
        elif args.prepare_models:
            prepare_models()
        elif args.case:
            case(args.case)
        elif args.all:
            all_cases()
        else:
            parser.error('Choose exactly one operation')
    except Exception:
        stage = 'prepare' if args.prepare_models else (args.case or 'controller')
        write(REPORT / (stage + '_FAILURE.json'), {'traceback': traceback.format_exc(),
              'source_sha256': sha(__file__), 'no_blind_retry': True})
        raise
