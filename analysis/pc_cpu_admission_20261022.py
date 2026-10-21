"""One Windows CPU portability request; no benchmark, GPU, training or TEST."""
import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'summary/edge_deployment_20261019/pc_cpu_admission_20261022/v1'
PRIVATE = ROOT / '.deployment/pc_cpu_admission_20261022/v1'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(name, obj):
    p = REPORT / name
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    tmp.replace(p)


def function(path, name, env):
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    return env[name]


def main():
    assert os.name == 'nt' and os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not (REPORT / 'STARTED.json').exists(), 'No automatic repeat'
    ready = json.loads((REPORT / 'READY.json').read_text(encoding='utf8'))
    assert sha(__file__) == ready['source_sha256']
    assert sha(REPORT / 'PROTOCOL.md') == ready['protocol_sha256']
    assert sha(PRIVATE / 'bundle.json') == ready['bundle_sha256']
    bundle = json.loads((PRIVATE / 'bundle.json').read_text(encoding='utf8'))
    for name, info in bundle['package']['files'].items():
        p = PRIVATE / 'package' / name
        assert p.stat().st_size == info['bytes'] and sha(p) == info['sha256'], name
    for rel, info in bundle['inputs'].items():
        p = PRIVATE / rel
        assert sha(p) == info['sha256'] and p.stat().st_size == info['bytes'], rel
    import psutil
    process = psutil.Process()
    process.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    versions = {n: importlib.metadata.version(n) for n in ready['versions']}
    assert versions == ready['versions']
    write('STARTED.json', {'pid': os.getpid(), 'create_time': process.create_time(),
                          'ready_sha256': sha(REPORT / 'READY.json'), 'CPU_only': True})
    start = time.monotonic()
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    assert torch.version.cuda is None and not torch.cuda.is_initialized()
    from PIL import Image
    from transformers import AutoProcessor
    sys.path.insert(0, str(PRIVATE / 'runner_source'))
    import onevision_bridge, frozen_vision, retrieval
    frozen_vision.ROOT = PRIVATE
    spec = importlib.util.spec_from_file_location('pc_width_runtime', PRIVATE / 'package/runtime.py')
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    core = runtime.WidthOneVision.from_pretrained(PRIVATE / 'package', torch_dtype='auto',
                                                  attn_implementation='sdpa', local_files_only=True)
    assert {p.dtype for p in core.parameters()} == {torch.float16}
    h = hashlib.sha256()
    for name, p in core.named_parameters():
        h.update(json.dumps([name, list(p.shape), str(p.dtype)]).encode())
        h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    digest = h.hexdigest()
    assert digest == bundle['package']['parameter_digest'], 'Fresh parameter digest mismatch'
    for p in core.parameters():
        p.requires_grad_(False)
    model = onevision_bridge.OneVisionBridge(core).eval()
    onevision_bridge.bind(model)
    frozen_vision.install(model)
    processor = AutoProcessor.from_pretrained(PRIVATE / 'package', local_files_only=True)
    loaded_seconds = time.monotonic() - start
    loaded_rss = process.memory_info().rss
    env = {'torch': torch, 'Image': Image, 'math': math, 'ROOT': PRIVATE, 'retrieve': retrieval.retrieve}
    source = PRIVATE / 'runner_source/runtime_utils.py'
    input_batch = function(source, 'input_batch', env)
    answer_loss = function(source, 'answer_loss', env)
    row, task = bundle['row'], bundle['task']
    tick = time.monotonic()
    with torch.inference_mode(), torch.autocast('cpu', dtype=torch.float16):
        batch, metadata = input_batch(row, task, processor, 'cpu')
        loss, target_tokens = answer_loss(model, batch, row['answer'], processor)
    value = float(loss)
    assert math.isfinite(value)
    exact = value == bundle['reference_NLL']
    info = process.memory_info()
    evidence = {'status': 'COMPATIBLE_EXACT' if exact else 'PLATFORM_DIFFERENCE',
                'CPU_only': True, 'GPU_calls': 0, 'optimizer_updates': 0,
                'requests': 1, 'generation_requests': 0, 'id': row['id'],
                'teacher_seen_TRAIN': True, 'parameter_digest': digest,
                'NLL': value, 'server_NLL': bundle['reference_NLL'], 'NLL_exact': exact,
                'absolute_NLL_difference': abs(value - bundle['reference_NLL']),
                'loaded_seconds': loaded_seconds, 'request_seconds': time.monotonic() - tick,
                'total_seconds': time.monotonic() - start,
                'loaded_RSS_bytes': loaded_rss, 'final_RSS_bytes': info.rss,
                'process_lifetime_peak_working_set_bytes': getattr(info, 'peak_wset', None),
                'PSS_bytes': None, 'USS_bytes': None, 'target_tokens': target_tokens,
                'request_metadata': metadata, 'versions': versions,
                'python': sys.version, 'platform': platform.platform(),
                'CPU_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'ready_sha256': sha(REPORT / 'READY.json'), 'not_benchmark': True}
    assert not torch.cuda.is_initialized()
    write('EVIDENCE.json', evidence)
    write('COMPLETE.json', {'status': evidence['status'], 'evidence_sha256': sha(REPORT / 'EVIDENCE.json'),
                            'source_sha256': sha(__file__), 'ready_sha256': sha(REPORT / 'READY.json')})


if __name__ == '__main__':
    try:
        main()
    except Exception:
        REPORT.mkdir(parents=True, exist_ok=True)
        write('FAILURE.json', {'traceback': traceback.format_exc(), 'source_sha256': sha(__file__)})
        raise
