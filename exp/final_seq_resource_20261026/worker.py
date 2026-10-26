"""Read-only original pct10 SeqFT S19: five fixed 32-token GPU requests, no F1.

CLI: --root ABS --output NEW_ABS --binding ABS --gpu PHYSICAL_INDEX
Binding JSON (all descriptors are {path, sha256}; paths may be root-relative):
  worker_sha256, backbone, original_run, formal_proof, config_sha256,
  checkpoint_metadata, input_manifest, input_split, prior_windows,
  helpers:{storage,readouts,clock}, runtime_files:{path:sha},
  environment:{path:sha}, versions:{distribution:version}, GPU_uuids:{index:uuid}.
The two existing formal proofs/CPs are fixed below. No CP reconstruction/export,
checkpoint_probe, model surgery, fold, answer scoring, retry or GPU allocation.
validate_metadata(root,binding_path) is stdlib-only; it never reads CP payloads.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import traceback

LIMIT = 600
DOMAINS = ('natural', 'medical', 'geography', 'finance', 'traffic')
INPUT_MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
INPUT_SPLIT_SHA = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
PCT10_MANIFEST_SHA = '02ef86674c22a9dbf7f0d406398615437f8406cff364415e14d54f5fdb5bfcb7'
ANCHORS = {
    'onevision': dict(proof='73fdbbd139a6910225088e78cc44033824456e09dec93411ed93582ac4d67c99',
        checkpoint='801ce3d059deb0e9e786147fcb03a983784194c4b13ce98c3767e68b93b8b618',
        bytes=1926008036, config='c652f2168ce7d01ba85e2777ae4762afc7659b5a15e2d377d8d527ea5818a3e4',
        coefficients=894216224, weight_bytes=1789513792, modules=48, adapter_coordinates=540672),
    'qwen': dict(proof='23e5d338cdc1579b6f8ca4f72d511f32e23896e248bfaf91e0ae775beb257081',
        checkpoint='0b657653a0f438169d0ec00380209a6a282921fc938274b5dfb407c757ac7927',
        bytes=4562998300, config='03312f0511147a9eb3aa88c20c51daf0bc67b482a5e83168e2c2998b5444aadf',
        coefficients=2210075136, weight_bytes=4422329344, modules=56, adapter_coordinates=1089536)}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_bytes())


def resolve(root, path):
    p = Path(path)
    assert p.is_absolute() or '..' not in p.parts
    return p if p.is_absolute() else root/p  # frozen data may be read-only symlinks


def checked(root, descriptor):
    p = resolve(root, descriptor['path']); assert sha(p) == descriptor['sha256'], str(p)
    return p


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def write(output, name, value):
    with (output/name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def identity(pid):
    p = Path('/proc')/str(pid); f = (p/'stat').read_text().rsplit(') ', 1)[1].split()
    assert f[0] != 'Z'
    return dict(pid=int(pid), startticks=f[19], uid=p.stat().st_uid, pgid=int(f[2]))


def validate_metadata(root, binding_path):
    """Small metadata/source/data verification only: no Torch, GPU or CP load/hash."""
    root = Path(root).resolve(); b = read(binding_path)
    assert sha(Path(__file__)) == b['worker_sha256']
    backbone = b['backbone']; anchor = ANCHORS[backbone]
    proof_path = checked(root, b['formal_proof']); assert sha(proof_path) == anchor['proof']
    proof = read(proof_path); original = resolve(root, b['original_run'])
    assert proof['candidate']['backbone'] == backbone and proof['candidate']['method'] == 'SeqFT_LoRA'
    assert proof['candidate']['condition'] == proof['condition'] == 'pct10'
    assert proof['candidate']['manifest_sha256'] == PCT10_MANIFEST_SHA
    logical_run = Path(proof['candidate']['run'])
    assert not logical_run.is_absolute() and '..' not in logical_run.parts
    assert proof['manifest_sha256'] == PCT10_MANIFEST_SHA and proof['final_QA'] == 21454
    config_path = original/'config.json'; config = read(config_path)
    assert sha(config_path) == proof['config_sha256'] == b['config_sha256'] == anchor['config']
    assert config['method'] == 'SeqFT_LoRA' and config['backbone'] == backbone and config['condition'] == 'pct10'
    assert config['manifest_sha256'] == PCT10_MANIFEST_SHA and not config['preflight_only']
    assert config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
    mp = checked(root, b['checkpoint_metadata']); meta = read(mp)
    assert meta == read(original/'checkpoints/latest.json')
    assert meta['stage'] == 19 and meta['path'] == 'checkpoints/stage_19.pt'
    assert meta['sha256'] == proof['checkpoint_sha256'] == anchor['checkpoint']
    assert meta['bytes'] == proof['checkpoint_bytes'] == anchor['bytes']
    assert (original/meta['path']).stat().st_size == anchor['bytes']
    # An archive mirror may have a different host-local path; its config,
    # sources and complete checkpoint must still equal the sealed logical run.
    assert proof['source_sha256'][(logical_run/'config.json').as_posix()] == anchor['config']
    for name, expected in config['execution_source_sha256'].items():
        assert Path(name).name == name and sha(original/'runner_source'/name) == expected, name
    # Import the ordinary port, never runner_source (its inferred ROOT is wrong).
    # Only modules actually imported are required to match the pct10 snapshot;
    # pct10 run.py metadata differences are not mistaken for runtime differences.
    port = root/'exp/two_backbone_native_cl_20261017_v2'
    for name in ('checkpoints.py', 'methods.py', 'runtime_utils.py', 'onevision_bridge.py', 'frozen_vision.py', 'retrieval.py'):
        assert sha(port/name) == config['execution_source_sha256'][name], name
    for field in ('runtime_files', 'environment'):
        assert b[field]
        for path, expected in b[field].items(): assert sha(resolve(root, path)) == expected, path
    for distribution, expected in b['versions'].items(): assert importlib.metadata.version(distribution) == expected
    assert {'torch', 'transformers', 'accelerate', 'cloudpickle'} <= set(b['versions'])
    helpers = {k: checked(root, b['helpers'][k]) for k in ('storage', 'readouts', 'clock')}
    manifest_path = checked(root, b['input_manifest'])
    assert b['input_manifest']['sha256'] == INPUT_MANIFEST_SHA
    manifest = read(manifest_path); split_path = checked(root, b['input_split']); split = read(split_path)
    assert b['input_split']['sha256'] == INPUT_SPLIT_SHA
    assert len(manifest['tasks']) == 19 and manifest['seed'] == split['seed'] == 42
    rows = []
    for task in manifest['tasks']:
        info = task['files']['train']; path = resolve(root, info['path'])
        assert sha(path) == info['sha256']
        items = [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()]
        assert len(items) == info['count']
        wanted = split['dev_ids'][str(task['task'])]; mapping = {r['id']: r for r in items}
        assert len(mapping) == len(items) and set(wanted) <= set(mapping)
        rows.extend((task, mapping[key]) for key in wanted)
    assert len(rows) == len({r['id'] for _, r in rows}) == 100
    assert all(sum(t['domain'] == d for t, _ in rows) == 20 for d in DOMAINS)
    selected = [min(((t, r) for t, r in rows if t['domain'] == d), key=lambda tr: hashlib.sha256(tr[1]['id'].encode()).digest()) for d in DOMAINS]
    prior_path = checked(root, b['prior_windows']); prior = read(prior_path)
    assert [(x['domain'], x['id']) for x in prior] == [(t['domain'], r['id']) for t, r in selected]
    assert all(x['new_tokens'] == 32 for x in prior)
    return SimpleNamespace(root=root, binding=b, binding_sha256=sha(binding_path), backbone=backbone,
        anchor=anchor, original=original, config=config, proof=proof, meta=meta, port=port,
        helpers=helpers, rows=selected, split=split, prior=prior,
        sources=dict(worker=sha(__file__), helpers={k: sha(p) for k, p in helpers.items()}))


def structure(model, method, seq_type, anchor):
    import torch
    discovered = {id(m): n for n, m in model.named_modules() if isinstance(m, seq_type)}
    core = getattr(model, 'core', model)
    decoder = core.language_model.model if hasattr(core, 'language_model') else model.model
    assert len(decoder.layers)*2 == anchor['modules']
    layers = {f'model.layers.{i}.self_attn.{kind}_proj': getattr(block.self_attn, kind+'_proj')
              for i, block in enumerate(decoder.layers) for kind in ('q', 'v')}
    assert all(isinstance(layer, seq_type) for layer in layers.values())
    assert len(discovered) == len(layers) == len(method.layers) == anchor['modules']
    assert set(discovered) == {id(x) for x in layers.values()} == {id(x) for x in method.layers}
    result = {}; coordinates = 0
    for name, layer in layers.items():
        assert layer.A.weight.shape[0] == layer.B.weight.shape[1] == 8
        assert layer.scale == 4 and layer.dropout.p == .1 and not layer.training
        assert layer.weight.dtype == torch.float16 and layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert all(x.is_cuda and x.device.index == 0 for x in (layer.weight, layer.A.weight, layer.B.weight))
        coordinates += layer.A.weight.numel()+layer.B.weight.numel()
        result[name] = dict(A_shape=list(layer.A.weight.shape), B_shape=list(layer.B.weight.shape), scale=layer.scale,
            base_dtype=str(layer.weight.dtype), adapter_dtype=str(layer.A.weight.dtype), dropout=layer.dropout.p,
            observed_registered_name=discovered[id(layer)])
    assert coordinates == anchor['adapter_coordinates']
    return dict(resident_adapter_banks=1, projection_modules=len(result), adapter_coordinates=coordinates,
        definition='One shared LoRA across layers/stages; projection modules are not independent banks', layers=result)


def generation(model):
    core = getattr(model, 'core', model)
    values = {'outer': core.generation_config}
    if hasattr(core, 'language_model'): values['language'] = core.language_model.generation_config
    return {k: v.to_dict() for k, v in values.items()}


def dtype_storage(model):
    result = {}; seen = set()
    for name, value in model.named_parameters(remove_duplicate=False):
        s = value.untyped_storage(); key = (str(value.device), s.data_ptr(), s.nbytes())
        if key in seen: continue
        seen.add(key); item = result.setdefault(str(value.dtype), dict(bytes=0, coefficients=0))
        item['bytes'] += s.nbytes(); item['coefficients'] += s.nbytes()//value.element_size()
    return result


def raw_binding(row, task, details, split):
    context = row.get('context', ''); indexes = details['retrieved_chunk_indexes']
    if len(context) > 6000:
        chunks = [context[i:i+1000] for i in range(0, len(context), 800)]
        context = '\n'.join(chunks[i] for i in indexes)
    else: assert indexes == []
    contract = dict(images=[dict(path=n, sha256=split['image_sha256'][n]) for n in row['images']],
        instruction=task['instruction'], question=row['question'], original_context=row.get('context', ''),
        resolved_context=context, retrieved_chunk_indexes=indexes, image_shapes=details['image_shapes'], common_image_area_cap=401408)
    return dict(raw_input_sha256=hashlib.sha256(json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest(), image_SHA256=contract['images'])


def main(args):
    root = args.root.resolve(); output = args.output
    assert root.is_absolute() and output.is_absolute() and args.binding.is_absolute()
    output.mkdir(parents=True, exist_ok=False)
    tick = time.monotonic(); who = identity(os.getpid()); assert who['pid'] == who['pgid']
    write(output, 'STARTED.json', dict(unix=time.time(), identity=who, physical_GPU=args.gpu, virtual_GPU=0,
        binding_sha256=sha(args.binding), limit=LIMIT, boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def expired(*unused): raise TimeoutError('Fixed final Seq resource600 deadline expired')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    c = validate_metadata(root, args.binding); b = c.binding
    # A caller's allocator supplies the slot. This worker only refuses occupied
    # hardware; it never allocates another GPU or signals an existing process.
    q = subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-gpu=uuid,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True).strip().split(',')
    assert q[0].strip() == b['GPU_uuids'][str(args.gpu)] and int(q[1]) < 500 and int(q[2]) < 10
    assert not subprocess.check_output(['nvidia-smi', f'--id={args.gpu}', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false', LLMCL_ROOT=str(root))
    import torch
    torch.set_num_threads(8); torch.cuda.set_device(0); assert torch.cuda.device_count() == 1
    sys.path.insert(0, str(c.port))
    import checkpoints, methods, runtime_utils as rt, onevision_bridge as bridge, frozen_vision as fv
    for name, value in [('checkpoints.py', checkpoints), ('methods.py', methods), ('runtime_utils.py', rt), ('onevision_bridge.py', bridge), ('frozen_vision.py', fv)]:
        assert sha(Path(value.__file__)) == c.config['execution_source_sha256'][name]
    # Original load hashes the full CP once and restores original CPU/CUDA0
    # placement/RNG. It does not invoke checkpoint_probe or delete anything.
    load_tick = time.monotonic()
    payload = checkpoints.load(c.original, expected=c.config)
    checkpoint_load_seconds = time.monotonic()-load_tick
    assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == 19
    method = payload['method']; model = method.model.eval(); processor = payload['processor']
    assert method.name == 'SeqFT_LoRA' and method.original_revision == 'standard-sequential-LoRA-qv-r8-alpha32-dropout0.1'
    assert model.device == torch.device('cuda:0') and model.dtype == torch.float16
    model.requires_grad_(False); core = getattr(model, 'core', model)
    assert {id(p) for p in model.parameters()} == {id(p) for p in core.parameters()}
    assert all(p.is_cuda and p.device.index == 0 and p.grad is None for p in core.parameters())
    storage = module('final_seq_storage', c.helpers['storage'])
    readouts = module('final_seq_readouts', c.helpers['readouts'])
    clock_module = module('final_seq_token_clock', c.helpers['clock'])
    layout = storage.layout(core)
    assert layout['weights'] == dict(coefficients=c.anchor['coefficients'], bytes=c.anchor['weight_bytes'])
    assert layout['trainable'] == 0
    before_structure = structure(model, method, methods.SeqLinear, c.anchor)
    before = storage.digest(core); buffers_before = storage.digest(list(core.named_buffers(remove_duplicate=False)))
    generation_before = generation(model); dtypes = dtype_storage(core)
    old_cache = model._llmcl_visual_cache_bytes
    model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
    model._llmcl_visual_cache_hits = model._llmcl_visual_cache_misses = 0; fv.LIMIT = 0
    # Match the existing frozen GPU transport: same visual arithmetic, no
    # CPU visual-feature staging/cache. No hashing is done inside preparation.
    def gpu_prepare(batch, row, active, original=None):
        pixels = batch.get('pixel_values')
        if pixels is None or (pixels.ndim == 2 and pixels.shape[-1] == active.config.hidden_size): return False
        extractor = original or active._llmcl_visual_original; dtype = active.visual.get_dtype()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach()
        assert features.is_cuda and features.ndim == 2 and features.shape[-1] == active.config.hidden_size
        active._llmcl_visual_cache_misses += 1; batch['pixel_values'] = features
        return False
    fv.prepare = gpu_prepare
    bridge.bind(model); bridge.LAST_NATIVE = None; bridge.NATIVE_GATE_MODE = False
    # Check all five files before timing. rt.ROOT may be the original asset
    # root behind a read-only symlink, so bind the actual copy it will open.
    for _, row in c.rows:
        for name in row['images']: assert sha(Path(rt.ROOT)/name) == c.split['image_sha256'][name], name
    stop = dict(generation_stop_ids=processor.tokenizer.eos_token_id, pad_id=processor.tokenizer.pad_token_id, context_limit=None)
    records = []
    for (task, row), prior in zip(c.rows, c.prior):
        method.set_instruction(task['task']); bridge.bind(model); bridge.LAST_NATIVE = None
        assert model._llmcl_visual_cache_bytes == 0
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(0); torch.cuda.reset_peak_memory_stats(0)
        baseline = torch.cuda.memory_allocated(0)
        clock = clock_module.FixedWindowTokenClock(synchronize=lambda: torch.cuda.synchronize(0), expected_tokens=32)
        original_generate = model.generate; saved_instance_generate = model.__dict__.get('generate')
        def measured_generate(*positional, **keywords):
            assert 'streamer' not in keywords; keywords['streamer'] = clock
            return original_generate(*positional, **keywords)
        model.generate = measured_generate
        try:
            # Only real preparation/visual/decoder work plus the original
            # finite-logit guard and synchronized streamer are inside timing.
            clock.start(); batch, details = rt.input_batch(row, task, processor, model.device)
            batch = {n: v for n, v in batch.items() if not n.startswith('_')}
            tokens = readouts._generate(model, batch, 32, stop, True)
            timing = clock.result(); torch.cuda.synchronize(0)
            peak = torch.cuda.max_memory_allocated(0); reserved = torch.cuda.max_memory_reserved(0)
        finally:
            # Restore the original instance/class dispatch, never another graph.
            if saved_instance_generate is None: del model.generate
            else: model.generate = saved_instance_generate
        assert len(tokens) == 32 and timing['generated_tokens'] == 32
        # Audit-only GPU->CPU tensor digests and contract SHA are deliberately
        # after the timing endpoints and the captured allocation peak.
        binding = raw_binding(row, task, details, c.split)
        assert binding['raw_input_sha256'] == prior['raw_input_sha256']
        assert binding['image_SHA256'] == prior['image_SHA256']
        for name in ('image_shapes', 'retrieved_chunk_indexes', 'common_image_area_cap'):
            assert details[name] == prior[name], name
        binding.update(details, prepared_sha256=storage.digest(batch),
            transport_sha256=storage.digest({n: v for n, v in batch.items() if n != 'pixel_values'}),
            prompt_tokens=batch['input_ids'].shape[1], input_shape=list(batch['input_ids'].shape),
            input_tensor_shapes={n: list(v.shape) for n, v in batch.items()})
        records.append(dict(binding, id=row['id'], task=task['task'], domain=task['domain'], new_tokens=32,
            baseline_allocated=baseline, peak_allocated=peak, peak_reserved=reserved, timing=timing,
            audit_hashes_in_timing=False, finite_logit_guard_in_timing=True, synchronized_token_callbacks=True))
        del batch, tokens, clock, original_generate, measured_generate, saved_instance_generate
    assert structure(model, method, methods.SeqLinear, c.anchor) == before_structure
    assert storage.digest(core) == before and storage.digest(list(core.named_buffers(remove_duplicate=False))) == buffers_before
    assert generation(model) == generation_before and model._llmcl_visual_cache_bytes == 0
    write(output, 'windows.json', records)
    seconds = time.monotonic()-tick; assert seconds <= LIMIT
    write(output, 'COMPLETE.json', dict(complete=True, identity=who, seconds=seconds,
        binding_sha256=c.binding_sha256, source_sha256=c.sources, backbone=c.backbone,
        original_run=str(c.original), sealed_logical_run=c.proof['candidate']['run'], formal_proof_sha256=ANCHORS[c.backbone]['proof'],
        config_sha256=c.anchor['config'], checkpoint_sha256=c.anchor['checkpoint'], checkpoint_bytes=c.anchor['bytes'],
        checkpoint_load_seconds=checkpoint_load_seconds,
        checkpoint_load_seconds_scope='original checkpoints.load including full checkpoint SHA, torch.load and RNG restoration; subset of total seconds',
        original_full_unfolded_graph=True, resident_adapter_banks=1, adapter_structure=before_structure,
        layout=layout, dtype_storage=dtypes, state_digest=before, all_buffers_digest=buffers_before,
        state_and_registered_buffers_unchanged=True, generation_public_unchanged=True,
        original_CPU_visual_cache_removed_bytes=old_cache, original_other_CPU_payload_retained=True,
        physical_GPU=args.gpu, virtual_GPU=0, GPU_uuid=q[0].strip(), runtime_image_root=str(rt.ROOT),
        window_order='fresh_reload_before_any_other_generation', window_selection='original per-domain hash-min DEV raw IDs',
        windows_sha256=sha(output/'windows.json'), new_generations=5, forced_new_tokens=160,
        new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0,
        timing='instrumented wall time; finite-logit checks and per-token synchronization included; not production throughput',
        audit_hashes_in_timing=False, audit_digest_after_captured_peak=True,
        checkpoint_deleted_or_rewritten=False, no_reconstruction_or_export=True,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate', 'cloudpickle')}))
    signal.alarm(0)
    print(json.dumps(dict(complete=True, output=str(output), backbone=c.backbone, new_generations=5, new_scores=0, seconds=seconds)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'output', 'binding'): parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=range(4), required=True)
    arguments = parser.parse_args()
    try: main(arguments)
    except BaseException:
        marker = arguments.output/'STARTED.json'
        if marker.is_file() and read(marker)['identity']['pid'] == os.getpid() and not (arguments.output/'FAILURE.json').exists():
            write(arguments.output, 'FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc(), new_quality_claim=False))
        raise
