"""Direct original Qwen SeqFT stage19 reload and five fixed resource windows.

No reconstruction/fold, training, F1, native-backend change or latency study.
Physical GPU1 is exposed as the checkpoint's original virtual CUDA0.
"""
from pathlib import Path
from types import SimpleNamespace
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import signal
import sys
import time
import traceback

ROOT = Path(os.environ['LLMCL_ROOT'])
RUN = Path(os.environ['LLMCL_RUN'])
ASSET = Path(os.environ['LLMCL_ASSET'])
LIMIT = 600
CP_SHA = 'd3df6622f2de818e3b5cb5209c106cc2bc827ae87a65841af823032dcd3d2f85'
CP_BYTES = 4562999772
CONFIG_SHA = '9db372397a602a25499fe83c74871fab9696bdadf5442352867a55277a2ccea1'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
WEIGHTS = dict(coefficients=2210075136, bytes=4422329344)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (RUN / name).open('x', encoding='utf8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value)
    return value


def root_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def structure(model, method):
    import torch
    from methods import SeqLinear
    layers = {name: layer for name, layer in model.named_modules() if isinstance(layer, SeqLinear)}
    assert len(layers) == len(method.layers) == 56
    assert {id(x) for x in layers.values()} == {id(x) for x in method.layers}
    expected = {f'model.layers.{i}.self_attn.{kind}_proj' for i in range(28) for kind in ('q', 'v')}
    assert set(layers) == expected
    result = {}
    adapters = {}
    for name, layer in layers.items():
        assert layer.A.weight.shape[0] == layer.B.weight.shape[1] == 8
        assert layer.scale == 4 and layer.dropout.p == .1 and not layer.training
        assert layer.weight.dtype == torch.float16 and layer.A.weight.dtype == layer.B.weight.dtype == torch.float32
        assert layer.weight.is_cuda and layer.A.weight.is_cuda and layer.B.weight.is_cuda
        adapters[name + '.A.weight'] = layer.A.weight
        adapters[name + '.B.weight'] = layer.B.weight
        result[name] = dict(A_shape=list(layer.A.weight.shape), B_shape=list(layer.B.weight.shape),
                            base_dtype=str(layer.weight.dtype), adapter_dtype=str(layer.A.weight.dtype),
                            scale=layer.scale, dropout=layer.dropout.p)
    assert len(adapters) == 112 and sum(x.numel() for x in adapters.values()) == 1089536
    assert model.config.hidden_size == 1536
    assert model.config._attn_implementation == 'sdpa' and model.visual.config._attn_implementation == 'sdpa'
    return result


def generation(model):
    config = model.generation_config
    return dict(public=config.to_dict(), original_object_hash_present=hasattr(config, '_original_object_hash'),
                original_object_hash=getattr(config, '_original_object_hash', None))


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert not any((RUN / n).exists() for n in ('STARTED.json', 'COMPLETE.json', 'FAILURE.json'))
    ready = read(RUN / 'READY.json'); rh = sha(RUN / 'READY.json')
    for field in ('sources', 'prior_files'):
        for name, expected in ready[field].items():
            assert sha(RUN / name) == expected, name
    for name, expected in ready['runtime_files'].items():
        assert sha(root_path(name)) == expected, name
    for name, expected in ready['environment'].items():
        assert sha(name) == expected, name
    helper = ASSET / 'exp/paired_image_support_20261024/v1/common.py'
    assert sha(helper) == ready['identity_helper_sha256']
    ids = module('qwen_resource_identity', helper)
    card = ids.inventory()[1]
    assert not card['owners'] and card['memory_MiB'] < 500 and card['utilization'] < 10
    assert card['uuid'] == ready['GPU_uuid']
    who = ids.identity(os.getpid()); assert who['pid'] == who['pgid'] and who['uid'] == os.getuid()
    tick = time.monotonic()
    write('STARTED.json', dict(unix=time.time(), identity=who, physical_GPU=1, virtual_GPU=0,
        GPU_uuid=card['uuid'], ready_sha256=rh, limit=LIMIT,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()))
    def expired(*args):
        raise TimeoutError('Fixed original Qwen resource600 deadline expired')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(8); torch.cuda.set_device(0); assert torch.cuda.device_count() == 1
    port = ROOT / 'exp/two_backbone_native_cl_20261017_v2'; sys.path.insert(0, str(port))
    import checkpoints, onevision_bridge as bridge, runtime_utils as rt, frozen_vision as fv
    assert rt.ROOT == ROOT
    original = root_path(ready['original_run']); config = read(original / 'config.json')
    formal = read(RUN / 'formal_Qwen_COMPLETE.json'); meta = read(original / 'checkpoints/latest.json')
    assert sha(original / 'config.json') == formal['config_sha256'] == ready['config_sha256'] == CONFIG_SHA
    assert meta['stage'] == 19 and meta['path'] == 'checkpoints/stage_19.pt'
    assert meta['sha256'] == formal['checkpoint_sha256'] == ready['checkpoint_sha256'] == CP_SHA
    assert meta['bytes'] == ready['checkpoint_bytes'] == CP_BYTES and (original / meta['path']).stat().st_size == CP_BYTES
    assert sha(original / 'checkpoints/latest.json') == ready['checkpoint_metadata_sha256']
    assert config['method'] == 'SeqFT_LoRA' and config['backbone'] == 'qwen' and config['condition'] == 'pct01'
    assert config['model_dtype'] == config['autocast_dtype'] == 'torch.float16'
    assert config['manifest_sha256'] == ready['manifest_sha256'] == MANIFEST_SHA
    assert formal['candidate']['backbone'] == 'qwen' and formal['candidate']['method'] == 'SeqFT_LoRA' and formal['final_QA'] == 21454
    for name, expected in config['execution_source_sha256'].items():
        assert sha(original / 'runner_source' / name) == expected and sha(port / name) == expected
    # This original loader performs the single explicit full checkpoint SHA;
    # retain CPU history/cache placement instead of mapping all tensors to GPU.
    payload = checkpoints.load(original, expected=config)
    assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == 19
    method = payload['method']; model = method.model.eval(); processor = payload['processor']
    assert model.device == torch.device('cuda:0') and model.dtype == torch.float16
    assert method.name == 'SeqFT_LoRA' and method.original_revision == 'standard-sequential-LoRA-qv-r8-alpha32-dropout0.1'
    model.requires_grad_(False)
    assert all(p.device == torch.device('cuda:0') and p.grad is None for p in model.parameters())
    storage_path = ROOT / 'exp/ffn_static_delete_20261025/v1/core.py'
    assert sha(storage_path) == ready['storage_helper_sha256']
    storage = module('qwen_resource_storage', storage_path)
    layout = storage.layout(model)
    assert layout['weights'] == WEIGHTS and layout['ordinary_buffers']['bytes'] == 7504 and layout['trainable'] == 0
    before_structure = structure(model, method)
    before = storage.digest(model); buffers = storage.digest(list(model.named_buffers(remove_duplicate=False)))
    before_generation = generation(model)
    old_cache_bytes = model._llmcl_visual_cache_bytes
    model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
    model._llmcl_visual_cache_hits = model._llmcl_visual_cache_misses = 0; fv.LIMIT = 0
    def gpu_prepare(batch, row, active, original=None):
        pixels = batch.get('pixel_values')
        if pixels is None or (pixels.ndim == 2 and pixels.shape[-1] == active.config.hidden_size):
            return False
        assert not any(p.requires_grad for p in active.visual.parameters())
        extractor = original or active._llmcl_visual_original; dtype = active.visual.get_dtype()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach()
        assert features.is_cuda and features.ndim == 2 and features.shape[-1] == active.config.hidden_size
        assert torch.isfinite(features).all()
        assert len(features) == int((batch['input_ids'] == active.config.image_token_id).sum())
        active._llmcl_visual_cache_misses += 1; batch['pixel_values'] = features
        assert active._llmcl_visual_cache_bytes == 0
        return False
    fv.prepare = gpu_prepare
    bridge.bind(model); bridge.LAST_NATIVE = None; bridge.NATIVE_GATE_MODE = False
    naive_ready = read(RUN / 'prior_naive_READY.json'); naive = read(RUN / 'prior_naive_VERIFIED.json')
    naive_eval = read(RUN / 'prior_naive_eval_COMPLETE.json'); prior = read(RUN / 'prior_naive_memory.json')
    assert naive['complete'] and naive['ready_sha256'] == sha(RUN / 'prior_naive_READY.json')
    assert sha(RUN / 'prior_naive_eval_COMPLETE.json') == naive['source_COMPLETE_sha256']['eval_qv']
    assert naive_eval['files']['qv_memory.json'] == sha(RUN / 'prior_naive_memory.json')
    assert prior == naive['memory']['plain_r8'] and naive['same_GPU_for_final_resource_windows'] == 1
    naive_run = Path(ready['naive_run'])
    for name in ('common.py', 'readouts.py'):
        assert sha(naive_run / name) == naive_ready['sources'][name]
    sys.path.insert(0, str(ASSET / 'exp/osft_increment_screen_20261019/v1'))
    import study
    path, manifest, source_rows = study.original_train(); split, _, dev = study.split_train(manifest, source_rows)
    assert sha(path) == MANIFEST_SHA and sha(study.REPORT / 'SPLIT.json') == ready['split_sha256'] == naive_ready['split_sha256']
    assert split == read(study.REPORT / 'SPLIT.json')
    rows = [(t, row) for t in manifest['tasks'] for row in dev[t['task']]]
    common = module('qwen_bound_batcher', naive_run / 'common.py')
    readouts = module('qwen_bound_readouts', naive_run / 'readouts.py')
    ctx = SimpleNamespace(storage=storage, split=split, bridge=bridge, rt=rt, processor=processor)
    batcher = common.Batcher(ctx, model, 'QWEN_RESOURCE')
    def batch_provider(m, task, row):
        for name in row['images']:
            assert sha(ROOT / name) == split['image_sha256'][name]
        method.set_instruction(task['task'])
        return batcher.readout(m, task, row)
    stop = dict(generation_stop_ids=processor.tokenizer.eos_token_id, pad_id=processor.tokenizer.pad_token_id, context_limit=None)
    gc.collect(); torch.cuda.empty_cache()
    windows, counts = readouts.resource_windows(model, processor, rows, storage,
        batch_provider=batch_provider, generation_policy=stop, prior_windows=prior)
    assert counts['new_generations'] == 5 and counts['new_scores'] == counts['NLL_requests'] == 0
    for current, old in zip(windows, prior):
        for name in ('id', 'domain', 'raw_input_sha256', 'image_SHA256', 'image_shapes', 'retrieved_chunk_indexes', 'common_image_area_cap', 'new_tokens'):
            assert current[name] == old[name], name
    assert structure(model, method) == before_structure
    assert storage.digest(model) == before and storage.digest(list(model.named_buffers(remove_duplicate=False))) == buffers
    after_generation = generation(model); assert after_generation['public'] == before_generation['public']
    assert model._llmcl_visual_cache_bytes == 0
    write('memory.json', windows)
    seconds = time.monotonic() - tick; assert seconds <= LIMIT
    write('COMPLETE.json', dict(complete=True, identity=who, ready_sha256=rh, seconds=seconds,
        physical_GPU=1, virtual_GPU=0, GPU_uuid=card['uuid'], checkpoint_sha256=CP_SHA, checkpoint_bytes=CP_BYTES,
        original_config_sha256=CONFIG_SHA, manifest_sha256=MANIFEST_SHA, layout=layout,
        state_digest=before, all_buffers_digest=buffers, state_and_registered_buffers_unchanged=True,
        original_structure=before_structure, generation_before=before_generation, generation_after=after_generation,
        original_generation_objects_retained=True, private_generation_hash_may_be_process_specific=True,
        original_CPU_visual_cache_removed_bytes=old_cache_bytes, CPU_checkpoint_storage_bytes=133183696,
        CPU_metadata_is_not_RSS=True, original_dtype_and_unfolded_graph=True, no_reconstruction_or_export=True,
        cross_backbone_raw_inputs_equal=True, cross_backbone_prepared_equality_required=False,
        memory_sha256=sha(RUN / 'memory.json'), readout_counts=counts,
        new_generations=5, forced_new_tokens=160, new_scores=0, NLL_requests=0, backwards=0, optimizer_updates=0,
        latency_measured=False, versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')}))
    signal.alarm(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write('FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc(), no_quality_claim=True))
        raise
