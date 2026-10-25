"""One trusted original PAS graph reload and five fixed GPU resource windows.

No model construction/export, training, quality scoring or latency measurement.
Physical GPU1 is exposed as the original checkpoint's virtual CUDA0.
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
import subprocess
import sys
import time
import traceback

ROOT = Path(os.environ['LLMCL_ROOT'])
RUN = Path(os.environ['LLMCL_RUN'])
ASSET = Path(os.environ['LLMCL_ASSET'])
LIMIT = 600
CP_SHA = '583ab5d02e187fbd6a54bd4a05a019e33101deb8af2d9533795253140721441a'
CP_BYTES = 3051292733
CONFIG_SHA = '79edcca96dd315260816932438b7c0ec57835f5761d2ec0b88d8ecdb5c837fb6'
MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'


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
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def root_path(value):
    value = Path(value)
    return value if value.is_absolute() else ROOT / value


def absent(identity):
    path = Path('/proc') / str(identity['pid']) / 'stat'
    return not path.exists() or path.read_text().rsplit(') ', 1)[1].split()[19] != str(identity['startticks'])


def pair_finished(ready):
    pair = Path(ready['pair_run'])
    pair_ready = read(pair / 'PAIR_READY.json')
    assert sha(pair / 'PAIR_READY.json') == ready['pair_ready_sha256']
    assert sha(pair / 'ALL_EXIT.json') == ready['pair_all_exit_sha256']
    end = read(pair / 'ALL_EXIT.json')
    assert end['complete'] and end['ready_sha256'] == ready['pair_ready_sha256']
    assert set(end['source_COMPLETE_sha256']) == {'train_dynamic', 'train_static', 'eval_dynamic', 'eval_static'}
    phase_proofs = {}
    for phase, expected in end['source_COMPLETE_sha256'].items():
        complete = read(pair / (phase + '_COMPLETE.json'))
        started = read(pair / (phase + '_STARTED.json'))
        ex = read(pair / (phase + '_EXIT.json'))
        assert sha(pair / (phase + '_COMPLETE.json')) == expected
        assert started['identity'] == complete['identity']
        assert ex['returncode'] == 0 and ex['pid'] == started['identity']['pid']
        assert complete['ready_sha256'] == started['ready_sha256'] == ready['pair_ready_sha256']
        assert absent(started['identity'])
        assert not (pair / (phase + '_FAILURE.json')).exists()
        phase_proofs[phase] = {suffix: sha(pair / (phase + '_' + suffix + '.json'))
                               for suffix in ('STARTED', 'COMPLETE', 'EXIT')}
    assert pair_ready['phase_GPUs']['eval_dynamic'] == pair_ready['phase_GPUs']['eval_static'] == 1
    status = subprocess.run(['systemctl', '--user', 'show', pair_ready['unit'],
                             '-p', 'ActiveState', '-p', 'MainPID', '-p', 'ExecMainStatus'],
                            check=True, capture_output=True, text=True).stdout
    fields = dict(line.split('=', 1) for line in status.splitlines() if '=' in line)
    assert fields['ActiveState'] == 'inactive' and fields['MainPID'] == '0' and fields['ExecMainStatus'] == '0'
    for name, expected in ready['pair_sources'].items():
        assert sha(pair / name) == expected == pair_ready['sources'][name]
    assert set(ready['pair_sources']) == {'paired_common.py', 'paired_readouts.py'}
    return pair, pair_ready, dict(phase_proof_sha256=phase_proofs, service=status)


def structure(model):
    """Observe original inference configuration; do not normalize any values."""
    import torch
    result = {}
    for name, layer in model.core.named_modules():
        if not hasattr(layer, '_current_E'):
            continue
        active = layer.active_adapter
        assert isinstance(active, str) and list(layer.r) == list(layer.lora_A) == list(layer.lora_B) == [active]
        assert layer.r[active] == 128 and layer.expert_num == layer._current_E() == 2
        assert layer.gate_rank == 16 and layer.gate_router == 'a' and layer.gate_tie == 'A_T'
        assert not layer.merged and not layer.disable_adapters and not layer.fan_in_fan_out
        assert not layer.training and not layer.analysis_collect and not layer._record_enabled
        assert math.isfinite(layer.gate_tau) and layer.gate_tau > 0
        mask = layer._expert_mask
        assert mask.is_cuda and mask.dtype == torch.float32 and mask.shape == (2,) and torch.isfinite(mask).all()
        result[name] = dict(active_adapter=active, r=layer.r, lora_alpha=layer.lora_alpha,
                            scaling=layer.scaling, gate_tau=layer.gate_tau, gate_rank=layer.gate_rank,
                            expert_num=layer.expert_num, expert_use_num=layer.expert_use_num,
                            gate_router=layer.gate_router, gate_tie=layer.gate_tie,
                            fan_in_fan_out=layer.fan_in_fan_out, merged=layer.merged,
                            disable_adapters=layer.disable_adapters, analysis_collect=layer.analysis_collect,
                            record_enabled=layer._record_enabled, dropout_p=layer.lora_dropout[active].p,
                            mask=mask.detach().cpu().tolist(), mask_dtype=str(mask.dtype),
                            usage_recorder_present=layer.usage_recorder is not None)
    assert len(result) == 168
    return result


def generation(model):
    core = model.core
    return {name: dict(public=config.to_dict(), original_object_hash_present=hasattr(config, '_original_object_hash'),
                       original_object_hash=getattr(config, '_original_object_hash', None))
            for name, config in [('outer', core.generation_config), ('language', core.language_model.generation_config)]}


def main():
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '1'
    assert not any((RUN / name).exists() for name in ('STARTED.json', 'COMPLETE.json', 'FAILURE.json'))
    ready = read(RUN / 'READY.json'); ready_hash = sha(RUN / 'READY.json')
    for name, expected in ready['sources'].items():
        assert sha(RUN / name) == expected
    for name, expected in ready['runtime_files'].items():
        assert sha(root_path(name)) == expected
    for name, expected in ready['environment'].items():
        assert sha(name) == expected
    pair, pair_ready, pair_proof = pair_finished(ready)
    helper = ASSET / 'exp/paired_image_support_20261024/v1/common.py'
    assert sha(helper) == ready['identity_helper_sha256']
    ids = module('pas_resource_identity', helper)
    card = ids.inventory()[1]
    assert not card['owners'] and card['memory_MiB'] < 500 and card['utilization'] < 10
    identity = ids.identity(os.getpid())
    assert identity['pid'] == identity['pgid'] and identity['uid'] == os.getuid()
    tick = time.monotonic()
    write('STARTED.json', dict(unix=time.time(), identity=identity, physical_GPU=1, virtual_GPU=0,
                              GPU_uuid=card['uuid'], ready_sha256=ready_hash, limit=LIMIT,
                              boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), pair=pair_proof))
    def expired(*args):
        raise TimeoutError('Fixed PAS resource600 deadline expired')
    signal.signal(signal.SIGALRM, expired); signal.alarm(LIMIT)
    os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(8); torch.cuda.set_device(0)
    assert torch.cuda.device_count() == 1
    port = ROOT / 'exp/two_backbone_native_cl_20261017_v2'
    sys.path.insert(0, str(port))
    import checkpoints, onevision_bridge as bridge, runtime_utils as rt, frozen_vision as fv
    assert rt.ROOT == ROOT
    original = root_path(ready['original_run'])
    config = read(original / 'config.json')
    formal_path = root_path(ready['formal_proof']['path'])
    assert sha(formal_path) == ready['formal_proof']['sha256']
    formal = read(formal_path)
    meta_path = root_path(ready['checkpoint_metadata']['path'])
    assert sha(meta_path) == ready['checkpoint_metadata']['sha256']
    meta = read(meta_path)
    assert meta == read(original / 'checkpoints/latest.json')
    assert meta['stage'] == 19 and meta['path'] == 'checkpoints/stage_19.pt'
    assert meta['sha256'] == formal['checkpoint_sha256'] == ready['checkpoint_sha256'] == CP_SHA
    assert meta['bytes'] == ready['checkpoint_bytes'] == CP_BYTES
    assert (original / meta['path']).stat().st_size == CP_BYTES
    assert sha(original / 'config.json') == formal['config_sha256'] == ready['config_sha256'] == CONFIG_SHA
    assert config['method'] == 'PASs_MoE' and config['backbone'] == 'onevision' and config['condition'] == 'pct01'
    assert config['manifest_sha256'] == ready['manifest_sha256'] == MANIFEST_SHA
    for name, expected in config['execution_source_sha256'].items():
        assert sha(original / 'runner_source' / name) == expected
        assert sha(port / name) == expected
    # Original load performs the sole explicit full checkpoint SHA and retains
    # original CPU/CUDA0 placement. Never map all training history onto GPU.
    payload = checkpoints.load(original, expected=config)
    assert payload['stage'] == 19 and payload['format'] == 'llmcl-method-graph-v1'
    method = payload['method']; model = method.model.eval(); processor = payload['processor']
    assert model.device == torch.device('cuda:0')
    assert all(p.is_cuda and p.device.index == 0 and p.grad is None for p in model.core.parameters())
    model.requires_grad_(False)
    assert {id(p) for p in model.parameters()} == {id(p) for p in model.core.parameters()}
    assert {id(layer) for layer in method.layers} == {id(layer) for _, layer in model.core.named_modules() if hasattr(layer, '_current_E')}
    storage_path = ROOT / 'exp/ffn_static_delete_20261025/v1/core.py'
    assert sha(storage_path) == ready['storage_helper_sha256']
    storage = module('pas_resource_storage', storage_path)
    layout = storage.layout(model.core)
    assert layout['weights'] == dict(coefficients=1034446880, bytes=2350436416)
    assert layout['ordinary_buffers']['bytes'] == 10376 and layout['trainable'] == 0
    before_structure = structure(model)
    state_before = storage.digest(model.core)
    buffers_before = storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    generation_before = generation(model)
    # Keep original callbacks/history and original HostedLinear arithmetic.
    # Only the input transport/cache policy is the same as the paired run.
    old_cache_bytes = model._llmcl_visual_cache_bytes
    model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
    model._llmcl_visual_cache_hits = model._llmcl_visual_cache_misses = 0
    fv.LIMIT = 0
    def gpu_prepare(batch, row, model, original=None):
        pixels = batch.get('pixel_values')
        if pixels is None or pixels.ndim == 2:
            return False
        assert not any(p.requires_grad for p in model.visual.parameters())
        extractor = original or model._llmcl_visual_original; dtype = model.visual.get_dtype()
        with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
            features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach()
        assert features.is_cuda and features.ndim == 2 and features.shape[-1] == 896 and torch.isfinite(features).all()
        model._llmcl_visual_cache_misses += 1; batch['pixel_values'] = features
        assert model._llmcl_visual_cache_bytes == 0
        return False
    fv.prepare = gpu_prepare
    bridge.bind(model); bridge.LAST_NATIVE = None; bridge.NATIVE_GATE_MODE = False
    sys.path.insert(0, str(ASSET / 'exp/osft_increment_screen_20261019/v1'))
    import study
    path, manifest, original_rows = study.original_train()
    split, _, dev = study.split_train(manifest, original_rows)
    assert sha(path) == MANIFEST_SHA and sha(study.REPORT / 'SPLIT.json') == ready['split_sha256']
    assert split == read(study.REPORT / 'SPLIT.json')
    rows = [(task, row) for task in manifest['tasks'] for row in dev[task['task']]]
    sys.path.insert(0, str(pair))
    from paired_common import Batcher
    from paired_readouts import resource_windows
    ctx = SimpleNamespace(storage=storage, split=split, bridge=bridge, rt=rt, processor=processor)
    batcher = Batcher(ctx, model, 'PAS_RESOURCE')
    def batch_provider(m, task, row):
        # runtime_utils retains its actual ROOT data path; hash that copy too.
        for name in row['images']:
            assert sha(ROOT / name) == split['image_sha256'][name]
        method.set_instruction(task['task'])
        return batcher.readout(m, task, row)
    priors = read(pair / 'dynamic_memory.json')
    stop = dict(generation_stop_ids=processor.tokenizer.eos_token_id,
                pad_id=processor.tokenizer.pad_token_id, context_limit=None)
    gc.collect(); torch.cuda.empty_cache()
    windows, counts = resource_windows(model, processor, rows, storage,
                                       batch_provider=batch_provider, generation_policy=stop, prior_windows=priors)
    assert counts['new_generations'] == 5 and counts['new_scores'] == counts['NLL_requests'] == 0
    for current, prior in zip(windows, priors):
        for key in ('id', 'domain', 'raw_input_sha256', 'prepared_sha256', 'transport_sha256',
                    'input_shape', 'input_tensor_shapes', 'new_tokens'):
            assert current[key] == prior[key], key
    assert structure(model) == before_structure
    assert storage.digest(model.core) == state_before
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers_before
    generation_after = generation(model)
    assert all(generation_before[key]['public'] == generation_after[key]['public'] for key in generation_before)
    assert model._llmcl_visual_cache_bytes == 0
    write('memory.json', windows)
    seconds = time.monotonic() - tick
    assert seconds <= LIMIT
    write('COMPLETE.json', dict(complete=True, identity=identity, ready_sha256=ready_hash, seconds=seconds,
        physical_GPU=1, virtual_GPU=0, GPU_uuid=card['uuid'], checkpoint_sha256=CP_SHA,
        checkpoint_bytes=CP_BYTES, original_config_sha256=CONFIG_SHA, manifest_sha256=MANIFEST_SHA,
        pair_ready_sha256=ready['pair_ready_sha256'], pair_all_exit_sha256=ready['pair_all_exit_sha256'],
        memory_sha256=sha(RUN / 'memory.json'), layout=layout, required_mask_bytes=1344,
        required_weight_including_mask_bytes=2350437760, state_digest=state_before,
        all_buffers_digest=buffers_before, state_and_registered_buffers_unchanged=True,
        original_structure=before_structure, generation_before=generation_before, generation_after=generation_after,
        original_generation_objects_retained=True, private_generation_hash_may_be_process_specific=True,
        original_CPU_training_history_retained=True, original_CPU_visual_cache_removed_bytes=old_cache_bytes,
        saved_metadata_CPU_storage_bytes=692318288, saved_metadata_extra_GPU_plain_stat_bytes=2688,
        history_sizes_are_sealed_checkpoint_metadata_not_runtime_RSS=True,
        no_reconstruction_or_export=True, original_mixed_dtype=True,
        new_generations=5, forced_new_tokens=160, new_scores=0, NLL_requests=0,
        backwards=0, optimizer_updates=0, latency_measured=False,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'accelerate')},
        windows_are_five_inputs_not_full_distribution=True, no_method_selection_or_quality_claim=True))
    signal.alarm(0)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        write('FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc(), no_quality_claim=True))
        raise
