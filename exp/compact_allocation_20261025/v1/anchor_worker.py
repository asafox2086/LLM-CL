"""WIP: reconstruct the sealed all-r8 deployment graph and generate 100 DEV.

No old Python graph is loaded on this host; the complete learned state is the
portable A/B payload plus a byte-matched original frozen base and all buffers.
"""
from pathlib import Path
import gc
import hashlib
import importlib.util
import json
import os
import signal
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(os.environ['LLMCL_ROOT'])
RUN = Path(os.environ['LLMCL_RUN'])
ASSET = Path(os.environ['LLMCL_ASSET'])


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    with (RUN / name).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    ready = read(RUN / 'READY.json')
    for name, expected in ready['sources'].items():
        assert sha(RUN / name) == expected, name
    for name, expected in ready['runtime_files'].items():
        assert sha(ASSET / name) == expected, name
    for name, expected in ready['prior_files'].items():
        assert sha(RUN / name) == expected, name
    prior_ready_path = RUN / 'prior_cf_READY.json'
    assert sha(prior_ready_path) == 'faf79752df6031b15aa87d640ff7046ce06e4b7498ce95726cd96adb085436a5'
    prior_ready = read(prior_ready_path)
    config_name = 'model/LLaVA-OneVision-Qwen2-0.5B-OV/config.json'
    config_sha = prior_ready['runtime_files'][config_name]
    assert ready['runtime_files'][config_name] == config_sha
    gpu = int(os.environ['LLMCL_GPU'])
    spec = importlib.util.spec_from_file_location('anchor_ids', ASSET / 'exp/paired_image_support_20261024/v1/common.py')
    ids = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ids)
    card = ids.inventory()[gpu]
    assert not card['owners'] and card['memory_MiB'] < 500 and card['utilization'] < 10
    who = ids.identity(os.getpid())
    assert who['pid'] == who['pgid']
    tick = time.monotonic()
    write('anchor_STARTED.json', dict(unix=time.time(), identity=who, GPU=gpu,
          boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
          ready_sha256=sha(RUN / 'READY.json')))

    def expired(*args):
        raise TimeoutError('Frozen 450 second anchor budget expired')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(450)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8',
                      MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0, str(ASSET / 'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0, str(ASSET / 'exp/osft_increment_screen_20261019/v1'))
    import onevision_bridge as bridge
    import frozen_vision as fv
    import study
    import adaptation
    from runtime_utils import input_batch
    torch.set_num_threads(8)
    storage_path = ROOT / 'exp/ffn_static_delete_20261025/v1/core.py'
    assert sha(storage_path) == ready['storage_helper_sha256']
    spec = importlib.util.spec_from_file_location('anchor_storage', storage_path)
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    asset = ready['anchor_asset']
    portable_path = RUN / asset['path']
    assert portable_path.parent == RUN and portable_path.stat().st_size == asset['bytes']
    assert sha(portable_path) == asset['sha256']
    portable = torch.load(portable_path, map_location='cpu', weights_only=True)
    receipt = portable['receipt']
    extracted = read(RUN / 'anchor_extract_COMPLETE.json')
    assert all(receipt[k] == extracted[k] for k in receipt)
    assert extracted['asset'] == asset
    assert receipt['format'] == 'llmcl-sealed-all-adapters-v1' and receipt['stage'] == 19
    assert receipt['source_checkpoint']['sha256'] == 'e05f7780263e313f5422ee6b547515631ad76bccd1752a51873947ab6658866c'
    assert (receipt['FIT_QA'], receipt['original_updates'], receipt['TEST_QA']) == (1213, 87, 0)
    parent = read(RUN / 'prior_parent_COMPLETE.json')
    assert receipt['attention_classes'] == parent['language_attention_classes']
    assert receipt['nonvision_buffers_digest'] == parent['preserved_nonvision_buffers_digest']
    base = ASSET / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    # Factory preserves FP32 derived buffers. Never call model.half().
    model = bridge.Factory.from_pretrained(base, torch_dtype=torch.float16,
                attn_implementation='sdpa', local_files_only=True)
    assert storage.digest(model.core) == receipt['non_adapter_state_digest']
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == receipt['all_buffers_digest']
    assert all(str(p.dtype) == 'torch.float16' for p in model.core.parameters())
    assert model.core.language_model.model.embed_tokens.weight is model.core.language_model.lm_head.weight
    assert model.model is model.core.language_model.model and model.lm_head is model.core.language_model.lm_head
    assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
    assert model.visual.image_newline is model.core.image_newline
    assert model.core.language_model.config._attn_implementation == receipt['language_attention_backend'] == 'sdpa'
    assert model.core.vision_tower.config._attn_implementation == receipt['vision_attention_backend'] == 'sdpa'
    assert sorted({type(l.self_attn).__name__ for l in model.model.layers}) == receipt['attention_classes']
    method = study.TargetLoRA(model, study.QV + study.EXTRA)
    plan = receipt['rank_plan']
    expected_plan = {f'model.layers.{i}.{p}': 8 for i in range(24)
                     for p in ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                               'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')}
    assert plan == expected_plan and len(method.layers) == 168
    assert {id(x) for x in method.layers} == {id(model.get_submodule(n)) for n in plan}
    expected_keys = {'language_model.' + n + s for n in plan for s in ('.A.weight', '.B.weight')}
    adapters = portable['adapters']
    assert set(adapters) == expected_keys == set(receipt['adapter_keys']) and len(adapters) == 336
    assert sum(v.numel() for v in adapters.values()) == receipt['adapter_coordinates'] == 4399104
    assert sum(v.numel() * v.element_size() for v in adapters.values()) == receipt['adapter_bytes'] == 17596416
    assert storage.digest(adapters) == receipt['adapter_digest']
    parameters = dict(model.core.named_parameters())
    assert {n for n, p in parameters.items() if p.requires_grad} == expected_keys
    for name, rank in plan.items():
        layer = model.get_submodule(name)
        saved = receipt['modules'][name]
        assert rank == saved['rank'] == 8 and layer.scale == saved['scale'] == 4
        assert layer.dropout.p == saved['dropout'] == .1
        assert list(layer.weight.shape) == saved['weight_shape']
        assert str(layer.weight.dtype) == saved['weight_dtype'] == 'torch.float16'
        assert (layer.bias is not None) == saved['bias']
    with torch.no_grad():
        for name in sorted(expected_keys):
            value = adapters[name]
            target = parameters[name]
            assert value.dtype == target.dtype == torch.float32 and torch.isfinite(value).all()
            assert list(value.shape) == list(target.shape) == receipt['adapter_shapes'][name]
            assert str(value.dtype) == receipt['adapter_dtypes'][name]
            target.copy_(value)
    assert storage.digest({n: parameters[n] for n in expected_keys}) == receipt['adapter_digest']
    before_non_ab = storage.digest({n: v for n, v in model.core.state_dict().items() if n not in expected_keys})
    assert before_non_ab == receipt['non_adapter_state_digest']
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == receipt['all_buffers_digest']
    changed_keys = {'language_model.' + n + '.weight' for n in plan}
    immutable_before = storage.digest({n: v for n, v in model.core.state_dict().items()
                                      if n not in changed_keys and n not in expected_keys})
    # The inherited arithmetic accepts a minimal namespace; no new fold recipe.
    folding = SimpleNamespace(rank_plan=plan, layers=method.layers)
    fold = adaptation.fold(model, folding)
    assert len(fold) == 168 and not method.layers and not folding.layers
    immutable_after = storage.digest({n: v for n, v in model.core.state_dict().items()
                                     if n not in changed_keys})
    assert immutable_after == immutable_before
    assert not any(n.endswith(('.A.weight', '.B.weight')) for n in model.core.state_dict())
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == receipt['all_buffers_digest']
    assert all(not p.requires_grad and p.grad is None for p in model.core.parameters())
    assert storage.layout(model) == parent['layout']
    del portable, adapters, parameters, folding, method, layer, target, value
    gc.collect()
    model.cuda().eval()
    bridge.bind(model)
    fv.LIMIT = 0
    fv.install(model)
    assert model.visual.tower is model.core.vision_tower and model.visual.projector is model.core.multi_modal_projector
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == receipt['all_buffers_digest']
    before = storage.digest(model)
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    policy = read(ASSET / 'exp/two_backbone_native_cl_20261017_v2/execution_policy.json')
    manifest_path = ASSET / policy['conditions']['pct01']['manifest']
    assert sha(manifest_path) == ready['manifest_sha256'] == receipt['source_checkpoint']['manifest_sha256']
    manifest = read(manifest_path)
    split_path = ASSET / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
    assert sha(split_path) == ready['split_sha256']
    split = read(split_path)
    prior = {r['id']: r for r in read(RUN / 'prior_parent_predictions.json')}
    rows = []
    for task in manifest['tasks']:
        source = ASSET / task['files']['train']['path']
        assert sha(source) == task['files']['train']['sha256']
        wanted = set(split['dev_ids'][str(task['task'])])
        rows.extend((task, r) for r in map(json.loads, source.read_bytes().splitlines()) if r['id'] in wanted)
    assert len(rows) == len(prior) == 100 and {r['id'] for _, r in rows} == set(prior)
    assert all(sum(t['domain'] == domain for t, _ in rows) == 20
               for domain in ('natural', 'medical', 'geography', 'finance', 'traffic'))

    class FiniteScores:
        def __call__(self, input_ids, scores):
            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
            assert torch.isfinite(scores).any(dim=-1).all()
            return scores

    predictions = []
    for task, row in rows:
        for name in row['images']:
            assert sha(ASSET / name) == split['image_sha256'][name]
        raw = {}

        def capture(module, args):
            assert not raw
            raw['raw_pixels_sha256'] = storage.digest({'pixels': args[0]})

        handle = model.core.vision_tower.register_forward_pre_hook(capture)
        try:
            batch, _ = input_batch(row, task, processor, model.device)
        finally:
            handle.remove()
        old = prior[row['id']]
        prepared = storage.digest(batch)
        transport = storage.digest({k: v for k, v in batch.items() if k != 'pixel_values'})
        assert prepared == old['prepared_sha256'] and transport == old['transport_sha256']
        assert len(batch['pixel_values']) == old['packed_tokens']
        assert raw['raw_pixels_sha256'] == old['raw_pixels_sha256']
        assert row['answers'] == old['references'] and task['max_new_tokens'] == old['generation_cap']
        with torch.inference_mode(), torch.autocast('cuda', dtype=model.dtype):
            out = model.generate(**batch, do_sample=False, num_beams=1,
                max_new_tokens=task['max_new_tokens'], use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id, eos_token_id=processor.tokenizer.eos_token_id,
                logits_processor=[FiniteScores()])
        tokens = out[0, batch['input_ids'].shape[1]:].tolist()
        predictions.append(dict(id=row['id'], task=task['task'], domain=task['domain'],
            prepared_sha256=prepared, transport_sha256=transport, packed_tokens=len(batch['pixel_values']),
            **raw, prediction=processor.tokenizer.decode(tokens, skip_special_tokens=True),
            output_tokens=tokens, references=row['answers'], generation_cap=task['max_new_tokens']))
        del batch, out
    assert len(predictions) == 100 and model._llmcl_visual_cache_bytes == 0
    after = storage.digest(model)
    assert before == after
    assert storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == receipt['all_buffers_digest']
    assert all(not p.requires_grad and p.grad is None for p in model.core.parameters())
    write('anchor_predictions.json', predictions)
    import resource
    write('anchor_COMPLETE.json', dict(identity=who, ready_sha256=sha(RUN / 'READY.json'),
          source_checkpoint=receipt['source_checkpoint'], adapter_asset=asset,
          adapter_digest=receipt['adapter_digest'], adapter_parameters=336, adapter_coordinates=4399104,
          copied_complete_cumulative_AB=True, fresh_base_non_AB_exact=True,
          non_adapter_state_digest=before_non_ab, all_buffers_digest=receipt['all_buffers_digest'],
          immutable_before_fold_digest=immutable_before, immutable_after_fold_digest=immutable_after,
          nonvision_buffers_digest=receipt['nonvision_buffers_digest'],
          attention_classes=receipt['attention_classes'], fold=fold,
          digest_before=before, digest_after=after, layout=storage.layout(model),
          embedding_head_tied=model.core.language_model.model.embed_tokens.weight is model.core.language_model.lm_head.weight,
          bridge_core_and_vision_aliases=True, no_full_checkpoint_written=True,
          native_config_file_sha256=config_sha,
          non_tensor_configuration_basis='Original sealed Factory/TargetLoRA construction sources and unchanged native base config provenance; tensor digests alone do not prove all non-tensor state.',
          new_generations=100, new_scores=0, backwards=0, optimizer_updates=0,
          FIT_model_requests=0, NLL_requests=0, TEST_QA=0,
          process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
          files={'anchor_predictions.json': sha(RUN / 'anchor_predictions.json')},
          seconds=time.monotonic() - tick))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write('anchor_FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc()))
        raise
