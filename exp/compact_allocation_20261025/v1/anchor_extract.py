"""WIP: CPU-only extraction from the sealed, trusted FIT LoRA-all graph.

No forward, scoring, optimizer, CUDA placement or old checkpoint loader runs.
The portable payload contains only the complete learned A/B tensors and receipts.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import os
import pickle
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    root = Path(os.environ['LLMCL_ROOT'])
    assets = Path(os.environ.get('LLMCL_ASSET', str(root)))
    output = Path(os.environ['LLMCL_RUN'])
    checkpoint = Path(os.environ['LLMCL_ANCHOR_CP'])
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert output.is_dir()
    assert not (output / 'anchor_adapters.pt').exists()
    assert not (output / 'anchor_extract_COMPLETE.json').exists()
    tick = time.monotonic()
    inherited_ready = Path(os.environ.get('LLMCL_CF_READY', str(output / 'prior_cf_READY.json')))
    assert sha(inherited_ready) == 'faf79752df6031b15aa87d640ff7046ce06e4b7498ce95726cd96adb085436a5'
    ready = read(inherited_ready)
    prefixes = ('exp/two_backbone_native_cl_20261017_v2/',
                'exp/osft_increment_screen_20261019/v1/')
    runtime = {n: h for n, h in ready['runtime_files'].items() if n.startswith(prefixes)}
    for required in ('methods.py', 'checkpoints.py', 'onevision_bridge.py', 'frozen_vision.py'):
        assert 'exp/two_backbone_native_cl_20261017_v2/' + required in runtime
    assert 'exp/osft_increment_screen_20261019/v1/study.py' in runtime
    for name, expected in runtime.items():
        assert sha(assets / name) == expected, name
    storage_path = root / 'exp/ffn_static_delete_20261025/v1/core.py'
    assert sha(storage_path) == ready['storage_helper_sha256']
    sealed_path = Path(os.environ.get('LLMCL_ANCHOR_SEALED', str(assets / 'summary/edge_deployment_20261019/osft_increment_screen/v1/sealed_case_reports_20261022/LoRA_all_VERIFIED.json')))
    assert sha(sealed_path) == '6003d1a794c2f9ea3b9faeca59ba1c026afb7e3d1a69988011f7977f1ef62b46'
    sealed = read(sealed_path)
    assert sealed['case'] == 'LoRA_all' and sealed['original_independent_verifier_passed'] is True
    assert (sealed['FIT_QA'], sealed['DEV_QA'], sealed['TEST_QA'],
            sealed['positive_finite_updates']) == (1213, 100, 0, 87)
    expected_cp = sealed['checkpoint']
    assert expected_cp['stage'] == 19 and expected_cp['method'] == 'LoRA_all'
    assert expected_cp['full_architecture_and_runtime'] is True
    assert expected_cp['sha256'] == 'e05f7780263e313f5422ee6b547515631ad76bccd1752a51873947ab6658866c'
    assert checkpoint.stat().st_size == expected_cp['bytes'] == 1941592509
    # The original 1.94 GB artifact is hashed exactly once here, before pickle.
    cp_sha = sha(checkpoint)
    assert cp_sha == expected_cp['sha256']
    os.environ.update(OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    sys.path.insert(0, str(assets / 'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0, str(assets / 'exp/osft_increment_screen_20261019/v1'))
    import torch
    import transformers
    import study
    import methods
    import onevision_bridge
    import frozen_vision
    torch.set_num_threads(8)
    spec = importlib.util.spec_from_file_location('anchor_storage', storage_path)
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    assert not torch.cuda.is_initialized()
    # This is our trusted graph, not an arbitrary uploaded pickle. Do not call
    # checkpoints.load(): it restores original CUDA placements and old RNG.
    payload = torch.load(checkpoint, map_location='cpu', pickle_module=pickle, weights_only=False)
    assert payload['format'] == 'llmcl-method-graph-v1' and payload['stage'] == 19
    config = payload['config']
    assert config['method'] == 'LoRA_all'
    assert (config['FIT_QA'], config['DEV_QA'], config['TEST_QA']) == (1213, 100, 0)
    assert config['manifest_sha256'] == expected_cp['manifest_sha256']
    assert config['manifest_sha256'] == ready['manifest_sha256']
    assert config['split_sha256'] == ready['split_sha256']
    assert config['seed'] == 42 and config['rank_for_LoRA'] == 8
    method = payload['method']
    model = method.model
    core = model.core
    assert model.model is core.language_model.model and model.lm_head is core.language_model.lm_head
    assert model.visual.tower is core.vision_tower
    assert model.visual.projector is core.multi_modal_projector
    assert model.visual.image_newline is core.image_newline
    assert all(p.device.type == 'cpu' for p in core.parameters())
    assert all(v.device.type == 'cpu' for _, v in core.named_buffers(remove_duplicate=False))
    projections = ('self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj',
                   'self_attn.o_proj', 'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')
    rank_plan = {f'model.layers.{i}.{p}': 8 for i in range(24) for p in projections}
    assert len(core.language_model.model.layers) == 24 and len(rank_plan) == 168
    assert len(method.layers) == 168
    assert {id(layer) for layer in method.layers} == {id(model.get_submodule(n)) for n in rank_plan}
    expected_keys = {'language_model.' + n + ending for n in rank_plan
                     for ending in ('.A.weight', '.B.weight')}
    parameters = dict(core.named_parameters())
    assert {n for n in parameters if n.endswith(('.A.weight', '.B.weight'))} == expected_keys
    assert {n for n, p in parameters.items() if p.requires_grad} == expected_keys
    adapters = {}
    modules = {}
    for name, rank in rank_plan.items():
        layer = model.get_submodule(name)
        assert isinstance(layer, methods.SeqLinear)
        assert layer.scale == 4 and layer.dropout.p == .1
        assert layer.weight.dtype == torch.float16
        assert tuple(layer.A.weight.shape) == (rank, layer.weight.shape[1])
        assert tuple(layer.B.weight.shape) == (layer.weight.shape[0], rank)
        modules[name] = dict(rank=rank, scale=layer.scale, dropout=layer.dropout.p,
                             weight_shape=list(layer.weight.shape), weight_dtype=str(layer.weight.dtype),
                             bias=layer.bias is not None)
        for suffix in ('.A.weight', '.B.weight'):
            key = 'language_model.' + name + suffix
            value = parameters[key]
            assert value.dtype == torch.float32 and bool(torch.isfinite(value).all())
            adapters[key] = value.detach().clone()
    assert len(adapters) == 336 and sum(t.numel() for t in adapters.values()) == 4399104
    assert sum(t.numel() * t.element_size() for t in adapters.values()) == 17596416
    state = core.state_dict()
    non_adapters = {n: v for n, v in state.items() if n not in expected_keys}
    buffers = list(core.named_buffers(remove_duplicate=False))
    nonvision_buffers = [(n, v) for n, v in buffers
                         if not n.startswith(('vision_tower.', 'multi_modal_projector.'))]
    buffer_metadata = {n: dict(shape=list(v.shape), dtype=str(v.dtype),
                               bytes=v.numel() * v.element_size()) for n, v in buffers}
    tied = core.language_model.model.embed_tokens.weight is core.language_model.lm_head.weight
    assert tied and all(p.dtype == torch.float16 for n, p in parameters.items() if n not in expected_keys)
    attention_classes = sorted({type(layer.self_attn).__name__ for layer in model.model.layers})
    receipt = dict(format='llmcl-sealed-all-adapters-v1', stage=19, FIT_QA=1213,
                   original_updates=87, original_DEV_QA=100, TEST_QA=0,
                   source_checkpoint=dict(bytes=expected_cp['bytes'], sha256=cp_sha,
                                          manifest_sha256=expected_cp['manifest_sha256']),
                   source_sealed_sha256=sha(sealed_path), inherited_READY_sha256=sha(inherited_ready),
                   runtime_files=runtime, storage_helper_sha256=ready['storage_helper_sha256'],
                   adapter_keys=sorted(adapters), adapter_shapes={n: list(t.shape) for n, t in adapters.items()},
                   adapter_dtypes={n: str(t.dtype) for n, t in adapters.items()},
                   adapter_parameters=336, adapter_coordinates=4399104, adapter_bytes=17596416,
                   adapter_digest=storage.digest(adapters), rank_plan=rank_plan, modules=modules,
                   non_adapter_state_digest=storage.digest(non_adapters),
                   all_buffers_digest=storage.digest(buffers), all_buffers=buffer_metadata,
                   nonvision_buffers_digest=storage.digest(nonvision_buffers),
                   attention_classes=attention_classes,
                   language_attention_backend=core.language_model.config._attn_implementation,
                   vision_attention_backend=core.vision_tower.config._attn_implementation,
                   embedding_head_tied=tied, bridge_core_and_vision_aliases=True,
                   original_layout=storage.layout(model), torch_version=str(torch.__version__),
                   transformers_version=str(transformers.__version__))
    assert receipt['language_attention_backend'] == receipt['vision_attention_backend'] == 'sdpa'
    assert receipt['original_layout']['weights'] == dict(coefficients=898074656, bytes=1804947520)
    path = output / 'anchor_adapters.pt'
    assert not path.exists()
    torch.save(dict(adapters=adapters, receipt=receipt), path)
    assert not torch.cuda.is_initialized()
    write(output / 'anchor_extract_COMPLETE.json', dict(**receipt,
          asset=dict(path=path.name, bytes=path.stat().st_size, sha256=sha(path)),
          extract_source_sha256=sha(Path(__file__)), seconds=time.monotonic() - tick,
          CUDA_initialized=False, model_forward_requests=0, new_generations=0,
          new_scores=0, optimizer_updates=0, backwards=0))


if __name__ == '__main__':
    try:
        main()
    except Exception:
        write(Path(os.environ['LLMCL_RUN']) / 'anchor_extract_FAILURE.json',
              dict(unix=time.time(), traceback=traceback.format_exc()))
        raise
