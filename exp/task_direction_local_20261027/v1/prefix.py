"""Read-only sealed S15 provenance and GPU shared-folded common starting graph.

Import and validate_metadata are stdlib only. load_shared_folded uses the
original trusted full-graph loader on physical GPU1, then the unchanged shared
fold. Its new seed42 deliberately replaces the loaded historical RNG stream.
"""
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import time

COMPLETE_SHA = 'ae5847e12d2773b54f5129dc59cd22921795c83556bc72a25d6b120e04af9de0'
CHECKPOINT_SHA = 'efb578fe8bd65430284d5d5e305ad0eb55946d06ef623265f304db3ba2ab69fe'
OLD_BINDING_SHA = 'dd2969ea9ba8d20286fc5e5e5a76db74fd74808fddbc12b5bb77088908423ae0'
VALIDATOR_SHA = '83fb92e4a8101f9b747c1ca258ebe1d778490c1f5a11f0cceeca1723586518c2'
UID = 1084


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def jd(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def owned(path, uid):
    path = Path(path)
    assert path.is_file() and not path.is_symlink() and path.stat().st_uid == uid, str(path)
    return path


def validate_metadata(binding):
    """Verify old COMPLETE/files/full-CP SHA without Torch or deserialization.

    binding.prefix contains run, complete_sha256, checkpoint_sha256,
    checkpoint_bytes, prefix_state_sha256, metadata_sha256, GPU=1, uid=1084,
    and generation_evidence={path:absolute,sha256}. No original module globals
    are replaced. The inherited validator performs the full-CP hash once here.
    """
    p = binding['prefix']
    assert os.name == 'posix' and os.getuid() == p['uid'] == UID
    assert p['GPU'] == 1 and p['complete_sha256'] == COMPLETE_SHA
    assert p['checkpoint_sha256'] == CHECKPOINT_SHA
    run = Path(p['run'])
    assert run.is_absolute() and run.name == 'prefix' and not run.is_symlink()
    complete_path = owned(run/'prefix_COMPLETE.json', UID)
    assert sha(complete_path) == COMPLETE_SHA
    old_root = run.parent.parent
    validator_path = owned(old_root/'exp/type_auxiliary_supervision_20261027/v2/controller.py', UID)
    assert sha(validator_path) == VALIDATOR_SHA
    outer = read(run.parent/'PHASE_prefix_prefix_STARTED.json')
    command = outer['command']; assert command.count('--binding') == 1
    old_binding_path = owned(command[command.index('--binding')+1], UID)
    assert sha(old_binding_path) == OLD_BINDING_SHA
    # Preserve all of the old validator's inherited-data/source constraints.
    # This is a separate input dictionary, not a modification of its module.
    inherited = dict(read(old_binding_path))
    inherited['reused_prefix'] = dict(run=str(run), GPU=1,
        complete_sha256=COMPLETE_SHA, checkpoint_sha256=CHECKPOINT_SHA,
        binding_sha256=OLD_BINDING_SHA)
    validator = module('local_direction_original_prefix_validator', validator_path)
    original = validator.validate_reused_prefix(inherited)
    complete = read(complete_path)
    assert complete['identity']['uid'] == UID and complete['GPU'] == 1
    state_path = owned(run/'PREFIX_STATE.json', UID)
    metadata_path = owned(run/'checkpoints/stage_15.json', UID)
    assert sha(state_path) == p['prefix_state_sha256'] == complete['files']['PREFIX_STATE.json']
    assert sha(metadata_path) == p['metadata_sha256'] == complete['files']['checkpoints/stage_15.json']
    assert sha(run/'checkpoints/latest.json') == p['metadata_sha256']
    state, metadata = read(state_path), read(metadata_path)
    assert state['checkpoint'] == metadata == complete['checkpoint'] == original['checkpoint']
    assert metadata['path'] == 'checkpoints/stage_15.pt' and metadata['stage'] == 15
    assert metadata['bytes'] == p['checkpoint_bytes'] and metadata['sha256'] == CHECKPOINT_SHA
    payload = owned(run/metadata['path'], UID)
    assert payload.stat().st_size == p['checkpoint_bytes']
    evidence_path = Path(p['generation_evidence']['path'])
    assert evidence_path.is_absolute() and sha(evidence_path) == p['generation_evidence']['sha256']
    evidence = read(evidence_path)
    assert evidence['stage'] == 15 and evidence['checkpoint_sha256'] == CHECKPOINT_SHA
    assert set(evidence['configs']) == {'outer', 'language'}
    for config in evidence['configs'].values():
        assert jd(config['public']) == config['public_sha256']
        assert config['from_model_config'] and config['nondefault_model_generation'] == {}
        assert config['derived_equals_public'] and config['derived_public'] == config['public']
    return dict(run=str(run), GPU=1, uid=UID, complete_sha256=COMPLETE_SHA,
        checkpoint_sha256=CHECKPOINT_SHA, checkpoint_bytes=metadata['bytes'],
        prefix_state_sha256=p['prefix_state_sha256'], metadata_sha256=p['metadata_sha256'],
        generation_evidence=dict(path=str(evidence_path), sha256=p['generation_evidence']['sha256']),
        original_validation=original, trusted_private_checkpoint_owner_verified=True,
        full_checkpoint_SHA_checked_without_deserialization=True)


def dependency(ctx, binding, relative):
    path = Path(relative)
    assert not path.is_absolute() and '..' not in path.parts
    expected = binding['inherited_sources'][relative]
    actual = Path(ctx.root)/path
    assert actual.is_file() and sha(actual) == expected, relative
    return actual


def generation_contract(model, evidence):
    """Public fields are stable; historical process-local Python hashes are not."""
    from transformers import GenerationConfig
    result = {}
    for name, core in (('outer', model.core), ('language', model.core.language_model)):
        config = core.generation_config
        public = config.to_dict(); expected = evidence['configs'][name]
        assert public == expected['public'] and jd(public) == expected['public_sha256']
        assert config._from_model_config is True
        assert core.config._get_non_default_generation_parameters() == {}
        assert GenerationConfig.from_model_config(core.config).to_dict() == public
        result[name] = jd(public)
    return dict(public_sha256=result, original_public_fields_unchanged=True,
                historical_process_local_hash_not_compared=True)


def load_shared_folded(ctx, binding):
    """Return (frozen dense model, small proof); no new QA or optimizer work.

    ctx supplies root, gpu=1, device='cuda:1', storage, and bind(model).
    ctx.processor is replaced by the trusted CP's original processor. The caller
    may install the new local-r8 method only after this function returns.
    """
    began = time.monotonic()
    provenance = validate_metadata(binding)
    import numpy as np
    import torch
    assert ctx.gpu == provenance['GPU'] == 1
    assert torch.device(ctx.device) == torch.device('cuda:1') and torch.cuda.current_device() == 1
    assert os.environ.get('CUDA_VISIBLE_DEVICES') in (None, '0,1,2,3')
    checkpoint_path = dependency(ctx, binding, 'exp/two_backbone_native_cl_20261017_v2/checkpoints.py')
    shared_path = dependency(ctx, binding, 'exp/conditional_shared_budget_20261026/v1/adaptation.py')
    package_path = dependency(ctx, binding, 'exp/plain_r8_gpu_control_20261025/v1/control_deployment.py')
    storage_path = dependency(ctx, binding, 'exp/ffn_static_delete_20261025/v1/core.py')
    assert sha(Path(ctx.storage.__file__)) == sha(storage_path)
    checkpoints = module('local_direction_original_checkpoint_loader', checkpoint_path)
    shared = module('local_direction_original_shared_fold', shared_path)
    package = module('local_direction_dense_receipt', package_path)
    state = read(Path(provenance['run'])/'PREFIX_STATE.json')
    # Original loader independently checks the CP SHA and restores its exact
    # historical placements/RNG. Do not monkey-patch or remap its globals.
    loaded = checkpoints.load(Path(provenance['run']), expected=state['config'])
    assert loaded['format'] == 'llmcl-method-graph-v1' and loaded['stage'] == 15
    method, model = loaded['method'], loaded['method'].model
    ctx.processor = loaded['processor']
    ctx.bind(model)
    assert all(p.device == torch.device('cuda:1') for p in model.parameters())
    core_before = ctx.storage.digest(model.core)
    buffers_before = ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    assert core_before == state['model_digest'] and buffers_before == state['buffers_digest']
    expected_plan = {f'model.layers.{i}.self_attn.{part}_proj':8 for i in range(24) for part in ('q','v')}
    assert method.rank_plan == expected_plan and model._conditional_arm == 'shared'
    assert not model._conditional_deployed
    ab = [(n,p) for n,p in model.named_parameters() if n.endswith(('.A.weight','.B.weight'))]
    assert len(ab) == 96 and sum(p.numel() for _,p in ab) == 540672
    assert all(p.dtype == torch.float32 and torch.isfinite(p).all() for _,p in ab)
    evidence = read(Path(provenance['generation_evidence']['path']))
    public_before = generation_contract(model, evidence)
    # Same existing shared fold; no factor reinitialization or compensation.
    shared.configure_gpu_math()
    model.requires_grad_(False).eval()
    fold = shared.deploy(model, method)
    assert fold['arm'] == 'shared' and fold['explicit_bank'] is False
    assert len(fold['folded_modules']) == 48 and {x['module'] for x in fold['folded_modules']} == set(expected_plan)
    assert not any(n.endswith(('.A.weight','.B.weight')) for n,_ in model.named_parameters())
    assert not any(p.requires_grad for p in model.parameters())
    dense = package.receipt(model, ctx.storage)
    assert dense['layout']['weights'] == dict(coefficients=893675552, bytes=1787351104)
    assert dense['all_buffers_digest'] == buffers_before
    assert generation_contract(model, evidence) == public_before
    del loaded, method, ab
    gc.collect()
    # This is a new paired intervention, not continuation of the old RNG path.
    random.seed(42); np.random.seed(42)
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    return model, dict(prefix=provenance, strict_loaded_prefix_core_digest=core_before,
        strict_loaded_prefix_buffers_digest=buffers_before, generation=public_before,
        shared_fold=fold, common_dense=dense, new_common_seed=42,
        old_RNG_restored_by_loader_then_deliberately_replaced=True,
        exact_old_explicit_AB_output_equivalence_not_claimed=True,
        reused_prefix_FIT=959, reused_prefix_updates=69, new_training_updates=0,
        new_generations=0, new_scores=0, seconds=time.monotonic()-began)
