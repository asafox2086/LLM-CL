"""Standalone fixed-budget three-arm train/eval worker. Not executed on import.

No old checkpoints, old scores, service or GPU allocator are consumed here.
Train: the original 87-update prefix plus RNG-neutral task before/acquired.
Eval: a separate process, self-contained strict reload, five windows then DEV.
"""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from types import SimpleNamespace

import adaptation

MANIFEST_SHA = '36993693bfdddb0c3143cc842edfb2b5572628726146263cc0c9207ed4281777'
SPLIT_SHA = 'd8e2dd59767c53eb833988dda87b4330ac208d2c804e42e327725f09d38bcb9d'
AST_SHA = '2b31ff9e06234e967ea42d3cb6d594c82050054c7c2c058d77f61bf8100dda7f'
STORAGE_SHA = '032904b7fe92a0912e18c0e664750c35f4c6fa7bf15fae6fa9a5a350f7277ef8'
FORMAT = 'conditional-shared-r8-complete-v1'
LIMITS = {'train': 5400, 'eval': 1800}


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class Context:
    def __init__(self, args):
        self.root, self.run = args.root, args.run
        self.arm, self.phase, self.gpu = args.arm, args.phase, args.gpu
        assert self.root.is_absolute() and self.run.is_absolute()
        assert self.run.is_dir() and self.gpu in range(4)
        self.start = time.monotonic()
        os.environ.update(LLMCL_ROOT=str(self.root), LLMCL_RUN=str(self.run),
            LLMCL_ASSET=str(self.root), OMP_NUM_THREADS='8', MKL_NUM_THREADS='8',
            TOKENIZERS_PARALLELISM='false')
        self.port = self.root / 'exp/two_backbone_native_cl_20261017_v2'
        self.plain = self.root / 'exp/plain_r8_gpu_control_20261025/v1'
        self.ids = module('conditional_proc_ids', self.root / 'exp/paired_image_support_20261024/v1/common.py')
        self.who = self.ids.identity(os.getpid())
        assert self.who and self.who['pid'] == self.who['pgid']
        self.inventory = self.ids.inventory()[self.gpu]
        assert not self.inventory['owners'], 'Caller must obtain an ownerless GPU from the existing allocator'
        write(self.run / (self.phase + '_STARTED.json'), dict(unix=time.time(),
            identity=self.who, arm=self.arm, phase=self.phase, GPU=self.gpu,
            boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            limit_seconds=LIMITS[self.phase], inventory=self.inventory))
        def expired(*unused):
            raise TimeoutError('Fixed conditional-r8 ' + self.phase + ' deadline')
        signal.signal(signal.SIGALRM, expired)
        signal.alarm(LIMITS[self.phase])
        import torch
        torch.set_num_threads(8)
        torch.cuda.set_device(self.gpu)
        self.device = f'cuda:{self.gpu}'
        adaptation.configure_gpu_math()
        sys.path.insert(0, str(self.port))
        sys.path.insert(0, str(self.root / 'exp/osft_increment_screen_20261019/v1'))
        import onevision_bridge as bridge, runtime_utils as rt, frozen_vision as fv
        self.bridge, self.rt, self.fv = bridge, rt, fv
        self.study = module('conditional_original_study', self.root / 'exp/osft_increment_screen_20261019/v1/study.py')
        self.study.ROOT, self.study.PORT = self.root, self.port
        storage_path = self.root / 'exp/ffn_static_delete_20261025/v1/core.py'
        assert sha(storage_path) == STORAGE_SHA
        self.storage = module('conditional_storage', storage_path)
        self.old_deploy = module('conditional_package_operators', self.plain / 'control_deployment.py')
        self.old_common = module('conditional_batch_operators', self.plain / 'common.py')
        self.readouts = module('conditional_readout_operators', self.plain / 'readouts.py')
        self.base = self.root / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(
            self.base if self.phase == 'train' else self.run / 'processor', local_files_only=True)
        self.stop = dict(generation_stop_ids=self.processor.tokenizer.eos_token_id,
            pad_id=self.processor.tokenizer.pad_token_id, context_limit=None)
        manifest_path, self.manifest, original = self.study.original_train()
        assert sha(manifest_path) == MANIFEST_SHA
        split_path = self.root / 'summary/edge_deployment_20261019/osft_increment_screen/v1/SPLIT.json'
        assert sha(split_path) == SPLIT_SHA
        self.split = read(split_path)
        self.fit, self.dev = {}, {}
        for task in self.manifest['tasks']:
            stage = task['task']
            by_id = {row['id']: row for row in original[stage]}
            assert len(by_id) == len(original[stage])
            fids, dids = self.split['fit_ids'][str(stage)], self.split['dev_ids'][str(stage)]
            assert not set(fids).intersection(dids) and set(fids + dids) == set(by_id)
            self.fit[stage] = [by_id[i] for i in fids]
            self.dev[stage] = [by_id[i] for i in dids]
        assert sum(map(len, self.fit.values())) == 1213
        assert sum(map(len, self.dev.values())) == 100
        self.rows = [(t, r) for t in self.manifest['tasks'] for r in self.dev[t['task']]]
        fv.LIMIT = 0
        def gpu_prepare(batch, row, model, original=None):
            pixels = batch.get('pixel_values')
            if pixels is None or pixels.ndim == 2:
                return False
            assert not any(p.requires_grad for p in model.visual.parameters())
            extractor = original or model._llmcl_visual_original
            dtype = model.visual.get_dtype()
            with torch.no_grad(), torch.autocast('cuda', dtype=dtype):
                features = extractor(pixels.to(dtype), batch['image_grid_thw']).detach()
            assert features.is_cuda and features.shape[-1] == 896 and torch.isfinite(features).all()
            batch['pixel_values'] = features
            model._llmcl_visual_cache_misses += 1
            assert model._llmcl_visual_cache_bytes == 0
            return False
        fv.prepare = gpu_prepare
        dependencies = [Path(__file__), Path(adaptation.__file__), storage_path,
            self.port / 'run.py', self.port / 'methods.py', self.port / 'onevision_bridge.py',
            self.port / 'runtime_utils.py', self.port / 'frozen_vision.py', self.port / 'retrieval.py',
            Path(self.study.__file__), self.root / 'exp/position_coordinate_control_20261022/v1/core.py',
            self.plain / 'control_adaptation.py', self.plain / 'control_deployment.py',
            self.plain / 'common.py', self.plain / 'readouts.py']
        self.source_hashes = {str(p): sha(p) for p in dependencies}
        self.data_binding = dict(manifest_path=str(manifest_path), manifest_sha256=MANIFEST_SHA,
            split_path=str(split_path), split_sha256=SPLIT_SHA,
            FIT_QA=1213, DEV_QA=100, original_TRAIN_QA=1313, TEST_QA=0)
        if self.phase == 'train':
            write(self.run / 'INPUT_BINDING.json', self.data_binding)
            write(self.run / 'SOURCE_BINDING.json', self.source_hashes)
        else:
            assert read(self.run / 'INPUT_BINDING.json') == self.data_binding
            assert read(self.run / 'SOURCE_BINDING.json') == self.source_hashes
        self.reset()

    def reset(self):
        import numpy as np
        import torch
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    def bind(self, model):
        self.bridge.bind(model)
        self.bridge.LAST_NATIVE = None
        self.bridge.NATIVE_GATE_MODE = False
        if not hasattr(model, '_llmcl_visual_original'):
            self.fv.install(model)

    def batcher(self, model, phase):
        return self.old_common.Batcher(self, model, phase)

    def end(self, fields, paths):
        seconds = time.monotonic() - self.start
        assert seconds <= LIMITS[self.phase]
        write(self.run / (self.phase + '_COMPLETE.json'), dict(arm=self.arm,
            phase=self.phase, identity=self.who, GPU=self.gpu, seconds=seconds,
            source_sha256=self.source_hashes, data_binding=self.data_binding,
            files={str(p.relative_to(self.run)): sha(p) for p in paths}, **fields))
        signal.alarm(0)


def receipt(ctx, model):
    import torch
    core, helper = model.core, ctx.old_deploy
    meta = adaptation.metadata(model)
    parameters = list(core.named_parameters(remove_duplicate=False))
    ab = [(n, p) for n, p in core.named_parameters() if n.endswith(('.A.weight', '.B.weight'))]
    assert len(ab) == (96 if meta['explicit_AB'] else 0)
    assert sum(p.numel() for _, p in ab) == (540672 if meta['explicit_AB'] else 0)
    assert all(p.device == torch.device(ctx.device) and not p.requires_grad and p.grad is None
               and p.dtype == (torch.float32 if n.endswith(('.A.weight', '.B.weight')) else torch.float16)
               for n, p in parameters)
    buffers = list(core.named_buffers(remove_duplicate=False))
    assert all(b.device == torch.device(ctx.device) for _, b in buffers)
    assert all(b.dtype == torch.float32 for n, b in buffers if n.endswith('inv_freq'))
    layout = ctx.storage.layout(core)
    assert layout['weights'] == dict(coefficients=894216224 if meta['explicit_AB'] else 893675552,
        bytes=1789513792 if meta['explicit_AB'] else 1787351104)
    assert layout['ordinary_buffers']['bytes'] == 9032 and layout['trainable'] == 0
    assert {id(p) for p in model.parameters()} == {id(p) for p in core.parameters()}
    assert {id(b) for b in model.buffers()} == {id(b) for b in core.buffers()}
    assert model.model is core.language_model.model and model.lm_head is core.language_model.lm_head
    assert model.visual.tower is core.vision_tower and model.visual.projector is core.multi_modal_projector
    assert model.visual.image_newline is core.image_newline
    assert core.language_model.config._attn_implementation == core.vision_tower.config._attn_implementation == 'sdpa'
    return dict(layout=layout, digest=ctx.storage.digest(core), all_buffers_digest=ctx.storage.digest(buffers),
        parameter_alias_groups=helper.aliases(parameters), buffer_alias_groups=helper.aliases(buffers),
        buffer_manifest={n: dict(shape=list(b.shape), dtype=str(b.dtype), bytes=b.numel()*b.element_size()) for n, b in buffers},
        functional_config_digest=helper.jd(helper.functional(core.config.to_dict())),
        generation_config_digest=helper.jd(core.generation_config.to_dict()),
        language_generation_config_digest=helper.jd(core.language_model.generation_config.to_dict()),
        language_attention_classes=sorted({type(l.self_attn).__name__ for l in model.model.layers}),
        vision_attention_classes=sorted({type(l.self_attn).__name__ for l in core.vision_tower.vision_model.encoder.layers}),
        method=meta)


def export_complete(ctx, model, path):
    import torch
    assert not path.exists()
    proof = receipt(ctx, model)
    state = ctx.storage.state_cpu(model.core)
    buffers = ctx.old_deploy._cpu_buffers(model.core, state)
    assert ctx.storage.digest(state) == proof['digest']
    assert ctx.storage.digest(buffers) == proof['all_buffers_digest']
    for config in (model.core.generation_config, model.core.language_model.generation_config):
        assert config._original_object_hash == hash(config)
    package = dict(format=FORMAT, state=state, buffers=buffers,
        base_config=model.core.config.to_dict(), method=proof['method'], receipt=proof,
        generation_config=model.core.generation_config.to_dict(),
        language_generation_config=model.core.language_model.generation_config.to_dict())
    torch.save(package, path)
    del package, state, buffers
    return dict(proof, path=str(path), bytes=path.stat().st_size, sha256=sha(path),
        self_contained=True, no_teacher_or_optimizer=True)


def load_complete(ctx, path):
    import torch
    from transformers import LlavaOnevisionConfig, LlavaOnevisionForConditionalGeneration
    from transformers.modeling_utils import no_init_weights
    package = torch.load(path, map_location='cpu', weights_only=True)
    assert package['format'] == FORMAT and package['method']['arm'] == ctx.arm
    assert package['method']['deployed']
    config = LlavaOnevisionConfig.from_dict(package['base_config'])
    config._attn_implementation = config.text_config._attn_implementation = config.vision_config._attn_implementation = 'sdpa'
    with torch.device('meta'), no_init_weights():
        core = LlavaOnevisionForConditionalGeneration(config)
        model = ctx.bridge.OneVisionBridge(core)
        adaptation.install_meta(model, ctx.arm, deployed=True, plan=package['method']['rank_plan'])
    bank = {}
    core.load_state_dict(ctx.old_deploy._transfer(package['state'], ctx.device, bank), strict=True, assign=True)
    assert set(package['buffers']) == {n for n, _ in core.named_buffers(remove_duplicate=False)}
    for name, value in ctx.old_deploy._transfer(package['buffers'], ctx.device, bank).items():
        parent, _, leaf = name.rpartition('.')
        core.get_submodule(parent)._buffers[leaf] = value
    for item in core.modules():
        if hasattr(item, 'original_inv_freq'):
            assert getattr(item, 'rope_type', 'default') == 'default'
            item.original_inv_freq = item.inv_freq
    core.language_model.tie_weights()
    core.generation_config = ctx.old_deploy._generation(package['generation_config'])
    core.language_model.generation_config = ctx.old_deploy._generation(package['language_generation_config'])
    ctx.old_deploy._rebind(model)
    model.requires_grad_(False).eval()
    actual = receipt(ctx, model)
    assert actual == package['receipt']
    del package, bank
    gc.collect()
    return model, dict(actual, strict_complete_reload=True, self_contained_meta_construction=True)


def rng_snapshot():
    import numpy as np
    import torch
    device = torch.cuda.current_device()
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            device, torch.cuda.get_rng_state(device))


def rng_restore(state):
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[4], state[3])


def stage_readout(ctx, model, task, phase, records, events):
    """No optimizer mutation; restore exact module modes, grad mask and RNG."""
    import torch
    state = rng_snapshot()
    modes = [(m, m.training) for m in model.modules()]
    parameters = [(p, p.requires_grad) for p in model.parameters()]
    digest_before = ctx.storage.digest(model.core)
    buffers_before = ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    batcher = ctx.batcher(model, phase)
    start = time.monotonic()
    try:
        model.requires_grad_(False).eval()
        for row in ctx.dev[task['task']]:
            batch, binding = ctx.readouts._batch(model, task, row, ctx.storage, batcher.readout)
            tokens = ctx.readouts._generate(model, batch, task['max_new_tokens'], ctx.stop)
            records.append(dict(binding, id=row['id'], task=task['task'], domain=task['domain'],
                phase=phase, stage=task['task'], prediction=ctx.processor.tokenizer.decode(tokens, skip_special_tokens=True),
                references=row['answers'], generation_cap=task['max_new_tokens'], output_tokens=tokens))
            del batch, tokens
        assert ctx.storage.digest(model.core) == digest_before
        assert ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers_before
    finally:
        for p, flag in parameters:
            p.requires_grad_(flag)
        for m, flag in modes:
            m.training = flag
        rng_restore(state)
    assert torch.equal(state[2], torch.get_rng_state())
    assert torch.equal(state[4], torch.cuda.get_rng_state(state[3]))
    events.append(dict(stage=task['task'], phase=phase, new_generations=len(ctx.dev[task['task']]),
        seconds=time.monotonic()-start, state_and_buffers_unchanged=True, training_RNG_restored=True,
        task_has_no_DEV=not bool(ctx.dev[task['task']])))


def train(ctx):
    import torch
    model, base = ctx.old_deploy.load_base(ctx.base, ctx.device, ctx.storage)
    ctx.bind(model)
    ctx.reset()
    method = adaptation.make_method(model, ctx.root, ctx.storage, ctx.arm,
        inherited_source=ctx.plain / 'control_adaptation.py')
    write(ctx.run / 'ADAPTER_INITIAL.json', dict(initialization=method.initialization,
        rank_plan=method.rank_plan, coordinates=540672, parameters=96, mode=adaptation.metadata(model)))
    batcher = ctx.batcher(model, 'TRAIN')
    frozen = lambda: ctx.storage.digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
    frozen_before = frozen()
    buffers_before = ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False)))
    adapter_before = ctx.storage.digest({n: p for n, p in model.named_parameters() if p.requires_grad})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    formal = module('conditional_formal_training', ctx.port / 'run.py')
    directory = ctx.run / 'training'
    directory.mkdir()
    (directory / 'logs').mkdir()
    monitor = formal.ResourceMonitor(directory)
    monitor.snapshot('training_before', model=model)
    data, plan, counts, holder = {}, [], {}, []
    for task in ctx.manifest['tasks']:
        stage = task['task']
        data[stage, 'train'], data[stage, 'dev'] = ctx.fit[stage], ctx.dev[stage]
        order = sorted(ctx.fit[stage], key=lambda r: hashlib.sha256(f"42:{stage}:{r['id']}".encode()).digest())
        task['training_order_sha256'] = hashlib.sha256(''.join(r['id']+'\n' for r in order).encode()).hexdigest()
        plan.append(dict(stage=stage, QA=len(order), updates=(len(order)+15)//16,
            order_sha256=task['training_order_sha256']))
    assert sum(p['QA'] for p in plan) == 1213 and sum(p['updates'] for p in plan) == 87
    raw_clip = torch.nn.utils.clip_grad_norm_
    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        stage = monitor.stage
        counts[stage] = counts.get(stage, 0) + 1
        grads = [p.grad for p in parameters if p.grad is not None]
        assert len(grads) == 96 and sum(g.numel()*g.element_size() for g in grads) == 2162688
        assert all(g.is_cuda and g.dtype == torch.float32 and torch.isfinite(g).all() for g in grads)
        write(ctx.run / f'grad_S{stage:02}_{counts[stage]:03}.json', dict(stage=stage,
            step=counts[stage], parameters=96, gradient_bytes=2162688, finite=True))
        return raw_clip(parameters, *args, **kwargs)
    def optimizer(m):
        assert not holder
        opt = torch.optim.AdamW([p for p in m.model.parameters() if p.requires_grad],
            lr=1e-4, betas=(.9,.999), eps=1e-8, weight_decay=0,
            foreach=False, fused=False, capturable=True)
        holder.append(opt)
        return opt
    before, acquired, readout_events = [], [], []
    def before_stage(stage):
        stage_readout(ctx, model, ctx.manifest['tasks'][stage-1], 'DEV_BEFORE', before, readout_events)
    def after_stage(stage):
        assert frozen() == frozen_before
        assert ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers_before
        opt = holder.pop()
        states = [t for values in opt.state.values() for t in values.values() if isinstance(t, torch.Tensor)]
        assert all(t.is_cuda and torch.isfinite(t).all() for t in states)
        assert sum(t.numel()*t.element_size() for t in states) == 4325760
        snapshot = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
        path = ctx.run / f'adapter_S{stage:02}.pt'
        assert not path.exists()
        torch.save(snapshot, path)
        write(ctx.run / f'STAGE_{stage:02}_COMPLETE.json', dict(stage=stage, updates=counts[stage],
            adapter_digest=ctx.storage.digest(snapshot), snapshot_sha256=sha(path), bytes=2162688,
            optimizer_state_bytes=4325760, all_optimizer_state_GPU=True, frozen_unchanged=True))
        del snapshot, states, opt
        stage_readout(ctx, model, ctx.manifest['tasks'][stage-1], 'DEV_ACQUIRED', acquired, readout_events)
    namespace = dict(vars(formal), model=model, method=method, processor=ctx.processor,
        manifest=ctx.manifest, data=data, run=directory, monitor=monitor,
        args=SimpleNamespace(method='SeqFT_LoRA'), resume_obj=None,
        input_batch=batcher, answer_loss=ctx.rt.answer_loss, prepare=lambda *unused: None,
        visual_cache_stats=ctx.fv.stats, make_optimizer=optimizer,
        before_stage=before_stage, after_stage=after_stage)
    prefix, inherited_ast = ctx.study.training_ast()
    assert inherited_ast == AST_SHA
    node = ast.For(target=ast.Name(id='stage', ctx=ast.Store()),
        iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(1), ast.Constant(20)], keywords=[]),
        body=ast.parse('before_stage(stage)').body + prefix + ast.parse('after_stage(stage)').body, orelse=[])
    ctx.reset()
    torch.nn.utils.clip_grad_norm_ = clip
    try:
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
            '<unchanged87prefix-with-RNG-neutral-readouts>', 'exec'), namespace)
    finally:
        torch.nn.utils.clip_grad_norm_ = raw_clip
    assert sum(counts.values()) == 87 and len(batcher.records) == 1213 and not holder
    assert len(before) == len(acquired) == 100
    assert frozen() == frozen_before
    assert ctx.storage.digest(list(model.core.named_buffers(remove_duplicate=False))) == buffers_before
    adapter_after = ctx.storage.digest({n: p for n, p in model.named_parameters() if p.requires_grad})
    assert adapter_after != adapter_before
    monitor.snapshot('training_after', model=model)
    memory = monitor.finish(model)
    assert not monitor.thread.is_alive()
    model.core.language_model.disable_input_require_grads()
    model.core.language_model.gradient_checkpointing_disable()
    namespace.clear()
    model.requires_grad_(False).eval()
    train_receipt = receipt(ctx, model)
    deployment = adaptation.deploy(model, method)
    package = export_complete(ctx, model, ctx.run / 'complete.pt')
    ctx.processor.save_pretrained(ctx.run / 'processor')
    for name, value in [('before_predictions.json', before), ('acquired_predictions.json', acquired),
                        ('READOUT_EVENTS.json', readout_events), ('FIT_INPUTS.json', batcher.records),
                        ('TRAIN_MEMORY.json', memory)]:
        write(ctx.run / name, value)
    files = [p for p in ctx.run.rglob('*') if p.is_file()
             and p.name not in ('train_STARTED.json', 'train.log') and not p.name.endswith('COMPLETE.json')]
    files.extend(ctx.run / f'STAGE_{stage:02}_COMPLETE.json' for stage in range(1,20))
    ctx.end(dict(new_generations=200, new_scores=0, FIT_backwards=1213, optimizer_updates=87,
        coordinates=540672, parameters=96, training_plan=plan, training_ast_sha256=AST_SHA,
        original_base=base, adapter_before=adapter_before, adapter_after=adapter_after,
        frozen_unchanged=True, buffers_unchanged=True, training_end=train_receipt,
        deployment=deployment, checkpoint=package,
        before_QA=100, acquired_QA=100, final_QA=0, TEST_QA=0), files)


def evaluate(ctx):
    previous = read(ctx.run / 'train_COMPLETE.json')
    assert previous['arm'] == ctx.arm and previous['source_sha256'] == ctx.source_hashes
    assert previous['data_binding'] == ctx.data_binding
    for relative, expected in previous['files'].items():
        assert sha(ctx.run / relative) == expected, relative
    model, loaded = load_complete(ctx, ctx.run / 'complete.pt')
    for key, value in receipt(ctx, model).items():
        assert previous['checkpoint'][key] == value, key
    ctx.bind(model)
    batcher = ctx.batcher(model, 'RESOURCE')
    memory, memory_proof = ctx.readouts.resource_windows(model, ctx.processor, ctx.rows,
        ctx.storage, batch_provider=batcher.readout, generation_policy=ctx.stop)
    batcher.phase = 'DEV_FINAL'
    predictions, prediction_proof = ctx.readouts.evaluate_dev(model, ctx.processor, ctx.rows,
        ctx.storage, batch_provider=batcher.readout, generation_policy=ctx.stop)
    for row in predictions:
        row['phase'] = 'DEV_FINAL'
    actual = receipt(ctx, model)
    assert all(loaded[key] == value for key, value in actual.items())
    write(ctx.run / 'memory.json', memory)
    write(ctx.run / 'final_predictions.json', predictions)
    ctx.end(dict(new_generations=105, new_scores=0, FIT_backwards=0, optimizer_updates=0,
        TEST_QA=0, reloaded=loaded, invariant=actual,
        complete_state_and_buffers_unchanged=True,
        resource_proof=memory_proof, prediction_proof=prediction_proof),
        [ctx.run / 'memory.json', ctx.run / 'final_predictions.json'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--arm', choices=adaptation.ARMS, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--phase', choices=('train', 'eval'), required=True)
    args = parser.parse_args()
    try:
        ctx = Context(args)
        train(ctx) if args.phase == 'train' else evaluate(ctx)
    except BaseException:
        if args.run.is_dir():
            failure = args.run / (args.phase + '_FAILURE.json')
            if not failure.exists():
                write(failure, dict(arm=args.arm, phase=args.phase, unix=time.time(),
                    traceback=traceback.format_exc(), no_quality_claim=True))
        raise


if __name__ == '__main__':
    main()
