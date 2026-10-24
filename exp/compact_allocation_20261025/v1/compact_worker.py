"""WIP: fresh compact/all learning and separate folded deployment readouts.

No quality scoring occurs here. Training generates no DEV answers. The four
domain-end cases restore cumulative snapshots on independent fresh models.
"""
from pathlib import Path
import ast
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
DOMAIN_CASES = {'natural1': (1, 'natural'), 'medical11': (11, 'medical'),
                'geography15': (15, 'geography'), 'finance16': (16, 'finance')}
LIMITS = {'train': 3000, 'compact_eval': 450, **{c: 180 for c in DOMAIN_CASES}}
BOUNDS = {'natural': 1, 'medical': 11, 'geography': 15, 'finance': 16, 'traffic': 19}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(name, value):
    with (RUN / name).open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main(case):
    ready = read(RUN / 'READY.json')
    for name, digest in ready['sources'].items():
        assert sha(RUN / name) == digest, name
    for name, digest in ready['runtime_files'].items():
        assert sha(ASSET / name) == digest, name
    for name, digest in ready['prior_files'].items():
        assert sha(RUN / name) == digest, name
    spec = importlib.util.spec_from_file_location('identity_only', ASSET / 'exp/paired_image_support_20261024/v1/common.py')
    ids = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ids)
    gpu = int(os.environ['LLMCL_GPU'])
    card = ids.inventory()[gpu]
    assert not card['owners'] and card['memory_MiB'] < 500 and card['utilization'] < 10
    who = ids.identity(os.getpid())
    assert who['pid'] == who['pgid']
    tick = time.monotonic()
    write(case + '_STARTED.json', dict(unix=time.time(), identity=who, GPU=gpu,
        boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        ready_sha256=sha(RUN / 'READY.json'), activation_sha256=sha(RUN / 'ACTIVATION.json')))

    def expired(*args):
        raise TimeoutError('Frozen compact allocation phase budget expired')

    signal.signal(signal.SIGALRM, expired)
    signal.alarm(LIMITS[case])
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8',
                      MKL_NUM_THREADS='8', TOKENIZERS_PARALLELISM='false')
    import random
    import numpy as np
    import torch
    from transformers import AutoProcessor
    sys.path.insert(0, str(ASSET / 'exp/two_backbone_native_cl_20261017_v2'))
    sys.path.insert(0, str(ASSET / 'exp/osft_increment_screen_20261019/v1'))
    import onevision_bridge as bridge
    import frozen_vision as fv
    import runtime_utils as rt
    import study
    import all_adaptation as adaptation
    import vision_interface as vi
    spec = importlib.util.spec_from_file_location('storage_only', ROOT / 'exp/ffn_static_delete_20261025/v1/core.py')
    storage = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(storage)
    assert sha(ROOT / 'exp/ffn_static_delete_20261025/v1/core.py') == ready['storage_helper_sha256']
    torch.set_num_threads(8)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    base = ASSET / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
    processor = AutoProcessor.from_pretrained(base, local_files_only=True)
    path, manifest, original = study.original_train()
    split, fit, dev = study.split_train(manifest, original)
    assert study.sha(path) == ready['manifest_sha256']
    assert split == read(study.REPORT / 'SPLIT.json')
    assert study.sha(study.REPORT / 'SPLIT.json') == ready['split_sha256']
    bounds = {d: max(t['task'] for t in manifest['tasks'] if t['domain'] == d) for d in BOUNDS}
    assert bounds == BOUNDS
    rows = [(t, row) for t in manifest['tasks'] for row in dev[t['task']]]
    assert len(rows) == 100
    assert all(sum(t['domain'] == d for t, _ in rows) == 20 for d in BOUNDS)
    cf_init = read(RUN / 'prior_cf_INITIALIZATION.json')
    cf_final = {row['id']: row for row in read(RUN / 'prior_cf_final_predictions.json')}
    assert len(cf_final) == 100 and set(cf_final) == {row['id'] for _, row in rows}
    cf_windows = {row['id']: row for row in read(RUN / 'prior_cf_VERIFIED.json')['memory']['compact']}
    teacher = read(RUN / 'prior_parent_COMPLETE.json')

    def buffer_digest(model):
        return storage.digest({n: v for n, v in model.core.named_buffers()
            if not n.startswith(('vision_tower.', 'multi_modal_projector.'))})

    def alias_check(model):
        assert model.visual.tower is model.core.vision_tower
        assert model.visual.projector is model.core.multi_modal_projector
        assert storage._measure(model.named_parameters(remove_duplicate=False)) == storage.layout(model)['weights']

    def prepare(model):
        bridge.bind(model)
        fv.LIMIT = 0
        fv.install(model)
        alias_check(model)
        assert sorted({type(layer.self_attn).__name__ for layer in model.model.layers}) == teacher['language_attention_classes']
        assert buffer_digest(model) == cf_init['buffer_digest'] == teacher['preserved_nonvision_buffers_digest']

    def no_cache(model):
        model._llmcl_visual_cache.clear()
        model._llmcl_visual_cache_bytes = 0
        fv.LIMIT = 0

    def fresh_all():
        model = bridge.Factory.from_pretrained(base, torch_dtype=torch.float16,
            attn_implementation='sdpa', local_files_only=True).cuda()
        bridge.bind(model)
        base_language_digest = storage.digest(model.core.language_model)
        assert base_language_digest == cf_init['base_language_digest']
        asset_path = RUN / 'fixed_vision_only.pt'
        assert sha(asset_path) == ready['vision_asset']['sha256']
        assert asset_path.stat().st_size == ready['vision_asset']['bytes']
        asset = torch.load(asset_path, map_location='cpu', weights_only=False)
        interface = vi.VisionInterface(vi.student(asset['student_config']), **asset['interface_config']).half().cuda()
        interface.load_state_dict(asset['state'], strict=True)
        student_config = asset['student_config']
        assert storage.digest(interface) == asset['digest'] == cf_init['vision_digest']
        del asset
        # The unchanged TargetLoRA constructor retains its original q/v RNG
        # stream and per-extra-module fork_rng seeds.
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        method = adaptation.make_all(model, ASSET, storage)
        initial_adapter = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
        initial_adapter_digest = storage.digest(initial_adapter)
        del initial_adapter
        receipt = adaptation.install_interface_preserving_adapters(model, interface, method)
        assert receipt == dict(adapter_parameters=336, adapter_coordinates=4399104,
                               identities_bytes_and_RNG_preserved=True)
        assert len(method.layers) == 168
        assert storage.digest({n: p for n, p in model.named_parameters() if p.requires_grad}) == initial_adapter_digest
        assert not any(isinstance(mod, torch.nn.Dropout) and mod.p > 0 for mod in interface.modules())
        assert interface.student.config.attention_dropout == 0
        assert all(torch.count_nonzero(model.get_submodule(n).B.weight).item() == 0 for n in method.rank_plan)
        frozen_digest = storage.digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
        assert frozen_digest == cf_init['frozen_digest']
        assert buffer_digest(model) == cf_init['buffer_digest']
        alias_check(model)
        return model, method, student_config, base_language_digest, initial_adapter_digest, receipt, frozen_digest

    class FiniteScores:
        def __call__(self, input_ids, scores):
            assert not torch.isnan(scores).any() and not torch.isposinf(scores).any()
            assert torch.isfinite(scores).any(dim=-1).all()
            return scores

    def generate(model, batch, task, fixed=False):
        kwargs = dict(do_sample=False, num_beams=1, max_new_tokens=32 if fixed else task['max_new_tokens'],
            use_cache=True, pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id, logits_processor=[FiniteScores()])
        if fixed:
            kwargs['min_new_tokens'] = 32
        with torch.inference_mode(), torch.autocast('cuda', dtype=model.dtype):
            out = model.generate(**batch, **kwargs)
        return out[0, batch['input_ids'].shape[1]:].tolist()

    def checked_batch(model, task, row):
        for image in row['images']:
            assert sha(ASSET / image) == split['image_sha256'][image]
        batch, _ = rt.input_batch(row, task, processor, model.device)
        reference = cf_final[row['id']]
        assert storage.digest(batch) == reference['prepared_sha256']
        assert storage.digest({k: v for k, v in batch.items() if k != 'pixel_values'}) == reference['transport_sha256']
        assert len(batch['pixel_values']) == reference['packed_tokens']
        assert row['answers'] == reference['references'] and task['max_new_tokens'] == reference['generation_cap']
        return batch, reference

    def evaluate(model, label, selected, stage=None):
        model.eval()
        no_cache(model)
        before = storage.digest(model)
        values = []
        for task, row in selected:
            batch, reference = checked_batch(model, task, row)
            tokens = generate(model, batch, task)
            value = dict(id=row['id'], task=task['task'], domain=task['domain'],
                prepared_sha256=storage.digest(batch), transport_sha256=reference['transport_sha256'],
                packed_tokens=len(batch['pixel_values']), prediction=processor.tokenizer.decode(tokens, skip_special_tokens=True),
                output_tokens=tokens, references=row['answers'], generation_cap=task['max_new_tokens'])
            if stage is not None:
                value['stage'] = stage
            values.append(value)
            del batch
        assert storage.digest(model) == before and model._llmcl_visual_cache_bytes == 0
        write(label + '_predictions.json', values)

    def resource_windows(model):
        model.eval()
        no_cache(model)
        measurements = []
        for domain in BOUNDS:
            task, row = min(((t, r) for t, r in rows if t['domain'] == domain),
                            key=lambda tr: hashlib.sha256(tr[1]['id'].encode()).digest())
            raw = {}

            def capture(module, args):
                assert not raw
                raw.update(raw_pixels_sha256=storage.digest({'pixels': args[0]}), tiles=args[0].shape[0])

            handle = model.core.vision_tower.register_forward_pre_hook(capture)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
            try:
                batch, _ = checked_batch(model, task, row)
            finally:
                handle.remove()
            reference = cf_windows[row['id']]
            assert raw['raw_pixels_sha256'] == reference['raw_pixels_sha256']
            assert raw['tiles'] == reference['tiles']
            tokens = generate(model, batch, task, True)
            torch.cuda.synchronize()
            assert len(tokens) == 32
            measurements.append(dict(id=row['id'], domain=domain,
                prepared_sha256=storage.digest(batch),
                transport_sha256=storage.digest({k: v for k, v in batch.items() if k != 'pixel_values'}),
                text_sha256=storage.digest({'input_ids': batch['input_ids']}),
                input_shape=list(batch['input_ids'].shape), packed_tokens=len(batch['pixel_values']), new_tokens=32,
                baseline_allocated=start, peak_allocated=torch.cuda.max_memory_allocated(),
                peak_reserved=torch.cuda.max_memory_reserved(), **raw))
            del batch, tokens
        write('compact_eval_memory.json', measurements)

    if case == 'train':
        model, method, student_config, base_language_digest, initial_adapter_digest, receipt, frozen_before = fresh_all()
        interface = model.core.vision_tower
        frozen = lambda: storage.digest({n: p for n, p in model.named_parameters() if not p.requires_grad})
        initial_layout = storage.layout(model)
        assert initial_layout['trainable'] == 4399104 and initial_layout['weights']['coefficients'] == 585612160
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        prepare(model)
        method.vision_original = model._llmcl_visual_original
        write('INITIALIZATION.json', dict(base_language_digest=base_language_digest,
            adapter_initial_digest=initial_adapter_digest, rank_plan=method.rank_plan,
            initialization=method.initialization, receipt=receipt, layout=initial_layout,
            frozen_digest=frozen_before, vision_digest=storage.digest(interface),
            buffer_digest=buffer_digest(model), no_S19_language_loaded=True,
            zero_B_modules=168, initial_new_generations=0,
            cf_initial_frozen_digest=cf_init['frozen_digest'],
            zero_B_same_base_vision_buffers_not_same_adapter_storage=True))
        fv.LIMIT = 128 * 1024**2
        spec = importlib.util.spec_from_file_location('inherited_train', ASSET / 'exp/two_backbone_native_cl_20261017_v2/run.py')
        formal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(formal)
        training = RUN / 'training'
        training.mkdir()
        (training / 'logs').mkdir()
        monitor = formal.ResourceMonitor(training)
        monitor.snapshot('training_before', model=model)
        mapped, plan = {}, []
        for task in manifest['tasks']:
            stage = task['task']
            mapped[stage, 'train'] = fit[stage]
            mapped[stage, 'dev'] = dev[stage]
            order = sorted(fit[stage], key=lambda row: hashlib.sha256(f"42:{stage}:{row['id']}".encode()).digest())
            task['training_order_sha256'] = hashlib.sha256(''.join(r['id'] + '\n' for r in order).encode()).hexdigest()
            plan.append(dict(stage=stage, QA=len(order), updates=(len(order) + 15) // 16,
                             order_sha256=task['training_order_sha256']))
        assert sum(p['QA'] for p in plan) == 1213 and sum(p['updates'] for p in plan) == 87
        assert plan == ready['training_plan']
        namespace = dict(vars(formal), model=model, method=method, processor=processor,
            manifest=manifest, data=mapped, run=training, monitor=monitor,
            args=SimpleNamespace(method=method.name), resume_obj=None)
        counts = {}
        raw_clip = torch.nn.utils.clip_grad_norm_

        def record_grad(parameters, *args, **kwargs):
            parameters = list(parameters)
            stage = monitor.stage
            counts[stage] = counts.get(stage, 0) + 1
            grads = {n: p.grad.detach().cpu() for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
            assert len(grads) == 336 and all(torch.isfinite(g).all() for g in grads.values())
            assert sum(g.numel() * g.element_size() for g in grads.values()) == 17596416
            write(f'grad_S{stage:02}_{counts[stage]:03}.json', dict(parameters=336, gradient_bytes=17596416, finite=True))
            if counts[stage] == 1:
                torch.save(grads, RUN / f'grad_S{stage:02}_first.pt')
            return raw_clip(parameters, *args, **kwargs)

        def save_stage(stage):
            assert frozen() == frozen_before
            adapter = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
            assert len(adapter) == 336
            torch.save(adapter, RUN / f'adapter_S{stage:02}.pt')
            write(f'STAGE_{stage:02}_COMPLETE.json', dict(stage=stage, updates=counts[stage],
                adapter_digest=storage.digest(adapter), adapter_bytes=sum(t.numel() * t.element_size() for t in adapter.values()),
                frozen_digest=frozen_before, vision_digest=storage.digest(interface)))

        namespace['save_stage'] = save_stage
        prefix, training_ast = study.training_ast()
        assert training_ast == ready['training_ast_sha256']
        node = ast.For(target=ast.Name(id='stage', ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(1), ast.Constant(20)], keywords=[]),
            body=prefix + ast.parse('save_stage(stage)').body, orelse=[])
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.nn.utils.clip_grad_norm_ = record_grad
        try:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                         '<inherited-87-updates>', 'exec'), namespace)
        finally:
            torch.nn.utils.clip_grad_norm_ = raw_clip
        assert sum(counts.values()) == 87 and len(counts) == 19 and frozen() == frozen_before
        monitor.snapshot('training_after', model=model)
        memory = monitor.finish(model)
        model.core.language_model.disable_input_require_grads()
        model.core.gradient_checkpointing_disable()
        folded = adaptation.fold(model, method)
        assert len(folded) == 168
        package = adaptation.export_package(model, student_config)
        package.update(digest=storage.digest(model), layout=storage.layout(model))
        assert package['layout']['weights'] == dict(coefficients=581213056, bytes=1162426112)
        assert package['layout']['trainable'] == 0
        cp = RUN / 'compact_complete.pt'
        torch.save(package, cp)
        proof = dict(initial_adapter_digest=initial_adapter_digest, base_language_digest=base_language_digest,
            frozen_base_unchanged=True, vision_digest=storage.digest(interface), buffer_digest=buffer_digest(model),
            counts=counts, training_plan=plan, training_ast_sha256=training_ast, FIT_QA=1213,
            FIT_model_requests=1213, updates=87, optimizer_updates=87, backwards=1213,
            new_generations=0, new_DEV_generations=0, DEV_used_for_optimization=0,
            checkpoint=dict(bytes=cp.stat().st_size, sha256=sha(cp)), digest=package['digest'],
            layout=package['layout'], fold=folded, training_memory=memory)
        files = [p for p in RUN.iterdir() if p.is_file() and p.name.startswith(('INITIALIZATION', 'grad_', 'adapter_', 'STAGE_'))]
        files += [p for p in training.rglob('*') if p.is_file()]
    elif case == 'compact_eval':
        end = read(RUN / 'train_COMPLETE.json')
        cp = RUN / 'compact_complete.pt'
        assert cp.stat().st_size == end['checkpoint']['bytes'] and sha(cp) == end['checkpoint']['sha256']
        package = torch.load(cp, map_location='cpu', weights_only=False)
        model = adaptation.load_export(package, 'cuda')
        assert storage.digest(model) == package['digest'] == end['digest']
        assert storage.layout(model) == package['layout'] == end['layout']
        del package
        prepare(model)
        before = storage.digest(model)
        resource_windows(model)
        evaluate(model, 'final', rows)
        assert storage.digest(model) == before
        assert storage.digest(model.core.vision_tower) == cf_init['vision_digest']
        proof = dict(digest=before, digest_before=before, digest_after=storage.digest(model),
            layout=storage.layout(model), new_generations=105, new_DEV_generations=100,
            strict_independent_static_reload=True, buffer_digest=buffer_digest(model),
            vision_digest=storage.digest(model.core.vision_tower), FIT_model_requests=0,
            updates=0, optimizer_updates=0, backwards=0, DEV_used_for_optimization=0)
        files = [RUN / 'final_predictions.json', RUN / 'compact_eval_memory.json']
    else:
        stage, domain = DOMAIN_CASES[case]
        end = read(RUN / 'train_COMPLETE.json')
        completed_eval = read(RUN / 'compact_eval_COMPLETE.json')
        assert completed_eval['digest'] == end['digest'] and completed_eval['strict_independent_static_reload'] is True
        model, method, _, _, _, _, _ = fresh_all()
        init = read(RUN / 'INITIALIZATION.json')
        record_path = RUN / f'STAGE_{stage:02}_COMPLETE.json'
        snapshot_path = RUN / f'adapter_S{stage:02}.pt'
        assert sha(record_path) == end['files'][record_path.name]
        assert sha(snapshot_path) == end['files'][snapshot_path.name]
        record = read(record_path)
        assert record['stage'] == stage == bounds[domain]
        snapshot = torch.load(snapshot_path, map_location='cpu', weights_only=True)
        reconstruction = adaptation.restore_snapshot(model, method, snapshot, record, init, storage)
        del snapshot
        folded = adaptation.fold(model, method)
        assert len(folded) == 168 and storage.layout(model) == end['layout']
        prepare(model)
        before = storage.digest(model)
        selected = [(task, row) for task, row in rows if task['domain'] == domain]
        assert len(selected) == 20
        evaluate(model, case, selected, stage)
        after = storage.digest(model)
        assert after == before and storage.digest(model.core.vision_tower) == init['vision_digest']
        assert buffer_digest(model) == init['buffer_digest']
        proof = dict(stage=stage, domain=domain, restore_receipt=reconstruction, fold=folded,
            snapshot_source_sha256=sha(snapshot_path), digest=before,
            digest_before=before, digest_after=after, layout=storage.layout(model),
            vision_digest=storage.digest(model.core.vision_tower), buffer_digest=buffer_digest(model),
            new_generations=20, new_DEV_generations=20, FIT_model_requests=0,
            updates=0, optimizer_updates=0, backwards=0, DEV_used_for_optimization=0)
        files = [RUN / (case + '_predictions.json')]
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    import resource
    proof.update(identity=who, ready_sha256=sha(RUN / 'READY.json'),
        files={p.relative_to(RUN).as_posix(): sha(p) for p in files},
        teacher_requests=0, NLL_requests=0, TEST_QA=0,
        process_RSS_highwater_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        last_reset_GPU_peak_allocated=torch.cuda.max_memory_allocated())
    proof['seconds'] = time.monotonic() - tick
    write(case + '_COMPLETE.json', proof)


if __name__ == '__main__':
    case = sys.argv[1]
    assert case in LIMITS
    try:
        main(case)
    except Exception:
        write(case + '_FAILURE.json', dict(unix=time.time(), traceback=traceback.format_exc()))
        raise
