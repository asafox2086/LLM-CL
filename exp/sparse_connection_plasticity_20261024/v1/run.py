"""One isolated init/Frozen/Plastic worker; imports Torch only after STARTED."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import random
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from common import (ASSETS, CASES, FFN_SLOTS, FINAL_VALUES, PATTERN_BYTES, PORT, QV_COORDINATES,
    REPORT, ROOT, WORK, all_weight_layout, check_ready, data, ffn_fp16_observation,
    finish_resources, inventory, mask_digest,
    process_identity, sha, tensor_digest, training_ast, write)


class ResourceBusy(RuntimeError):
    pass


def main(phase, physical):
    ready = check_ready()
    admission = json.loads((REPORT / 'ADMISSION_REVIEW.json').read_text())
    assert admission['decision'] == 'accept_single_fixed_sparse_pilot_activation'
    assert admission['ready_sha256'] == sha(REPORT / 'READY.json')
    selection = json.loads((REPORT / 'DEVICE_SELECTION.json').read_text())
    assert selection['host'] == 'OLD' and selection['physical_GPU'] == physical
    assert physical in ready['allowed_physical_GPUs']
    cards = inventory(); card = cards[physical]
    if card['owners'] or card['memory_MiB'] >= 500 or card['utilization'] >= 10:
        raise ResourceBusy('Resource changed before STARTED/CUDA; unattempted wait')
    assert shutil.disk_usage(ROOT).free > 32 * 1024**3
    public = REPORT / 'phases' / phase; run = WORK / phase
    assert not run.exists() and not any((public / n).exists() for n in ('STARTED.json', 'FAILURE.json', 'COMPLETE.json'))
    if phase != 'init':
        init = json.loads((REPORT / 'phases/init/COMPLETE.json').read_text())
        assert init['resource_gate_passed']
        if phase == 'Plastic':
            assert json.loads((REPORT / 'phases/Frozen/COMPLETE.json').read_text())['complete']
    begin = time.monotonic(); cap = 3600 if phase == 'init' else 7200
    write(public / 'STARTED.json', dict(phase=phase, **process_identity(), monotonic=begin,
        budget_seconds=cap, ready_sha256=sha(REPORT / 'READY.json'), physical_GPU=physical, UTC=time.time()))
    def expired(*unused):
        raise TimeoutError('Frozen phase budget expired; stop entire fixed pilot')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, cap)
    path, manifest, split, fit, dev, mapped, plan, groups, calibration, diagnostic = data()
    assert plan == ready['training_plan'] and sha(path) == ready['manifest_sha256']
    assert [hashlib.sha256(r['id'].encode()).hexdigest() for r in calibration] == ready['calibration_ID_SHA256']
    assert [hashlib.sha256(r['id'].encode()).hexdigest() for r in diagnostic] == ready['diagnostic_ID_SHA256']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical)
    os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '8'
    import torch
    import numpy as np
    from transformers import AutoProcessor
    from core import Packed2of4Linear, finalize_ffn_values
    from runtime import SparseMethod, add_common_qv, measured_optimizer
    from initialize import initialize
    sys.path.insert(0, str(PORT))
    spec = importlib.util.spec_from_file_location('sparse_pilot_formal_run', PORT / 'run.py')
    formal = importlib.util.module_from_spec(spec); spec.loader.exec_module(formal)
    sys.modules['run'] = formal
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    def seed():
        random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    seed()
    run.mkdir(parents=True)
    for folder in ('logs', 'gradients', 'optimizer', 'layouts'):
        (run / folder).mkdir()
    monitor = formal.ResourceMonitor(run)
    monitor.snapshot('process_before_model')
    seen_images = {}
    training_inputs = (run / 'training_input_IDs.jsonl').open('x', encoding='utf8')
    def checked_input(row, task, processor, device):
        image_hashes = {}
        for name in row['images']:
            if name not in seen_images:
                seen_images[name] = sha(ROOT / name)
            assert seen_images[name] == split['image_sha256'][name]
            image_hashes[name] = seen_images[name]
        batch, details = formal.input_batch(row, task, processor, device)
        details['image_sha256'] = image_hashes
        details['prompt_ID_sha256'] = tensor_digest([('input_ids', batch['input_ids'])])
        if phase != 'init' and model.training and not formal.onevision_bridge.NATIVE_GATE_MODE:
            training_inputs.write(json.dumps(dict(stage=method.stage,
                id_sha256=hashlib.sha256(row['id'].encode()).hexdigest(),
                prompt_ID_sha256=details['prompt_ID_sha256'], image_sha256=image_hashes)) + '\n')
            training_inputs.flush()
        return batch, details
    # Native compatibility imports this function at call time. Keep the original
    # formal.input_batch object above, avoiding recursion while auditing every view.
    import runtime_utils as original_runtime
    original_runtime.input_batch = checked_input
    config = dict(method='SparseConnection' + phase, phase=phase, FIT_QA=1213, DEV_QA=100, TEST_QA=0,
        manifest_sha256=ready['manifest_sha256'], split_sha256=ready['split_sha256'],
        updates=87, qv_learning_rate=1e-4, FFN_learning_rate=1e-5 if phase == 'Plastic' else None,
        final_weight_values=FINAL_VALUES, pattern_bytes=PATTERN_BYTES, seed=42,
        physical_GPU=physical, host='OLD', ready_sha256=sha(REPORT / 'READY.json'),
        dtype='FP16 base/outer CUDA autocast; FP32 FFN master and common qv; final FFN FP16',
        checkpoint_payload_phase_note='Inherited label is historical; actual saves follow each TRAIN stage, before final DEV',
        temporary_dense_reconstruction_charged=True, no_quantization=True)
    write(run / 'config.json', config)
    if phase == 'init':
        check_ready(base=True)
        base = ROOT / 'model/LLaVA-OneVision-Qwen2-0.5B-OV'
        processor = AutoProcessor.from_pretrained(base, local_files_only=True)
        model = formal.onevision_bridge.Factory.from_pretrained(base, torch_dtype=torch.float16,
            attn_implementation='sdpa', local_files_only=True).cuda()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        formal.onevision_bridge.bind(model)
        vision = formal.install(model)
        cache_before = formal.visual_cache_stats(model)
        # The initialization helper performs current-S1 compatibility checks.
        result = initialize(model, processor, formal, calibration, diagnostic,
            manifest['tasks'][0], checked_input, monitor, run / 'initialization')
        if result['resource_gate_passed']:
            add_common_qv(model)
            method = SparseMethod(model, 'Frozen'); method.vision_original = vision
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            assert all_weight_layout(model)['weight_coefficients'] == FINAL_VALUES
            assert all_weight_layout(model)['pattern_bytes'] == PATTERN_BYTES
            meta = formal.save_checkpoint(method, processor, run, 0, config)
            write(run / 'CP0_BINDING.json', dict(checkpoint=meta, model_parameter_digest=tensor_digest(list(model.named_parameters())),
                mask_digest=mask_digest(model), layout=all_weight_layout(model), untrained=True))
        memory = finish_resources(monitor, model, phase)
        training_inputs.close()
        init_files = [run / 'resources.jsonl', run / 'memory_summary.json', run / 'config.json',
            run / 'training_input_IDs.jsonl', *sorted((run / 'initialization').glob('*.json'))]
        if result['resource_gate_passed']:
            init_files += [run / 'CP0_BINDING.json', *sorted((run / 'checkpoints').glob('*.json'))]
        init_file_hashes = {p.relative_to(run).as_posix(): sha(p) for p in init_files}
        assert time.monotonic() - begin <= cap
        write(public / 'COMPLETE.json', dict(complete=True, phase=phase, worker=process_identity(),
            model_RUN_seconds=time.monotonic() - begin, resource_gate_passed=result['resource_gate_passed'],
            initial_function_sha256=sha(run / 'initialization/INITIAL_FUNCTION.json'),
            initial_layout=all_weight_layout(model), memory=memory,
            cache_path_before=cache_before, cache_path_after=formal.visual_cache_stats(model),
            native_core_bypass_NLL_paths=2, native_core_bypass_short_generations=2,
            cache_counts_are_not_total_visual_tower_calls=True,
            file_sha256=init_file_hashes,
            CP0_binding_sha256=sha(run / 'CP0_BINDING.json') if result['resource_gate_passed'] else None,
            ready_sha256=sha(REPORT / 'READY.json'), TEST_QA=0))
    else:
        obj = formal.load_checkpoint(WORK / 'init')
        assert obj['stage'] == 0
        model, processor = obj['method'].model, obj['processor']; del obj
        binding = json.loads((WORK / 'init/CP0_BINDING.json').read_text())
        assert tensor_digest(list(model.named_parameters())) == binding['model_parameter_digest']
        assert mask_digest(model) == binding['mask_digest']
        formal.onevision_bridge.bind(model)
        method = SparseMethod(model, phase); method.vision_original = model._llmcl_visual_original
        cache_before = formal.visual_cache_stats(model)
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        model.enable_input_require_grads()
        expected_trainable = QV_COORDINATES + (FFN_SLOTS if phase == 'Plastic' else 0)
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_trainable
        stable = [(n, p) for n, p in model.named_parameters() if not p.requires_grad and not n.endswith('.values')]
        stable_before = tensor_digest(stable); masks_before = mask_digest(model)
        frozen_values_before = tensor_digest([(n, p) for n, p in model.named_parameters() if n.endswith('.values')])
        initial_ffn_fp16 = ffn_fp16_observation(model)
        write(run / 'INITIAL_BINDING.json', dict(CP0_binding_sha256=sha(WORK / 'init/CP0_BINDING.json'),
            initial_parameter_digest=binding['model_parameter_digest'], mask_digest=masks_before,
            training_layout=all_weight_layout(model), trainable_coordinates=expected_trainable))
        seed(); counts = {}; prior = None; native_gates = []
        def pre_stage(stage):
            formal.onevision_bridge.NATIVE_GATE_MODE = True
            with torch.no_grad():
                native_gates.append(dict(stage=stage, **formal.onevision_bridge.forward_gate(fit[stage][0], manifest['tasks'][stage - 1], processor)))
            formal.onevision_bridge.NATIVE_GATE_MODE = False
            formal.onevision_bridge.LAST_NATIVE = None
        def commit_optimizer(record):
            write(run / 'optimizer' / f'S{monitor.stage:02d}_{counts[monitor.stage]:03d}.json', record)
        def finish_stage(stage):
            nonlocal prior
            assert tensor_digest(stable) == stable_before and mask_digest(model) == masks_before
            if phase == 'Frozen':
                assert tensor_digest([(n, p) for n, p in model.named_parameters() if n.endswith('.values')]) == frozen_values_before
            if stage == 19:
                master_after = tensor_digest([(n, p) for n, p in model.named_parameters() if n.endswith('.values')])
                before_cast = ffn_fp16_observation(model)
                finalize_ffn_values(model)
                actual_final = ffn_fp16_observation(model)
                assert before_cast == actual_final
                write(run / 'FFN_ADAPTATION.json', dict(initial_FP32_master_sha256=frozen_values_before,
                    before_final_cast_FP32_master_sha256=master_after,
                    initial_FP16=initial_ffn_fp16, final_FP16=actual_final,
                    FP16_changed_layers=sum(initial_ffn_fp16['layer_sha256'][name] != digest
                        for name, digest in actual_final['layer_sha256'].items()),
                    zero_change_is_reportable=True, change_not_knowledge_or_quality_proof=True))
                for parameter in model.parameters():
                    parameter.requires_grad_(False); parameter.grad = None
                model._llmcl_visual_cache.clear(); model._llmcl_visual_cache_bytes = 0
            write(run / 'layouts' / f'S{stage:02d}.json', all_weight_layout(model))
            meta = formal.save_checkpoint(method, processor, run, stage, config)
            if prior is not None:
                old = (run / prior['path']).resolve()
                assert old.is_relative_to((run / 'checkpoints').resolve())
                old.unlink()
            prior = meta
        namespace = dict(vars(formal), model=model, method=method, processor=processor,
            manifest=manifest, data=mapped, run=run, monitor=monitor, args=SimpleNamespace(method=method.name),
            resume_obj=None, input_batch=checked_input, pre_stage=pre_stage, finish_stage=finish_stage,
            make_optimizer=lambda m: measured_optimizer(m, commit_optimizer))
        prefix, digest = training_ast(); assert digest == ready['training_ast_sha256']
        node = ast.For(target=ast.Name(id='stage', ctx=ast.Store()),
            iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()), args=[ast.Constant(1), ast.Constant(20)], keywords=[]),
            body=ast.parse('pre_stage(stage)').body + prefix + ast.parse('finish_stage(stage)').body, orelse=[])
        original_clip = torch.nn.utils.clip_grad_norm_
        def clip(parameters, *args, **kwargs):
            parameters = list(parameters); stage = monitor.stage
            counts[stage] = counts.get(stage, 0) + 1
            actual = {n: p for n, p in model.named_parameters() if p.requires_grad}
            assert sum(p.numel() for p in actual.values()) == expected_trainable
            grads = {n: dict(norm=float(p.grad.detach().float().norm()), finite=bool(torch.isfinite(p.grad).all()))
                for n, p in actual.items() if p.grad is not None}
            assert grads.keys() == actual.keys() and all(v['finite'] for v in grads.values())
            write(run / 'gradients' / f'S{stage:02d}_{counts[stage]:03d}.json', dict(raw=grads))
            if counts[stage] == 1:
                torch.save({n: p.grad.detach().cpu() for n, p in actual.items()}, run / 'gradients' / f'S{stage:02d}_first_raw.pt')
            return original_clip(parameters, *args, **kwargs)
        torch.nn.utils.clip_grad_norm_ = clip
        try:
            exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), '<original-TRAIN-prefix>', 'exec'), namespace)
        finally:
            torch.nn.utils.clip_grad_norm_ = original_clip
        assert len(counts) == 19 and sum(counts.values()) == 87
        assert tensor_digest(stable) == stable_before and mask_digest(model) == masks_before
        final_layout = all_weight_layout(model)
        assert final_layout['weight_coefficients'] == FINAL_VALUES and final_layout['packed_modules'] == 72
        assert final_layout['pattern_bytes'] == PATTERN_BYTES and final_layout['trainable_coordinates'] == 0
        assert all(layer.values.dtype == torch.float16 for layer in model.modules() if isinstance(layer, Packed2of4Linear))
        write(run / 'NATIVE_GATE.json', dict(gates=native_gates, NLL_paths=38, short_generations=38, current_stage_only=True))
        before_reload = tensor_digest(list(model.named_parameters()) + list(model.named_buffers()))
        obj = formal.load_checkpoint(run); recovered = obj['method']; del obj
        assert all_weight_layout(recovered.model) == final_layout
        assert tensor_digest(list(recovered.model.named_parameters()) + list(recovered.model.named_buffers())) == before_reload
        namespace.clear(); del stable, model, method
        method = recovered; model = method.model; formal.onevision_bridge.bind(model)
        gc.collect(); torch.cuda.empty_cache()
        from generation import generate
        predictions = generate(method, processor, manifest, dev, run, checked_input)
        assert tensor_digest(list(model.named_parameters()) + list(model.named_buffers())) == before_reload
        memory = finish_resources(monitor, model, phase)
        training_inputs.flush(); os.fsync(training_inputs.fileno()); training_inputs.close()
        files = [p for folder in ('gradients', 'optimizer', 'layouts', 'validity') for p in (run / folder).iterdir()]
        files += [run / 'NATIVE_GATE.json', run / 'FFN_ADAPTATION.json', run / 'config.json', run / 'INITIAL_BINDING.json', run / 'logs/events.jsonl', run / 'training_input_IDs.jsonl',
                  run / 'resources.jsonl', run / 'memory_summary.json', *sorted((run / 'checkpoints').glob('*.json'))]
        file_hashes = {p.relative_to(run).as_posix(): sha(p) for p in files}
        assert time.monotonic() - begin <= cap
        write(public / 'COMPLETE.json', dict(complete=True, phase=phase, worker=process_identity(),
            model_RUN_seconds=time.monotonic() - begin, updates=87, FIT_QA=1213, DEV_QA=100, TEST_QA=0,
            final_layout=final_layout, predictions=predictions, full_reload=True, final_state_digest=before_reload,
            mask_digest=masks_before, stable_before=stable_before, stable_after=stable_before,
            training_coordinates=expected_trainable, file_sha256=file_hashes, memory=memory,
            cache_path_before=cache_before, cache_path_after=formal.visual_cache_stats(model),
            native_core_bypass_NLL_paths=19, native_core_bypass_short_generations=19,
            cache_counts_are_not_total_visual_tower_calls=True,
            ready_sha256=sha(REPORT / 'READY.json')))
    signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--phase', choices=('init', *CASES), required=True)
    ap.add_argument('--physical-gpu', type=int, required=True); args = ap.parse_args()
    try:
        main(args.phase, args.physical_gpu)
    except ResourceBusy as error:
        print(str(error), flush=True); sys.exit(75)
    except BaseException:
        write(REPORT / 'phases' / args.phase / 'FAILURE.json', dict(traceback=traceback.format_exc(), no_retry=True))
        raise
