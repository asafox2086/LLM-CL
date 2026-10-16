"""Full-data and nested 1% continual QA, with mandatory resource instrumentation."""
import argparse
import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, get_cosine_schedule_with_warmup

from methods import ROOT, make_method
from runtime_utils import input_batch, answer_loss, evaluate, make_optimizer, activation_storage, write
from frozen_vision import install, prepare
from resource_monitor import ResourceMonitor
from final_parameters import save_final

HERE = Path(__file__).resolve().parent
METHODS = ['SeqFT_LoRA', 'SMoLoRA', 'PASs_MoE', 'Hyper_LLaVA', 'PaRSP',
           'OSFT', 'SpaRTA', 'FV_guided_Training', 'ALER']


def load_protocol(condition):
    policy = json.loads((HERE / 'execution_policy.json').read_text())
    info = policy['conditions'][condition]
    path = ROOT / info['manifest']
    manifest = json.loads(path.read_text())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == info['manifest_sha256']
    assert manifest['formal_ready'] and not manifest['assets_pending']
    assert manifest['condition'] == condition and manifest['seed'] == 42
    assert manifest['experiment_id'] == HERE.name and len(manifest['tasks']) == 6
    assert policy.get('training_enabled') is True
    assert policy.get('save_final_parameters') is True
    data = {}
    for task in manifest['tasks']:
        for split, f in task['files'].items():
            p = ROOT / f['path']
            assert hashlib.sha256(p.read_bytes()).hexdigest() == f['sha256']
            rows = [json.loads(x) for x in p.read_text().splitlines()]
            assert len(rows) == f['count'] and len({r['id'] for r in rows}) == len(rows)
            data[task['task'], split] = rows
        assert not {x['id'] for x in data[task['task'], 'train']} & {x['id'] for x in data[task['task'], 'test']}
    return policy, manifest, digest, data


def aggregate(run, manifest):
    matrices = {score: [] for score in ['exact_match', 'token_f1', 'rougeL']}
    for stage in range(len(manifest['tasks'])+1):
        for score in matrices:
            row = []
            for t in manifest['tasks']:
                p = run / 'evaluations' / f'stage_{stage:02d}' / f"T{t['task']:02d}_test_predictions.scores.json"
                if not p.exists():
                    return
                result = json.loads(p.read_text())
                assert result['count'] == t['files']['test']['count']
                row.append(result[score])
            matrices[score].append(row)
    from result_tables import metrics
    import csv
    for score, values in matrices.items():
        with (run / f'{score}_{len(values)}x{len(manifest["tasks"])}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['stage']+[task['task_id'] for task in manifest['tasks']])
            writer.writerows([[stage]+row for stage,row in enumerate(values)])
    write(run / 'score_matrices.json', matrices)
    write(run / 'metrics_by_score.json', {s: metrics(a) for s, a in matrices.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', choices=METHODS, required=True)
    ap.add_argument('--condition', choices=['full', 'pct01'], required=True)
    ap.add_argument('--run-dir', type=Path, required=True)
    ap.add_argument('--preflight-only', action='store_true')
    args = ap.parse_args()
    policy, manifest, digest, data = load_protocol(args.condition)
    physical = int(os.environ['CUDA_VISIBLE_DEVICES'])
    assert physical in policy['conditions'][args.condition]['allowed_gpus'] and physical != 0
    assert not policy['save_checkpoints'] and policy['seed'] == 42
    run = args.run_dir.resolve()
    assert run.is_relative_to(HERE / 'runs')
    assert not (run / 'config.json').exists(), 'Fresh model run required; no checkpoint or old predictions reused'
    (run / 'logs').mkdir(parents=True, exist_ok=True)
    monitor = ResourceMonitor(run)
    monitor.snapshot('process_before_model')
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(8)
    model_path = ROOT / 'model/Qwen2-VL-2B-Instruct'
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = Qwen2VLForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.float16,
                 attn_implementation='sdpa', local_files_only=True).cuda()
    name = 'SeqLoRA' if args.method == 'SeqFT_LoRA' else args.method
    method = make_method(model, name, [t['instruction'] for t in manifest['tasks']])
    if args.method == 'SeqFT_LoRA':
        method.name = args.method
        method.original_revision = 'standard-sequential-LoRA-qv-r8-alpha32-dropout0.1'
    if hasattr(method, 'state_root'):
        method.state_root = run / 'method_state'
        method.save_checkpoints = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    if hasattr(method, 'bind_context'):
        method.bind_context(processor, manifest, data, run, args.preflight_only)
    method.vision_original = install(model)
    task_counts = {t['task_id']: {s: f['count'] for s, f in t['files'].items()} for t in manifest['tasks']}
    config = {'experiment_id': HERE.name, 'condition': args.condition, 'method': args.method,
              'seed': 42, 'epochs': 1, 'manifest_sha256': digest, 'task_counts': task_counts,
              'save_checkpoints': False, 'save_final_parameters': True, 'batch_size': 16, 'micro_batch_size': 1,
              'base_model': str(model_path), 'primary_score': 'rougeL',
              'test_fraction': 1.0, 'train_fraction': 1.0 if args.condition == 'full' else .01,
              'preflight_only': args.preflight_only,
              'execution_source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    write(run / 'config.json', config)
    gc.collect()
    torch.cuda.empty_cache()
    monitor.snapshot('model_loaded_before_initial_evaluation', model=model)
    write(run / 'run_status.json', {'status': 'initial_evaluation', 'completed_stage': 0})
    if args.preflight_only:
        monitor.snapshot('training_before', model=model)
        # Test the same gradient and resource paths on a real new-task image and a text QA.
        method.begin_stage(1)
        optimizer = make_optimizer(method)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        for tid in [1, 2, 3, 4, 5]:
            method.set_instruction(tid)
            row = data[tid, 'train'][0]
            batch, _ = input_batch(row, manifest['tasks'][tid-1], processor, model.device)
            prepare(batch, row, model, method.vision_original)
            if hasattr(method, 'prepare_row'):
                method.prepare_row(row, batch)
            optimizer.zero_grad(set_to_none=True)
            if hasattr(method, 'prepare_update'):
                method.prepare_update(processor)
            if hasattr(method, 'recording'):
                method.recording(True)
            with activation_storage(method), torch.autocast('cuda', dtype=torch.float16):
                loss, _ = method.training_loss(batch, row, processor) if hasattr(method, 'training_loss') else answer_loss(model, batch, row['answer'], processor)
                if hasattr(method, 'extra_loss'):
                    loss = loss + method.extra_loss()
            if hasattr(method, 'recording'):
                method.recording(False)
            scaler.scale(loss).backward()
            if hasattr(method, 'post_backward'):
                method.post_backward(scaler.get_scale(), .5)
            scaler.unscale_(optimizer)
            grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
            assert torch.isfinite(loss) and torch.isfinite(grad)
            scaler.step(optimizer)
            scaler.update()
            print(json.dumps({'event': 'preflight', 'task': tid, 'loss': float(loss.detach()), 'gradient_l2_preclip': float(grad)}), flush=True)
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scaler, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        monitor.snapshot('training_after', model=model)
        monitor.finish(model)
        write(run / 'gpu_preflight.json', {'passed': True, 'weights_discarded': True, 'not_formal_results': True})
        return
    evaluate(method, processor, manifest['tasks'], data, run, 0)
    gc.collect()
    torch.cuda.empty_cache()
    monitor.snapshot('training_before', model=model)
    for stage in range(1, len(manifest['tasks'])+1):
        monitor.stage = stage
        monitor.snapshot('stage_before_auxiliary', stage, model)
        write(run / 'run_status.json', {'status': 'training', 'completed_stage': stage-1, 'active_stage': stage})
        method.begin_stage(stage)
        monitor.snapshot('stage_after_auxiliary', stage, model)
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = make_optimizer(method)
        rows = data[stage, 'train']
        # Same deterministic order for all methods; independent of initialization RNG.
        order = sorted(rows, key=lambda x: hashlib.sha256(f"42:{stage}:{x['id']}".encode()).digest())
        assert hashlib.sha256(''.join(x['id']+'\n' for x in order).encode()).hexdigest()==manifest['tasks'][stage-1]['training_order_sha256']
        updates = math.ceil(len(order)/16)
        scheduler = get_cosine_schedule_with_warmup(optimizer, math.ceil(.03*updates), updates)
        scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        method.set_instruction(stage)
        model.train()
        torch.cuda.reset_peak_memory_stats()
        with (run / 'logs/events.jsonl').open('a', encoding='utf-8') as events:
            for step, start in enumerate(range(0, len(order), 16), 1):
                tick = time.monotonic()
                optimizer.zero_grad(set_to_none=True)
                minibatch = order[start:start+16]
                if hasattr(method, 'prepare_update'):
                    method.prepare_update(processor)
                loss_sum = 0.
                target_tokens = prompt_tokens = 0
                for index, row in enumerate(minibatch):
                    batch, details = input_batch(row, manifest['tasks'][stage-1], processor, model.device)
                    prepare(batch, row, model, method.vision_original)
                    if hasattr(method, 'prepare_row'):
                        method.prepare_row(row, batch)
                    if hasattr(method, 'recording'):
                        method.recording(True)
                    with activation_storage(method), torch.autocast('cuda', dtype=torch.float16):
                        loss, length = method.training_loss(batch, row, processor) if hasattr(method, 'training_loss') else answer_loss(model, batch, row['answer'], processor)
                        if hasattr(method, 'extra_loss') and (getattr(method, 'extra_loss_per_micro', False) or index == len(minibatch)-1):
                            loss = loss + method.extra_loss()
                    if hasattr(method, 'recording'):
                        method.recording(False)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('nonfinite loss')
                    scaler.scale(loss/len(minibatch)).backward()
                    if hasattr(method, 'post_backward'):
                        method.post_backward(scaler.get_scale(), start/len(order))
                    loss_sum += float(loss.detach())/len(minibatch)
                    target_tokens += length
                    prompt_tokens += details['prompt_tokens']
                scaler.unscale_(optimizer)
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1.)
                if not torch.isfinite(grad):
                    raise FloatingPointError('nonfinite gradient')
                postclip=torch.linalg.vector_norm(torch.stack([p.grad.detach().float().norm(2)
                    for p in parameters if p.grad is not None]),2)
                scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() < scale:
                    raise FloatingPointError('optimizer step skipped')
                scheduler.step()
                if hasattr(method, 'after_update'):
                    method.after_update(step, updates)
                record = {'event': 'train_update', 'stage': stage, 'step': step,
                          'qa_seen': start+len(minibatch), 'total_qa': len(order), 'loss': loss_sum,
                          'gradient_l2_preclip': float(grad), 'gradient_l2_postclip': float(postclip),
                          'gradient_clip_threshold': 1., 'learning_rate': scheduler.get_last_lr(),
                          'seconds': time.monotonic()-tick, 'prompt_tokens': prompt_tokens,
                          'target_tokens': target_tokens,
                          'gpu_allocated_bytes': torch.cuda.memory_allocated(),
                          'gpu_reserved_bytes': torch.cuda.memory_reserved(),
                          'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
                events.write(json.dumps(record)+'\n')
                events.flush()
                print(json.dumps(record), flush=True)
        monitor.snapshot('stage_training_end_with_optimizer', stage, model)
        optimizer.zero_grad(set_to_none=True)
        if hasattr(method, 'end_stage'):
            method.end_stage(stage)
        del optimizer, scheduler, scaler, batch, loss
        gc.collect()
        torch.cuda.empty_cache()
        monitor.snapshot('stage_training_end_persistent', stage, model)
        if stage == len(manifest['tasks']):
            monitor.snapshot('training_after', stage, model)
        write(run / 'run_status.json', {'status': 'evaluation', 'completed_stage': stage-1, 'active_stage': stage})
        scores = evaluate(method, processor, manifest['tasks'], data, run, stage)
        write(run / 'stages' / f'stage_{stage:02d}.json', {'complete': True, 'weights_saved': False,
              'stage': stage, 'task_scores': scores, 'train_qa': len(order), 'test_fraction': 1.0})
        monitor.snapshot('stage_after_evaluation', stage, model)
        write(run / 'run_status.json', {'status': 'stage_complete', 'completed_stage': stage})
    aggregate(run, manifest)
    monitor.finish(model)
    write(run / 'run_status.json', {'status': 'saving_final_parameters', 'completed_stage': len(manifest['tasks'])})
    artifact = save_final(method, processor, run, manifest, config)
    write(run / 'run_status.json', {'status': 'completed', 'completed_stage': len(manifest['tasks']),
          'final_parameters': 'final_parameters/artifact_manifest.json', 'final_parameter_file_bytes': artifact['total_file_bytes']})


if __name__ == '__main__':
    main()
