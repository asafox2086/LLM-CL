"""Optimizer-step recovery, using the existing experimental training protocol."""
import json
import math
import os
import random
import time
import torch
from transformers import get_linear_schedule_with_warmup
from common import write_json
from recovery import save_training, load_training, save_method
from cv_runtime import batch_tensors, answer_loss, evaluate

def train_task(method, tokenizer, task_data, config, destination, task_id, resume=False):
    model = method.model
    method.begin_task()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    parameters = method.parameters()
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], betas=(0.9, 0.999),
                                 eps=1e-8, weight_decay=config['weight_decay'])
    updates_per_epoch = math.ceil(len(task_data['train']) / config['batch_size'])
    total_updates = updates_per_epoch * config['epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, math.ceil(total_updates * config['warmup_ratio']), total_updates)
    device = model.get_input_embeddings().weight.device
    use_amp = device.type == 'cuda' and config['precision'] != 'fp32_cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024)
    checkpoint = destination / 'best.pt'
    recovery_path = destination / 'recovery.pt'
    best_score, update = -math.inf, 0
    first_epoch, first_offset, prior_seconds = 0, 0, 0.0
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    if resume and recovery_path.exists():
        state = load_training(recovery_path, method, config, optimizer, scheduler, scaler)
        first_epoch, first_offset = state['epoch'], state['offset']
        best_score, update, prior_seconds = state['best_score'], state['update'], state['elapsed_seconds']
        print(json.dumps({'event': 'training_resume', 'task': task_id, 'epoch': first_epoch,
                          'next_offset': first_offset, 'completed_updates': update}), flush=True)
    elif resume and (destination / 'training.jsonl').exists():
        raise ValueError(f'Training exists without a recovery checkpoint: {destination}')
    log_path = destination / 'training.jsonl'
    previous_records = []
    if resume and log_path.exists():
        for line in log_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record['update'] <= update:
                previous_records.append(record)

    def checkpoint_training(epoch, offset):
        save_training(recovery_path, method, config, optimizer, scheduler, scaler,
                      epoch=epoch, offset=offset, update=update, best_score=best_score,
                      task_id=task_id, elapsed_seconds=prior_seconds + time.monotonic() - started)

    if not recovery_path.exists():
        checkpoint_training(0, 0)
    with log_path.open('w') as log:
        for record in previous_records:
            log.write(json.dumps(record) + '\n')
        for epoch in range(first_epoch, config['epochs']):
            model.train()
            order = list(range(len(task_data['train'])))
            random.Random(config['seed'] + epoch).shuffle(order)
            initial_offset = first_offset if epoch == first_epoch else 0
            for offset in range(initial_offset, len(order), config['batch_size']):
                indices = order[offset:offset + config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                if hasattr(method, 'before_update'):
                    method.before_update(update)
                update_started = time.monotonic()
                loss_value = 0.0
                ce_value, penalty_value, token_count = 0.0, 0.0, 0
                for micro_start in range(0, len(indices), config['micro_batch_size']):
                    selected = [task_data['train'][index] for index in indices[micro_start:micro_start + config['micro_batch_size']]]
                    batch = batch_tensors(selected, tokenizer, device, True, model.config.is_encoder_decoder)
                    if hasattr(method, 'prepare_batch'):
                        method.prepare_batch(selected, tokenizer, batch, training=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        base_loss = answer_loss(model, batch)
                        regularization = method.penalty(config['orthogonal_weight'], config['l2_weight'])
                        ce_value += base_loss.detach().item() * len(selected) / len(indices)
                        penalty_value += float(regularization.detach() if torch.is_tensor(regularization) else regularization) * len(selected) / len(indices)
                        token_count += int(batch['attention_mask'].sum().item())
                        loss = base_loss + regularization
                        loss = loss * len(selected) / len(indices)
                    if hasattr(method, 'after_forward'):
                        method.after_forward()
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at {task_id}, epoch={epoch}, update={update}')
                    scaler.scale(loss).backward()
                    loss_value += loss.detach().item()
                scaler.unscale_(optimizer)
                if hasattr(method, 'before_step'):
                    method.before_step()
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f'Nonfinite gradient at {task_id}, update={update}')
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                update += 1
                record = {'event': 'training_update', 'task': task_id, 'epoch': epoch + 1, 'update': update,
                          'total_updates': total_updates, 'loss': loss_value, 'answer_cross_entropy': ce_value,
                          'regularization_loss': penalty_value, 'input_tokens': token_count, 'update_seconds': time.monotonic()-update_started, 'lr': scheduler.get_last_lr()[0],
                          'gradient_norm': grad_norm.item(), 'amp_scale': scaler.get_scale(), 'elapsed_seconds': prior_seconds + time.monotonic()-started,
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else 0,
                          'instance_ids': [task_data['train'][index].get('instance_id') for index in indices]}
                log.write(json.dumps(record) + '\n')
                log.flush()
                checkpoint_training(epoch, offset + len(indices))
                if update == 1 or update % 5 == 0:
                    print(json.dumps(record), flush=True)
            if config.get('checkpoint_selection') == 'last_epoch':
                save_method(method, checkpoint)
                write_json(destination / 'selection.json', {'epoch': epoch + 1, 'selection': 'fixed_last_epoch'})
                checkpoint_training(epoch + 1, 0)
                continue
            scores = evaluate(model, tokenizer, task_data['dev'], config,
                              destination / f'dev_epoch_{epoch + 1:02d}.jsonl', {'split': 'dev', 'epoch': epoch + 1}, method, resume)
            write_json(destination / f'dev_epoch_{epoch + 1:02d}.json', scores)
            if scores['rougeL'] > best_score:
                best_score = scores['rougeL']
                save_method(method, checkpoint)
                write_json(destination / 'selection.json', {'epoch': epoch + 1, 'dev_rougeL': best_score})
            checkpoint_training(epoch + 1, 0)
    method.load(checkpoint)
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    model.config.use_cache = True
    resources = {'updates': update, 'seconds': prior_seconds + time.monotonic() - started,
            'recovery_checkpoint': str(recovery_path), 'best_checkpoint': str(checkpoint),
            'current_trainable_parameters': sum(parameter.numel() for parameter in parameters),
            'total_adapter_parameters': sum(parameter.numel() for layer in method.layers.values() for parameter in layer.adapters.parameters()),
            'generator_parameters': sum(parameter.numel() for layer in method.layers.values()
                                        if getattr(layer, 'generator', None) is not None for parameter in layer.generator.parameters()),
            'router_parameters': sum(parameter.numel() for parameter in method.router.parameters()) if hasattr(method, 'router') else 0}
    write_json(destination / 'resources.json', resources)
    return resources

