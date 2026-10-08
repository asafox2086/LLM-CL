import argparse
import io
import importlib.util
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from functools import partial

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from common import ROOT, Scorer, continual_metrics, fingerprint, prepare_data, tokenize_examples, write_json, validate_model_config
from recovery import atomic_save, capture_rng, restore_rng, method_bytes, load_training, save_method, save_training


def create_method(model, config):
    constructors = {'olora': 'OLoRA', 'migu_lora': 'MIGULoRA', 'sapt_lora': 'SAPTLoRA', 'seq_lora': 'SeqLoRA'}
    name = config['method']
    if name not in constructors:
        raise ValueError(f'Unsupported method: {name}')
    spec = importlib.util.spec_from_file_location(f'llmcl_{name}', ROOT / 'code' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if name == 'olora':
        return module.OLoRA(model, config['rank'], config['alpha'], config['dropout'], config['targets'])
    return getattr(module, constructors[name])(model, config)


def batch_tensors(examples, tokenizer, device, training, seq2seq=False):
    sequences = [example['input_ids'] if training else example['prompt_ids'] for example in examples]
    width = max(map(len, sequences))
    inputs, masks, labels = [], [], []
    right_pad = training or seq2seq
    label_width = max(len(example['labels']) for example in examples) if training and seq2seq else width
    for example, sequence in zip(examples, sequences):
        padding = width - len(sequence)
        inputs.append(sequence + [tokenizer.pad_token_id] * padding if right_pad else [tokenizer.pad_token_id] * padding + sequence)
        masks.append([1] * len(sequence) + [0] * padding if right_pad else [0] * padding + [1] * len(sequence))
        if training:
            labels.append(example['labels'] + [-100] * (label_width - len(example['labels'])))
    batch = {'input_ids': torch.tensor(inputs, device=device), 'attention_mask': torch.tensor(masks, device=device)}
    if training:
        batch['labels'] = torch.tensor(labels, device=device)
        if seq2seq:
            batch['decoder_attention_mask'] = (batch['labels'] != -100).long()
    return batch


def evaluate(model, tokenizer, examples, config, output, metadata, method=None, resume=False):
    model.eval()
    scorer = Scorer()
    totals = {'rougeL': 0.0, 'exact_match': 0.0, 'token_f1': 0.0}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.jsonl.tmp')
    timing_path = output.with_suffix('.timing.json')
    records = []
    if resume:
        source = output if output.exists() else temporary
        if source.exists():
            lines = source.read_text().splitlines()
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    if source == temporary and index == len(lines) - 1:
                        break
                    raise
                if index >= len(examples):
                    raise ValueError(f'Too many cached predictions: {source}')
                example = examples[index]
                if (record['instance_id'] != example['instance_id'] or record['task_id'] != example['task_id']
                        or record['references'] != example['references']
                        or any(record.get(key) != value for key, value in metadata.items())):
                    raise ValueError(f'Cached prediction does not match this evaluation: {source}, row {index}')
                record.update(prompt=example['prompt'], prompt_token_ids=example['prompt_ids'])
                records.append(record)
            if source == output and len(records) != len(examples):
                raise ValueError(f'Completed prediction file has missing rows: {source}')
            if len(records) < len(examples):
                records = records[:len(records) // config['eval_batch_size'] * config['eval_batch_size']]
    timing = json.loads(timing_path.read_text()) if resume and timing_path.exists() else {}
    recorded_seconds = timing.get('recorded_seconds', 0.0)
    unknown_previous_time = timing.get('unknown_previous_time', bool(records) and not timing)
    for record in records:
        for metric in totals:
            totals[metric] += record['scores'][metric]
    device = model.get_input_embeddings().weight.device
    started = time.monotonic()
    if records:
        print(json.dumps({'event': 'evaluation_resume', **metadata, 'task': examples[0]['task_id'],
                          'reused_predictions': len(records), 'total': len(examples)}), flush=True)
    rebuilding = output.with_suffix('.jsonl.rebuilding')
    with rebuilding.open('w') as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())
    rebuilding.replace(temporary)
    with temporary.open('a') as handle, torch.inference_mode():
        for start in range(len(records), len(examples), config['eval_batch_size']):
            selected = examples[start:start + config['eval_batch_size']]
            batch = batch_tensors(selected, tokenizer, device, False, model.config.is_encoder_decoder)
            if hasattr(method, 'prepare_batch'):
                method.prepare_batch(selected, tokenizer, batch, training=False)
            generated = model.generate(**batch, do_sample=False, num_beams=1,
                                       max_new_tokens=config['target_tokens'], use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            # Encoder-decoder outputs contain only decoder start + answer.
            start_token = 1 if model.config.is_encoder_decoder else batch['input_ids'].shape[1]
            token_rows = generated[:, start_token:].tolist()
            predictions = tokenizer.batch_decode(token_rows, skip_special_tokens=True)
            for example, prediction, tokens in zip(selected, predictions, token_rows):
                if tokenizer.eos_token_id in tokens:
                    tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
                scores = scorer.score(prediction, example['references'])
                for metric, score in scores.items():
                    totals[metric] += score
                handle.write(json.dumps({**metadata, 'task_id': example['task_id'], 'instance_id': example['instance_id'],
                                         'prompt': example['prompt'], 'prompt_token_ids': example['prompt_ids'],
                                         'prediction': prediction, 'generated_token_ids': tokens,
                                         'references': example['references'], 'scores': scores}, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
            write_json(timing_path, {'recorded_seconds': recorded_seconds + time.monotonic() - started,
                                    'unknown_previous_time': unknown_previous_time,
                                    'completed_predictions': start + len(selected)})
            if start % (config['eval_batch_size'] * 25) == 0:
                print(json.dumps({'event': 'evaluation_progress', **metadata, 'task': examples[0]['task_id'],
                                  'done': start + len(selected), 'total': len(examples)}), flush=True)
    temporary.replace(output)
    elapsed = recorded_seconds + time.monotonic() - started
    write_json(timing_path, {'recorded_seconds': elapsed, 'unknown_previous_time': unknown_previous_time,
                            'completed_predictions': len(examples)})
    return {**{metric: score / len(examples) for metric, score in totals.items()},
            'count': len(examples), 'seconds': None if unknown_previous_time else elapsed,
            'recorded_seconds': elapsed, 'reused_predictions': len(records)}


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
                loss_value = 0.0
                for micro_start in range(0, len(indices), config['micro_batch_size']):
                    selected = [task_data['train'][index] for index in indices[micro_start:micro_start + config['micro_batch_size']]]
                    batch = batch_tensors(selected, tokenizer, device, True, model.config.is_encoder_decoder)
                    if hasattr(method, 'prepare_batch'):
                        method.prepare_batch(selected, tokenizer, batch, training=True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        if model.config.is_encoder_decoder:
                            decoder_ids = model.prepare_decoder_input_ids_from_labels(labels=batch['labels'])
                            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                                           decoder_input_ids=decoder_ids,
                                           decoder_attention_mask=batch['decoder_attention_mask']).logits
                            shifted_logits = logits.float().contiguous()
                            shifted_labels = batch['labels'].contiguous()
                        else:
                            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                            shifted_logits = logits[:, :-1, :].float().contiguous()
                            shifted_labels = batch['labels'][:, 1:].contiguous()
                        token_loss = torch.nn.functional.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1),
                            ignore_index=-100, reduction='none').view(shifted_labels.shape)
                        per_example = token_loss.sum(dim=1) / (shifted_labels != -100).sum(dim=1).clamp_min(1)
                        loss = per_example.mean() + method.penalty(config['orthogonal_weight'], config['l2_weight'])
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
                          'total_updates': total_updates, 'loss': loss_value, 'lr': scheduler.get_last_lr()[0],
                          'gradient_norm': grad_norm.item(), 'amp_scale': scaler.get_scale(),
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


def run(args):
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise FileExistsError(f'Use a new run directory; refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    config['job'] = 'continual'
    config['evaluation_protocol'] = 'cl_standard_fwt_v1'
    if args.resume and (output / 'config.json').exists() and json.loads((output / 'config.json').read_text()) != config:
        raise ValueError('Resume worker configuration differs from the original run')
    write_json(output / 'config.json', config)
    started = time.monotonic()
    write_json(output / 'status.json', {'status': 'running', 'phase': 'prepare_data', 'pid': os.getpid()})
    try:
        random.seed(config['seed'])
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
            torch.cuda.reset_peak_memory_stats()
        torch.set_num_threads(4)
        datasets, split_manifest = prepare_data(config)
        write_json(output / 'split_manifest.json', split_manifest)
        model_path = Path(args.model).resolve()
        model_config = json.loads((model_path / 'config.json').read_text())
        smoke = config.get('smoke', False)
        validate_model_config(model_config, config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        seq2seq = bool(model_config.get('is_encoder_decoder'))
        if tokenizer.eos_token_id is None or (not seq2seq and tokenizer.bos_token_id is None):
            raise ValueError('Tokenizer must define EOS (and BOS for causal models)')
        if not seq2seq:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.pad_token_id is None:
            raise ValueError('Encoder-decoder tokenizer must define PAD')
        statistics = {}
        for task_id, splits in datasets.items():
            statistics[task_id] = {}
            for split, examples in splits.items():
                if smoke:
                    examples = examples[:2]
                datasets[task_id][split], statistics[task_id][split] = tokenize_examples(tokenizer, examples, config)
        write_json(output / 'tokenization.json', statistics)
        kwargs = {'local_files_only': True, 'attn_implementation': 'eager' if seq2seq else 'sdpa'}
        if config['precision'] == 'nf4_fp16':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for NF4 training')
            kwargs.update(torch_dtype=torch.float16, device_map={'': 0},
                          quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        elif config['precision'] == 'fp16':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for FP16 training')
            kwargs.update(torch_dtype=torch.float16, device_map={'': 0})
        elif config['precision'] == 'fp32_cuda':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for FP32 GPU training')
            kwargs.update(torch_dtype=torch.float32, device_map={'': 0})
        elif smoke and config['precision'] == 'fp32_cpu':
            kwargs.update(torch_dtype=torch.float32)
        else:
            raise ValueError('Unsupported precision')
        model_class = AutoModelForSeq2SeqLM if seq2seq else AutoModelForCausalLM
        model = model_class.from_pretrained(model_path, **kwargs)
        method = create_method(model, config)
        provenance = {'config_sha256': fingerprint(config), 'split_sha256': fingerprint(split_manifest),
                      'code_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                      'dirty_diff_sha256': fingerprint(subprocess.check_output(['git', 'diff', 'HEAD'], cwd=ROOT, text=True)),
                      'model_path': str(model_path), 'model_identity': json.loads((model_path / 'llmcl_identity.json').read_text())
                      if (model_path / 'llmcl_identity.json').exists() else {'smoke': smoke, 'unverified_local_model': True},
                      'python': sys.version, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
                      'command': sys.argv, 'smoke': smoke}
        if not smoke and provenance['model_identity'].get('unverified_local_model'):
            raise ValueError('Model identity file missing; use exp/launch_olora.py')
        provenance_path = output / 'provenance.json'
        if args.resume and provenance_path.exists():
            provenance_path = output / 'resume_provenance' / f'{time.time_ns()}.json'
        write_json(provenance_path, provenance)
        (output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        resources = []
        metadata = {'protocol': config['protocol'], 'method': config['method'], 'order': config['order'], 'seed': config['seed']}
        matrix = []
        for stage in range(len(config['tasks']) + 1):
            boundary = output / 'checkpoints' / f'stage_{stage:02d}' / 'completed.pt'
            if args.resume and boundary.exists():
                state = torch.load(boundary, map_location='cpu', weights_only=False)
                if state['config_sha256'] != fingerprint(config):
                    raise ValueError('Stage checkpoint configuration differs')
                if stage:
                    method.load(io.BytesIO(state['method']))
                restore_rng(state['rng'])
                matrix, resources = state['matrix'], state['resources']
                print(json.dumps({'event': 'stage_resume', 'completed_stage': stage}), flush=True)
                continue
            if stage:
                task_id = config['tasks'][stage - 1]
                write_json(output / 'status.json', {'status': 'running', 'phase': 'train', 'stage': stage, 'task': task_id, 'pid': os.getpid()})
                resources.append(train_task(method, tokenizer, datasets[task_id], config,
                                            output / 'checkpoints' / f'stage_{stage:02d}', task_id, args.resume))
                if hasattr(method, 'finish_task') and stage < len(config['tasks']):
                    write_json(output / 'status.json', {'status': 'running', 'phase': 'auxiliary_generator',
                               'stage': stage, 'task': task_id, 'pid': os.getpid()})
                    reflection_path = output / 'reflection' / f'stage_{stage:02d}'
                    if args.resume and (reflection_path / 'stage_state.pt').exists():
                        method.load(reflection_path / 'stage_state.pt')
                        resources[-1]['auxiliary'] = json.loads((reflection_path / 'resources.json').read_text())
                    else:
                        resources[-1]['auxiliary'] = method.finish_task(tokenizer, datasets[task_id]['train'], reflection_path,
                                                                       partial(train_task, resume=args.resume))
            row = []
            per_task = {}
            for task_id in config['tasks']:
                write_json(output / 'status.json', {'status': 'running', 'phase': 'test', 'stage': stage, 'task': task_id, 'pid': os.getpid()})
                scores = evaluate(model, tokenizer, datasets[task_id]['test'], config,
                                  output / 'predictions' / f'stage_{stage:02d}' / f'{task_id}.jsonl',
                                  {**metadata, 'split': 'test', 'stage': stage, 'job': 'continual'}, method, args.resume)
                row.append(scores['rougeL'])
                per_task[task_id] = scores
                write_json(output / 'scores' / f'stage_{stage:02d}' / f'{task_id}.json', scores)
            matrix.append(row)
            write_json(output / 'score_matrix.json', {'tasks': config['tasks'], 'rows': matrix})
            write_json(output / 'stages' / f'stage_{stage:02d}.json', {
                **metadata, 'stage': stage, 'config': config, 'config_sha256': fingerprint(config),
                'per_task': per_task, 'mean_all_task_rougeL': sum(row) / len(row),
                'mean_seen_task_rougeL': sum(row[:stage]) / stage if stage else None,
                'resources': resources[-1] if stage else None,
                'predictions': f'predictions/stage_{stage:02d}/',
                'evaluation_protocol': config['evaluation_protocol']})
            write_json(output / 'resources.json', {'stages': resources, 'wall_seconds_this_process': time.monotonic() - started,
                       'peak_allocated_bytes_this_process': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                       'peak_reserved_bytes_this_process': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
            atomic_save({'config_sha256': fingerprint(config), 'method': method_bytes(method),
                         'rng': capture_rng(), 'matrix': matrix, 'resources': resources}, boundary)
        write_json(output / 'metrics.json', {'evaluation_protocol': config['evaluation_protocol'], **continual_metrics(matrix)})
        write_json(output / 'resources.json', {'stages': resources, 'wall_seconds_this_process': time.monotonic() - started,
                   'peak_allocated_bytes_this_process': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   'peak_reserved_bytes_this_process': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
        write_json(output / 'status.json', {'status': 'completed', 'job': 'continual', 'smoke': smoke})
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                          'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}; use --resume to continue')

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    run(parser.parse_args())
