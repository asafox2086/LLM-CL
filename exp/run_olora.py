import argparse
import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup

from common import ROOT, Scorer, continual_metrics, fingerprint, prepare_data, tokenize_examples, write_json


spec = importlib.util.spec_from_file_location('llmcl_olora', ROOT / 'code/olora.py')
method_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(method_module)


def batch_tensors(examples, tokenizer, device, training):
    sequences = [example['input_ids'] if training else example['prompt_ids'] for example in examples]
    width = max(map(len, sequences))
    inputs, masks, labels = [], [], []
    for example, sequence in zip(examples, sequences):
        padding = width - len(sequence)
        inputs.append(sequence + [tokenizer.pad_token_id] * padding if training else [tokenizer.pad_token_id] * padding + sequence)
        masks.append([1] * len(sequence) + [0] * padding if training else [0] * padding + [1] * len(sequence))
        if training:
            labels.append(example['labels'] + [-100] * padding)
    batch = {'input_ids': torch.tensor(inputs, device=device), 'attention_mask': torch.tensor(masks, device=device)}
    if training:
        batch['labels'] = torch.tensor(labels, device=device)
    return batch


def evaluate(model, tokenizer, examples, config, output, metadata):
    model.eval()
    scorer = Scorer()
    totals = {'rougeL': 0.0, 'exact_match': 0.0, 'token_f1': 0.0}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.jsonl.tmp')
    device = model.get_input_embeddings().weight.device
    started = time.monotonic()
    with temporary.open('w') as handle, torch.inference_mode():
        for start in range(0, len(examples), config['eval_batch_size']):
            selected = examples[start:start + config['eval_batch_size']]
            batch = batch_tensors(selected, tokenizer, device, False)
            generated = model.generate(**batch, do_sample=False, num_beams=1,
                                       max_new_tokens=config['target_tokens'], use_cache=True,
                                       pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            predictions = tokenizer.batch_decode(generated[:, batch['input_ids'].shape[1]:], skip_special_tokens=True)
            for example, prediction in zip(selected, predictions):
                scores = scorer.score(prediction, example['references'])
                for metric, score in scores.items():
                    totals[metric] += score
                handle.write(json.dumps({**metadata, 'task_id': example['task_id'], 'instance_id': example['instance_id'],
                                         'prediction': prediction, 'references': example['references'], 'scores': scores}, ensure_ascii=False) + '\n')
            if start % (config['eval_batch_size'] * 25) == 0:
                print(json.dumps({'event': 'evaluation_progress', **metadata, 'task': examples[0]['task_id'],
                                  'done': min(start + len(selected), len(examples)), 'total': len(examples)}), flush=True)
    temporary.replace(output)
    return {**{metric: score / len(examples) for metric, score in totals.items()},
            'count': len(examples), 'seconds': time.monotonic() - started}


def train_task(method, tokenizer, task_data, config, destination, task_id):
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
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp, init_scale=1024)
    checkpoint = destination / 'best.pt'
    best_score, update = -math.inf, 0
    started = time.monotonic()
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / 'training.jsonl').open('w') as log:
        for epoch in range(config['epochs']):
            model.train()
            order = list(range(len(task_data['train'])))
            random.Random(config['seed'] + epoch).shuffle(order)
            for offset in range(0, len(order), config['batch_size']):
                indices = order[offset:offset + config['batch_size']]
                optimizer.zero_grad(set_to_none=True)
                loss_value = 0.0
                for micro_start in range(0, len(indices), config['micro_batch_size']):
                    selected = [task_data['train'][index] for index in indices[micro_start:micro_start + config['micro_batch_size']]]
                    batch = batch_tensors(selected, tokenizer, device, True)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']).logits
                        shifted_logits = logits[:, :-1, :].float().contiguous()
                        shifted_labels = batch['labels'][:, 1:].contiguous()
                        token_loss = torch.nn.functional.cross_entropy(
                            shifted_logits.view(-1, shifted_logits.shape[-1]), shifted_labels.view(-1),
                            ignore_index=-100, reduction='none').view(shifted_labels.shape)
                        per_example = token_loss.sum(dim=1) / (shifted_labels != -100).sum(dim=1).clamp_min(1)
                        loss = per_example.mean() + method.penalty(config['orthogonal_weight'], config['l2_weight'])
                        loss = loss * len(selected) / len(indices)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'Nonfinite loss at {task_id}, epoch={epoch}, update={update}')
                    scaler.scale(loss).backward()
                    loss_value += loss.detach().item()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, config['max_grad_norm'])
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError(f'Nonfinite gradient at {task_id}, update={update}')
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                update += 1
                record = {'event': 'training_update', 'task': task_id, 'epoch': epoch + 1, 'update': update,
                          'total_updates': total_updates, 'loss': loss_value, 'lr': scheduler.get_last_lr()[0]}
                log.write(json.dumps(record) + '\n')
                log.flush()
                if update == 1 or update % 5 == 0:
                    print(json.dumps(record), flush=True)
            scores = evaluate(model, tokenizer, task_data['dev'], config,
                              destination / f'dev_epoch_{epoch + 1:02d}.jsonl', {'split': 'dev', 'epoch': epoch + 1})
            write_json(destination / f'dev_epoch_{epoch + 1:02d}.json', scores)
            if scores['rougeL'] > best_score:
                best_score = scores['rougeL']
                method.save(checkpoint)
                write_json(destination / 'selection.json', {'epoch': epoch + 1, 'dev_rougeL': best_score})
    method.load(checkpoint)
    model.gradient_checkpointing_disable()
    model.disable_input_require_grads()
    model.config.use_cache = True
    return {'updates': update, 'seconds': time.monotonic() - started,
            'current_trainable_parameters': sum(parameter.numel() for parameter in parameters),
            'total_adapter_parameters': sum(parameter.numel() for layer in method.layers.values() for parameter in layer.adapters.parameters())}


def run(args):
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new run directory; refusing to overwrite {output}')
    output.mkdir(parents=True, exist_ok=True)
    config['job'] = args.job
    config['single_task'] = args.single_task
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
        if not smoke and (model_config.get('model_type') != 'llama' or model_config.get('hidden_size') != 4096
                          or model_config.get('num_hidden_layers') != 32 or model_config.get('vocab_size') != 32000):
            raise ValueError('Expected Llama-2-7B; refusing to substitute another model')
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, use_fast=True)
        if tokenizer.bos_token_id is None or tokenizer.eos_token_id is None:
            raise ValueError('Tokenizer must define BOS and EOS')
        tokenizer.pad_token = tokenizer.eos_token
        statistics = {}
        for task_id, splits in datasets.items():
            statistics[task_id] = {}
            for split, examples in splits.items():
                if smoke:
                    examples = examples[:2]
                datasets[task_id][split], statistics[task_id][split] = tokenize_examples(tokenizer, examples, config)
        write_json(output / 'tokenization.json', statistics)
        kwargs = {'local_files_only': True, 'attn_implementation': 'sdpa'}
        if config['precision'] == 'nf4_fp16':
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required for NF4 training')
            kwargs.update(torch_dtype=torch.float16, device_map={'': 0},
                          quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                                 bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16))
        elif smoke and config['precision'] == 'fp32_cpu':
            kwargs.update(torch_dtype=torch.float32)
        else:
            raise ValueError('Unsupported precision')
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
        method = method_module.OLoRA(model, config['rank'], config['alpha'], config['dropout'], config['targets'])
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
        write_json(output / 'provenance.json', provenance)
        (output / 'environment.txt').write_text(subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True))
        resources = []
        metadata = {'protocol': config['protocol'], 'method': config['method'], 'order': config['order'], 'seed': config['seed']}
        if args.job == 'single':
            if args.single_task not in config['tasks']:
                raise ValueError('Single-task job needs a valid task ID')
            task_id = args.single_task
            resources.append(train_task(method, tokenizer, datasets[task_id], config, output / 'checkpoints' / task_id, task_id))
            scores = evaluate(model, tokenizer, datasets[task_id]['test'], config, output / 'predictions' / f'{task_id}.jsonl',
                              {**metadata, 'split': 'test', 'stage': 1, 'job': 'single'})
            write_json(output / 'single_score.json', {'task_id': task_id, **scores})
        else:
            matrix = []
            for stage in range(len(config['tasks']) + 1):
                if stage:
                    task_id = config['tasks'][stage - 1]
                    write_json(output / 'status.json', {'status': 'running', 'phase': 'train', 'stage': stage, 'task': task_id, 'pid': os.getpid()})
                    resources.append(train_task(method, tokenizer, datasets[task_id], config,
                                                output / 'checkpoints' / f'stage_{stage:02d}', task_id))
                row = []
                for task_id in config['tasks']:
                    write_json(output / 'status.json', {'status': 'running', 'phase': 'test', 'stage': stage, 'task': task_id, 'pid': os.getpid()})
                    scores = evaluate(model, tokenizer, datasets[task_id]['test'], config,
                                      output / 'predictions' / f'stage_{stage:02d}' / f'{task_id}.jsonl',
                                      {**metadata, 'split': 'test', 'stage': stage, 'job': 'continual'})
                    row.append(scores['rougeL'])
                    write_json(output / 'scores' / f'stage_{stage:02d}' / f'{task_id}.json', scores)
                matrix.append(row)
                write_json(output / 'score_matrix.json', {'tasks': config['tasks'], 'rows': matrix})
            write_json(output / 'metrics.json', continual_metrics(matrix))
        write_json(output / 'resources.json', {'stages': resources, 'wall_seconds': time.monotonic() - started,
                   'peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   'peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0})
        write_json(output / 'status.json', {'status': 'completed', 'job': args.job, 'smoke': smoke})
    except BaseException as error:
        write_json(output / 'status.json', {'status': 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                                          'error': str(error), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--job', choices=['continual', 'single'], default='continual')
    parser.add_argument('--single-task')
    run(parser.parse_args())
