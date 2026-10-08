"""Real Phi-2 FP16 compatibility/memory check; never publish as experiment scores."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, prepare_data, tokenize_examples, write_json
from run_olora import batch_tensors, create_method, evaluate, train_task

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
config = json.loads((ROOT / f'exp/configs/{args.method}_phi2.json').read_text())
model_path = ROOT / 'model/phi-2'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
    torch_dtype=torch.float16, device_map={'': 0}, attn_implementation='sdpa')
method = create_method(model, config)
datasets, _ = prepare_data(config)
task = config['tasks'][0]
train, _ = tokenize_examples(tokenizer, datasets[task]['train'], config)
train = sorted(train, key=lambda row: len(row['input_ids']), reverse=True)[:2]
test, _ = tokenize_examples(tokenizer, datasets[task]['test'], config)
test = sorted(test, key=lambda row: len(row['prompt_ids']), reverse=True)[:4]
# Exercise maximum task capacity and subsequent-task gradient masking.
for _ in range(6):
    method.begin_task()
check_config = {**config, 'batch_size': 1, 'warmup_ratio': 0,
                'checkpoint_selection': 'last_epoch', 'smoke': True}
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
resources = train_task(method, tokenizer, {'train': train}, check_config,
                       args.output / 'training', task)
# Verify recovery at maximum capacity as well as the model-specific forward path.
scores = evaluate(model, tokenizer, test, config, args.output / 'predictions.jsonl',
                  {'smoke': True, 'stage': 7}, method)
if args.method == 'sapt_lora':
    resources['auxiliary'] = method.finish_task(tokenizer, train, args.output / 'reflection',
        lambda meth, tok, data, cfg, dest, task_id: train_task(
            meth, tok, data, {**cfg, 'smoke': True, 'batch_size': 1, 'warmup_ratio': 0}, dest, task_id))
report = {'method': args.method, 'model_id': config['model_id'], 'precision': config['precision'],
          'smoke': True, 'train_lengths': [len(row['input_ids']) for row in train],
          'eval_prompt_lengths': [len(row['prompt_ids']) for row in test],
          'eval_batch_size': len(test), 'max_new_tokens': config['target_tokens'],
          'seconds': time.monotonic()-started, 'evaluation': scores, 'training': resources,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
write_json(args.output / 'report.json', report)
print(json.dumps(report), flush=True)
