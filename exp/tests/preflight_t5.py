"""T5-Large real-GPU stage 6/7 updates, SAPT reflection, long-input generation."""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ROOT, prepare_data, tokenize_examples, write_json
from run_olora import create_method, evaluate, train_task

parser = argparse.ArgumentParser()
parser.add_argument('--method', required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(42)
config = json.loads((ROOT / f'exp/configs/{args.method}_t5large.json').read_text())
config['smoke'] = True
model_path = ROOT / 'model/t5-large'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True,
    torch_dtype=torch.float32, device_map={'': 0}, attn_implementation='eager')
method = create_method(model, config)
datasets, _ = prepare_data(config)
task = config['tasks'][0]
train, _ = tokenize_examples(tokenizer, datasets[task]['train'], config)
all_train = train
train = sorted(train, key=lambda row: len(row['input_ids']) + len(row['labels']), reverse=True)[:2]
test, _ = tokenize_examples(tokenizer, datasets[task]['test'], config)
test = sorted(test, key=lambda row: len(row['prompt_ids']), reverse=True)[:config['eval_batch_size']]
for _ in range(5):
    method.begin_task()
check_config = {**config, 'batch_size': 1, 'warmup_ratio': 0, 'checkpoint_selection': 'last_epoch'}
torch.cuda.reset_peak_memory_stats()
started = time.monotonic()
resources = [train_task(method, tokenizer, {'train': train}, check_config,
                       args.output / 'stage_06', task)]
if args.method == 'sapt_lora':
    # Tiny two-update reconstruction is insufficient for pretrained T5.
    # Exercise the actual 1000-example/63-update auxiliary budget and 128 samples.
    method.config['smoke'] = False
    resources[-1]['auxiliary'] = method.finish_task(tokenizer, all_train, args.output / 'reflection', train_task)
    method.config['smoke'] = True
resources.append(train_task(method, tokenizer, {'train': train}, check_config,
                            args.output / 'stage_07', task))
scores = evaluate(model, tokenizer, test, config, args.output / 'predictions.jsonl',
                  {'smoke': True, 'stage': 7}, method)
report = {'method': args.method, 'model_id': config['model_id'], 'precision': config['precision'],
          'smoke': True, 'train_shapes': [[len(row['input_ids']), len(row['labels'])] for row in train],
          'eval_prompt_lengths': [len(row['prompt_ids']) for row in test],
          'eval_batch_size': len(test), 'max_new_tokens': config['target_tokens'],
          'seconds': time.monotonic()-started, 'evaluation': scores, 'training': resources,
          'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
          'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
write_json(args.output / 'report.json', report)
print(json.dumps(report), flush=True)
