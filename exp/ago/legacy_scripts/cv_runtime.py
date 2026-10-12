"""Qwen2-VL cached-vision batching and exact answer-only causal loss."""
import json
import os
import time
from collections import defaultdict
from functools import lru_cache
import torch
from common import ROOT, Scorer, write_json
from cv_data import feature_path

@lru_cache(maxsize=2048)
def cached_features(path):
    return torch.load(path, map_location='cpu', weights_only=True)['features']

def install_cached_vision(model):
    # pixel_values carries frozen, audited merger outputs in this runner only.
    # Qwen's stock forward/generate still insert them at real image-token positions.
    def cached_forward(hidden_states, grid_thw):
        assert hidden_states.ndim == 2 and hidden_states.shape[1] == model.config.hidden_size
        assert hidden_states.shape[0] == int(grid_thw.prod(dim=1).sum().item()) // 4
        return hidden_states
    model.visual.forward = cached_forward

def batch_tensors(examples, tokenizer, device, training, seq2seq=False):
    sequences = [e['input_ids'] if training else e['prompt_ids'] for e in examples]
    width = max(map(len, sequences))
    ids, masks, labels = [], [], []
    features, grids = [], []
    for row, seq in zip(examples, sequences):
        n = width-len(seq)
        ids.append(seq+[tokenizer.pad_token_id]*n if training else [tokenizer.pad_token_id]*n+seq)
        masks.append([1]*len(seq)+[0]*n if training else [0]*n+[1]*len(seq))
        if training:
            labels.append(row['labels']+[-100]*n)
        if row.get('image'):
            features.append(cached_features(str(feature_path(row))))
            grids.append(row['image_grid_thw'])
    batch = {'input_ids': torch.tensor(ids, device=device), 'attention_mask': torch.tensor(masks, device=device)}
    if training:
        batch['labels'] = torch.tensor(labels, device=device)
    if features:
        batch.update(pixel_values=torch.cat(features).to(device), image_grid_thw=torch.tensor(grids, device=device))
    return batch

def fused_embeddings(model, batch):
    embedded = model.get_input_embeddings()(batch['input_ids'])
    if 'pixel_values' in batch:
        mask = (batch['input_ids'] == model.config.image_token_id).unsqueeze(-1).expand_as(embedded)
        embedded = embedded.masked_scatter(mask, batch['pixel_values'].to(embedded.dtype))
    return embedded

def answer_loss(model, batch):
    positions, _ = model.get_rope_index(batch['input_ids'], batch.get('image_grid_thw'), attention_mask=batch['attention_mask'])
    output = model.model(inputs_embeds=fused_embeddings(model, batch), position_ids=positions,
                         attention_mask=batch['attention_mask'], use_cache=False, return_dict=True)
    losses = []
    for hidden, labels in zip(output.last_hidden_state, batch['labels']):
        valid = labels[1:] != -100
        # Identical masked CE, but avoid a 151936-way projection for ignored prompt positions.
        logits = model.lm_head(hidden[:-1][valid]).float()
        losses.append(torch.nn.functional.cross_entropy(logits, labels[1:][valid]))
    return torch.stack(losses).mean()

def prompt_pools(method, examples, tokenizer):
    device = method.model.get_input_embeddings().weight.device
    batch = batch_tensors(examples, tokenizer, device, False)
    with torch.no_grad():
        embeds = fused_embeddings(method.model, batch).float()
        return embeds.masked_fill(~batch['attention_mask'].bool().unsqueeze(-1), -torch.inf).amax(dim=1)

def install_sapt(method):
    if not hasattr(method, 'router'):
        return
    import types
    def prepare(self, examples, tokenizer, batch, training):
        self.tokenizer = tokenizer
        if self.task_count:
            weights = self.router(prompt_pools(self, examples, tokenizer)).softmax(-1)
            for layer in self.layers.values():
                layer.attention = weights
    def penalty(self, orthogonal_weight, l2_weight):
        loss = 0.0
        for memory in self.memory:
            count = self.config['sapt_replay_batch_size']
            pool = memory['pools']
            indices = [(self.update*count+i) % len(pool) for i in range(count)]
            logits = self.router(pool[indices].to(self.router.keys[0].device))
            target = torch.nn.functional.pad(logits.new_tensor(memory['attention']), (0, self.task_count-len(memory['attention'])))
            loss = loss + torch.nn.functional.kl_div(logits.log_softmax(-1), target.expand(count,-1), reduction='batchmean')
        return loss*self.config['sapt_kl_weight']
    method.prepare_batch = types.MethodType(prepare, method)
    method.penalty = types.MethodType(penalty, method)

def generate(model, method, tokenizer, rows, config, sample=False):
    batch = batch_tensors(rows, tokenizer, model.device, False)
    if hasattr(method, 'prepare_batch'):
        method.prepare_batch(rows, tokenizer, batch, False)
    with torch.inference_mode():
        output = model.generate(**batch, do_sample=sample, temperature=1.0, top_p=.9 if sample else 1.0,
            top_k=0 if sample else 50, num_beams=1, max_new_tokens=config['target_tokens'], use_cache=True,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    tokens = output[:, batch['input_ids'].shape[1]:].tolist()
    for i, row in enumerate(tokens):
        if tokenizer.eos_token_id in row:
            tokens[i] = row[:row.index(tokenizer.eos_token_id)+1]
    return tokens, tokenizer.batch_decode(tokens, skip_special_tokens=True)

def evaluate(model, tokenizer, examples, config, output, metadata, method=None, resume=False):
    model.eval()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.jsonl.tmp')
    records = []
    source = output if output.exists() else tmp
    if resume and source.exists():
        lines = source.read_text().splitlines()
        for index, line in enumerate(lines):
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                if source == tmp and index == len(lines)-1:
                    break
                raise
            assert record['instance_id'] == examples[index]['instance_id']
            assert record['references'] == examples[index]['references']
            assert all(record.get(k) == v for k,v in metadata.items())
            records.append(record)
    if source == output and resume:
        assert len(records) == len(examples)
    elif records:
        records = records[:len(records)//config['eval_batch_size']*config['eval_batch_size']]
    # Atomic reconstruction protects valid rows if interrupted during resume itself.
    rebuild = output.with_suffix('.rebuild')
    rebuild.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records))
    rebuild.replace(tmp)
    scorer = Scorer()
    started = time.monotonic()
    with tmp.open('a') as handle:
        for start in range(len(records), len(examples), config['eval_batch_size']):
            rows = examples[start:start+config['eval_batch_size']]
            tick = time.monotonic()
            tokens, texts = generate(model, method, tokenizer, rows, config)
            elapsed = time.monotonic()-tick
            for row, ids, text in zip(rows, tokens, texts):
                scores = scorer.score(text, row['references'])
                record = {**metadata, **{k: row[k] for k in ['task_id', 'instance_id', 'prompt', 'rendered_prompt',
                    'references', 'image', 'image_metadata', 'image_grid_thw', 'image_tokens', 'answer_type', 'organ',
                    'prompt_truncated', 'target_truncated'] if k in row},
                    'prompt_token_ids': row['prompt_ids'], 'generated_token_ids': ids, 'prediction': text,
                    'scores': scores, 'generation_batch_seconds': elapsed, 'generation_batch_size': len(rows),
                    'hit_generation_limit': len(ids) == config['target_tokens'] and ids[-1] != tokenizer.eos_token_id}
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
            print(json.dumps({'event': 'evaluation_progress', **metadata, 'task': examples[0]['task_id'],
                'done': start+len(rows), 'total': len(examples), 'batch_seconds': elapsed}), flush=True)
    tmp.replace(output)
    totals = {k: sum(r['scores'][k] for r in records)/len(records) for k in ['rougeL','exact_match','token_f1']}
    groups = {}
    for field in ['answer_type', 'organ']:
        values = {r[field] for r in records if field in r}
        for value in values:
            group = [r for r in records if r.get(field) == value]
            groups[field+':'+str(value)] = {'count': len(group), **{k: sum(r['scores'][k] for r in group)/len(group) for k in totals}}
    result = {**totals, 'count': len(records), 'seconds_this_attempt': time.monotonic()-started,
        'groups': groups, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}
    write_json(output.with_suffix('.scores.json'), result)
    return result
