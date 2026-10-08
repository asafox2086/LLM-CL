"""SAPT-LoRA with shared attention and generative attentive reflection.

Reference: SAPT (ACL 2024), sections 4.2–4.3, and archived llama_prompt.py.
The attention target is computed on current training prompts to avoid using
test inputs for training. Auxiliary generator costs are recorded separately.
"""

import json
import math
import time

from common import encode_prompt, tokenize_examples

import torch
from torch import nn
from torch.nn import functional


def new_adapter(base, rank):
    adapter = nn.ModuleDict({
        'down': nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32),
        'up': nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32),
    })
    nn.init.kaiming_uniform_(adapter['down'].weight, a=math.sqrt(5))
    nn.init.zeros_(adapter['up'].weight)
    return adapter


class AttentiveLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ModuleList()
        self.generator = None
        self.generator_mode = False
        self.attention = None

    def forward(self, inputs):
        output = self.base(inputs)
        if not self.adapters and not self.generator_mode:
            return output
        dropped = self.dropout(inputs).float()
        if self.generator_mode:
            delta = self.generator['up'](self.generator['down'](dropped))
            return output + delta.to(output.dtype) * self.scale
        if self.attention is None:
            raise RuntimeError('SAPT needs prompt-only routing before each batch')
        for index, adapter in enumerate(self.adapters):
            delta = adapter['up'](adapter['down'](dropped))
            output = output + (delta * self.attention[:, index, None, None]).to(output.dtype) * self.scale
        return output


class SharedAttention(nn.Module):
    def __init__(self, hidden_size, bottleneck, device):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, bottleneck, bias=False, device=device, dtype=torch.float32),
            nn.Linear(bottleneck, hidden_size, bias=False, device=device, dtype=torch.float32),
            nn.SiLU(), nn.LayerNorm(hidden_size, device=device, dtype=torch.float32))
        self.keys = nn.ParameterList()
        self.temperature = math.sqrt(hidden_size)

    def forward(self, pooled):
        with torch.autocast(device_type=pooled.device.type, enabled=False):
            query = self.projection(pooled.float())
            return functional.linear(query, torch.stack(list(self.keys))) / self.temperature


class SAPTLoRA:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        model.requires_grad_(False)
        self.layers = {}
        for name, module in list(model.named_modules()):
            if name.rsplit('.', 1)[-1] in config['targets']:
                parent, child = name.rsplit('.', 1)
                wrapper = AttentiveLinear(module, config['rank'], config['alpha'], config['dropout'])
                setattr(model.get_submodule(parent), child, wrapper)
                self.layers[name] = wrapper
        if not self.layers:
            raise ValueError('No LoRA target layers found')
        embedding = model.get_input_embeddings()
        self.router = SharedAttention(embedding.embedding_dim, config['sapt_router_dim'], embedding.weight.device)
        model.add_module('sapt_router', self.router)
        self.memory = []
        self.update = 0
        self.tokenizer = None

    @property
    def task_count(self):
        return len(self.router.keys)

    def begin_task(self):
        self.model.requires_grad_(False)
        self.router.projection.requires_grad_(True)
        device = self.model.get_input_embeddings().weight.device
        key = nn.Parameter(torch.empty(self.model.config.hidden_size, device=device, dtype=torch.float32))
        nn.init.uniform_(key, -1, 1)
        self.router.keys.append(key)
        for layer in self.layers.values():
            layer.adapters.append(new_adapter(layer.base, layer.rank))

    def parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def pooled_inputs(self, sequences):
        device = self.model.get_input_embeddings().weight.device
        width = max(map(len, sequences))
        tokens = torch.tensor([sequence + [self.tokenizer.pad_token_id] * (width - len(sequence))
                               for sequence in sequences], device=device)
        valid = torch.arange(width, device=device)[None, :] < torch.tensor(list(map(len, sequences)), device=device)[:, None]
        with torch.no_grad():
            embedded = self.model.get_input_embeddings()(tokens).float()
            return embedded.masked_fill(~valid.unsqueeze(-1), -torch.inf).amax(dim=1)

    def prepare_batch(self, examples, tokenizer, batch, training):
        self.tokenizer = tokenizer
        if not self.task_count:
            return
        weights = self.router(self.pooled_inputs([example['prompt_ids'] for example in examples])).softmax(dim=-1)
        for layer in self.layers.values():
            layer.attention = weights

    def before_update(self, update):
        self.update = update

    def penalty(self, orthogonal_weight, l2_weight):
        loss = 0.0
        for memory in self.memory:
            count = self.config['sapt_replay_batch_size']
            prompts = memory['prompt_ids']
            chosen = [prompts[(self.update * count + offset) % len(prompts)] for offset in range(count)]
            logits = self.router(self.pooled_inputs(chosen))
            target = logits.new_tensor(memory['attention'])
            target = functional.pad(target, (0, self.task_count - len(target)))
            loss = loss + functional.kl_div(logits.log_softmax(dim=-1), target.expand(len(chosen), -1), reduction='batchmean')
        return loss * self.config['sapt_kl_weight']

    def save(self, path):
        torch.save({'task_count': self.task_count, 'memory': self.memory,
                    'router': {name: value.detach().cpu() for name, value in self.router.state_dict().items()},
                    'layers': {name: {key: value.detach().cpu() for key, value in layer.adapters.state_dict().items()}
                               for name, layer in self.layers.items()}}, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if self.task_count > state['task_count'] or set(state['layers']) != set(self.layers):
            raise ValueError('SAPT checkpoint layout mismatch')
        while self.task_count < state['task_count']:
            self.begin_task()
        self.router.load_state_dict(state['router'], strict=True)
        self.memory = state['memory']
        for name, layer in self.layers.items():
            layer.adapters.load_state_dict(state['layers'][name], strict=True)

    def finish_task(self, tokenizer, examples, destination, train_function):
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer = tokenizer
        self.model.eval()
        started = time.monotonic()
        with torch.inference_mode():
            attention_sum = torch.zeros(self.task_count, device=self.router.keys[0].device)
            for start in range(0, len(examples), 16):
                sequences = [example['prompt_ids'] for example in examples[start:start + 16]]
                attention_sum += self.router(self.pooled_inputs(sequences)).softmax(dim=-1).sum(dim=0)
            target = (attention_sum / len(examples)).cpu().tolist()
        generator = GeneratorMethod(self)
        prompt_ids = encode_prompt(tokenizer, '[Gen]', self.config)
        reconstructed = []
        for example in examples:
            text = example['prompt'].split('\n\nInput: ', 1)[1].rsplit('\n\nResponse:\n', 1)[0]
            tokenized, _ = tokenize_examples(tokenizer, [{'prompt': '[Gen]', 'references': [text],
                'instance_id': example.get('instance_id'), 'task_id': example['task_id']}], self.config)
            reconstructed.extend(tokenized)
        auxiliary_config = {**self.config, 'epochs': self.config['sapt_generator_epochs'], 'checkpoint_selection': 'last_epoch'}
        device = self.model.get_input_embeddings().weight.device
        devices = [device.index] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.config['seed'] + 10000 + self.task_count)
            resources = train_function(generator, tokenizer, {'train': reconstructed}, auxiliary_config,
                                       destination / 'generator', f'generator_task_{self.task_count}')
            self.model.eval()
            count = 2 if self.config.get('smoke') else self.config['sapt_pseudo_samples']
            generated_inputs = []
            attempts = 0
            while len(generated_inputs) < count and attempts < count * 4:
                batch_size = min(self.config['eval_batch_size'], count - len(generated_inputs))
                tokens = torch.tensor([prompt_ids] * batch_size, device=device)
                with torch.inference_mode():
                    generated = self.model.generate(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                                    do_sample=True, top_p=0.9, temperature=1.0, num_beams=1,
                                                    max_new_tokens=self.config['target_tokens'], use_cache=True,
                                                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                attempts += batch_size
                start_token = 1 if self.model.config.is_encoder_decoder else len(prompt_ids)
                generated_inputs.extend(text.strip() for text in tokenizer.batch_decode(generated[:, start_token:], skip_special_tokens=True)
                                        if text.strip())
                print(json.dumps({'event': 'sapt_pseudo_generation', 'stage': self.task_count,
                                  'done': len(generated_inputs), 'total': count}), flush=True)
            if len(generated_inputs) != count:
                raise RuntimeError('SAPT generator produced too few nonempty pseudo inputs')
        generator.release()
        instruction = examples[0]['prompt'].split('\n\nInput: ', 1)[0]
        pseudo_prompts = [f'{instruction}\n\nInput: {text}\n\nResponse:\n' for text in generated_inputs]
        replay_ids = [encode_prompt(tokenizer, prompt, self.config) for prompt in pseudo_prompts]
        self.memory.append({'task_id': examples[0]['task_id'], 'prompt_ids': replay_ids, 'attention': target})
        (destination / 'pseudo_inputs.json').write_text(json.dumps(generated_inputs, ensure_ascii=False, indent=2) + '\n')
        resources.update({'pseudo_count': count, 'unique_pseudo_count': len(set(generated_inputs)),
                          'generation_attempts': attempts, 'attention_target': target, 'attention_target_source': 'current_train_only',
                          'wall_seconds_including_generation': time.monotonic() - started})
        (destination / 'resources.json').write_text(json.dumps(resources, indent=2) + '\n')
        temporary = destination / 'stage_state.pt.tmp'
        self.save(temporary)
        temporary.replace(destination / 'stage_state.pt')
        return resources


class GeneratorMethod:
    def __init__(self, parent):
        self.parent = parent
        self.model = parent.model
        self.layers = parent.layers

    def begin_task(self):
        self.model.requires_grad_(False)
        for layer in self.layers.values():
            layer.attention = None
            layer.generator = new_adapter(layer.base, layer.rank)
            layer.generator_mode = True

    def parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def penalty(self, orthogonal_weight, l2_weight):
        return 0.0

    def save(self, path):
        torch.save({name: {key: value.detach().cpu() for key, value in layer.generator.state_dict().items()}
                    for name, layer in self.layers.items()}, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        for name, layer in self.layers.items():
            layer.generator.load_state_dict(state[name], strict=True)

    def release(self):
        for layer in self.layers.values():
            layer.generator_mode = False
            layer.generator = None
            layer.adapters[-1].requires_grad_(True)
        self.parent.router.projection.requires_grad_(True)
        self.parent.router.keys[-1].requires_grad_(True)
