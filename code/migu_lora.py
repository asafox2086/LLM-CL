"""MIGU-LoRA: fixed-size LoRA with magnitude-guided gradient masks.

Reference: Unlocking Continual Learning Abilities in Language Models,
sections 3.2–3.3, and the archived accelerate_local.py gradient masking.
"""

import math

import torch
from torch import nn


class MagnitudeLinear(nn.Module):
    def __init__(self, base, rank, alpha, dropout):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.adapters = nn.ModuleList()
        self.record = False
        self.token_mask = None
        self.magnitudes = {}

    def initialize(self):
        adapter = nn.ModuleDict({
            'down': nn.Linear(self.base.in_features, self.rank, bias=False, device=self.base.weight.device, dtype=torch.float32),
            'up': nn.Linear(self.rank, self.base.out_features, bias=False, device=self.base.weight.device, dtype=torch.float32),
        })
        nn.init.kaiming_uniform_(adapter['down'].weight, a=math.sqrt(5))
        nn.init.zeros_(adapter['up'].weight)
        self.adapters.append(adapter)

    def forward(self, inputs):
        output = self.base(inputs)
        if not self.adapters:
            return output
        adapter = self.adapters[0]
        hidden = adapter['down'](self.dropout(inputs).to(adapter['down'].weight.dtype))
        output = output + adapter['up'](hidden).to(output.dtype) * self.scale
        if self.record and self.training:
            with torch.no_grad():
                for name, values in [('down', hidden), ('up', output)]:
                    magnitude = values.detach().float().abs()
                    magnitude = (magnitude * self.token_mask.unsqueeze(-1)).sum(dim=(0, 1))
                    self.magnitudes[name] = self.magnitudes.get(name, 0) + magnitude
        return output


class MIGULoRA:
    def __init__(self, model, config):
        self.model = model
        self.threshold = config['migu_mask_ratio']
        if not 0 <= self.threshold < 1:
            raise ValueError('MIGU mask ratio must be in [0, 1)')
        self.task_count = 0
        model.requires_grad_(False)
        self.layers = {}
        for name, module in list(model.named_modules()):
            if name.rsplit('.', 1)[-1] in config['targets']:
                parent, child = name.rsplit('.', 1)
                wrapper = MagnitudeLinear(module, config['rank'], config['alpha'], config['dropout'])
                setattr(model.get_submodule(parent), child, wrapper)
                self.layers[name] = wrapper
        if not self.layers:
            raise ValueError('No LoRA target layers found')

    def begin_task(self):
        if not self.task_count:
            for layer in self.layers.values():
                layer.initialize()
        self.task_count += 1

    def parameters(self):
        return [parameter for parameter in self.model.parameters() if parameter.requires_grad]

    def penalty(self, orthogonal_weight, l2_weight):
        return 0.0

    def before_update(self, update):
        for layer in self.layers.values():
            layer.magnitudes = {}

    def prepare_batch(self, examples, tokenizer, batch, training):
        for name, layer in self.layers.items():
            # Cross-attention V reads encoder states; its Q reads decoder states.
            decoder_tokens = (self.model.config.is_encoder_decoder and name.startswith('decoder.')
                              and not ('.EncDecAttention.' in name and name.endswith('.v')))
            mask_name = 'decoder_attention_mask' if training and decoder_tokens else 'attention_mask'
            layer.token_mask = batch[mask_name].bool()
            layer.record = training and self.task_count > 1

    def after_forward(self):
        for layer in self.layers.values():
            layer.record = False

    def before_step(self):
        if self.task_count < 2:
            return
        for layer in self.layers.values():
            for name, linear in layer.adapters[0].items():
                magnitude = layer.magnitudes[name]
                mask = magnitude >= torch.quantile(magnitude, self.threshold)
                if linear.weight.grad is None:
                    raise RuntimeError('Missing LoRA gradient')
                linear.weight.grad.mul_(mask.unsqueeze(-1))

    def save(self, path):
        torch.save({'task_count': self.task_count,
                    'layers': {name: {key: value.detach().cpu() for key, value in layer.adapters.state_dict().items()}
                               for name, layer in self.layers.items()}}, path)

    def load(self, path):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if not self.task_count:
            self.begin_task()
        if set(state['layers']) != set(self.layers):
            raise ValueError('LoRA checkpoint target mismatch')
        self.task_count = state['task_count']
        for name, layer in self.layers.items():
            layer.adapters.load_state_dict(state['layers'][name], strict=True)
